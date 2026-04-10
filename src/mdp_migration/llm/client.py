from __future__ import annotations

import json
import os
import time
from json import JSONDecodeError
from typing import Any

import requests
from requests import RequestException


_OPENROUTER_SCHEMA = {
    "name": "single_user_service_migration_control",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "objective_mode": {
                "type": "string",
                "enum": ["latency_first", "stability_first", "balanced"],
                "description": "High-level optimization preference.",
            },
            "gamma": {
                "type": "number",
                "minimum": 0.7,
                "maximum": 0.99,
                "description": "Discount factor for the lower-level MDP.",
            },
            "migration_weight": {
                "type": "number",
                "minimum": 0.5,
                "maximum": 1.8,
                "description": "Weight applied to migration cost terms.",
            },
            "transmission_weight": {
                "type": "number",
                "minimum": 0.5,
                "maximum": 1.8,
                "description": "Weight applied to transmission distance cost terms.",
            },
            "solver_mode": {
                "type": "string",
                "enum": ["threshold", "myopic", "mdp"],
                "description": "Lower-level solver choice.",
            },
            "reason": {
                "type": "string",
                "description": "Short explanation of the control decision.",
            },
        },
        "required": ["objective_mode", "gamma", "migration_weight", "transmission_weight", "solver_mode", "reason"],
        "additionalProperties": False,
    },
}


def _extract_state_from_prompt(prompt: str) -> dict[str, Any]:
    marker = "STATE JSON:\n"
    if marker not in prompt:
        raise ValueError("Prompt does not contain STATE JSON marker.")
    return json.loads(prompt.split(marker, 1)[1])


def _mock_query(llm_state: dict[str, Any], failure_mode: str | None = None) -> dict[str, Any]:
    if failure_mode == "timeout":
        raise TimeoutError("Mock LLM timed out.")

    business_profile = str(llm_state.get("business_profile", "balanced"))
    operator_text = str(llm_state.get("operator_text", ""))
    current_state = llm_state.get("current_state", {})
    history = llm_state.get("history", {})

    latency_signal = any(token in operator_text for token in ("低时延", "AR", "时延敏感", "latency"))
    stability_signal = any(token in operator_text for token in ("优先稳定", "减少切换", "避免频繁", "短暂", "过渡", "误导"))
    high_mobility = bool(current_state.get("high_mobility_hint", False))
    moving_away = bool(current_state.get("predicted_moving_away", False))
    recent_migrations = int(history.get("migration_count_recent", 0))

    objective_mode = "balanced"
    gamma = 0.9
    migration_weight = 1.0
    transmission_weight = 1.0
    solver_mode = "mdp"
    reason_parts = []

    if business_profile in {"latency_sensitive"} or latency_signal:
        objective_mode = "latency_first"
        gamma = 0.94 if high_mobility else 0.91
        migration_weight = 0.8
        transmission_weight = 1.45
        solver_mode = "threshold" if high_mobility else "mdp"
        reason_parts.append("latency-oriented profile")
    if business_profile in {"migration_sensitive", "high_stability_required"} or stability_signal:
        if objective_mode == "latency_first":
            objective_mode = "balanced"
            gamma = 0.88
            migration_weight = 1.1
            transmission_weight = 1.1
            solver_mode = "mdp"
            reason_parts.append("conflicting stability signal balanced the request")
        else:
            objective_mode = "stability_first"
            gamma = 0.82 if recent_migrations >= 2 else 0.86
            migration_weight = 1.5
            transmission_weight = 0.85
            solver_mode = "myopic" if recent_migrations >= 2 else "threshold"
            reason_parts.append("stability-oriented profile")
    if business_profile in {"delay_tolerant"} and objective_mode == "balanced":
        gamma = 0.84
        migration_weight = 1.2
        transmission_weight = 0.9
        solver_mode = "threshold"
        reason_parts.append("delay-tolerant profile")
    if moving_away and objective_mode == "balanced":
        gamma = 0.92
        transmission_weight = 1.2
        solver_mode = "threshold" if high_mobility else "mdp"
        reason_parts.append("user drifting away from service")

    if not reason_parts:
        reason_parts.append("balanced baseline response")

    response = {
        "objective_mode": objective_mode,
        "gamma": gamma,
        "migration_weight": migration_weight,
        "transmission_weight": transmission_weight,
        "solver_mode": solver_mode,
        "reason": "; ".join(reason_parts),
    }
    if failure_mode == "invalid_enum":
        response["solver_mode"] = "freeform"
    elif failure_mode == "missing_field":
        response.pop("gamma", None)
    elif failure_mode == "out_of_range":
        response["gamma"] = 1.4
        response["migration_weight"] = -3.0
    elif failure_mode == "invalid_json":
        return {"payload": "{broken json}"}
    return response


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        raise JSONDecodeError("No JSON object start found", text, 0)
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    raise JSONDecodeError("Unterminated JSON object", text, start)


def _parse_model_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    candidates: list[str] = [stripped]
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            candidates.append("\n".join(lines[1:-1]).strip())
    try:
        candidates.append(_extract_json_object(stripped))
    except JSONDecodeError:
        pass
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    preview = stripped[:500]
    raise RuntimeError(f"Model output is not valid JSON: {preview}")


def _openrouter_query(
    prompt: str,
    *,
    model: str,
    api_base: str,
    api_key_env: str,
    timeout_sec: float,
    max_retries: int = 2,
    retry_delay_sec: float = 1.0,
) -> dict[str, Any]:
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {api_key_env} is not set.")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You control a single-user service migration system. Return only valid JSON that matches the provided schema.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": _OPENROUTER_SCHEMA,
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    site_url = os.getenv("OPENROUTER_SITE_URL")
    app_name = os.getenv("OPENROUTER_APP_NAME")
    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_name:
        headers["X-Title"] = app_name

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                f"{api_base.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout_sec,
            )
            response.raise_for_status()
            body = response.json()
            try:
                content = body["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(f"Unexpected OpenRouter response shape: {body}") from exc
            if isinstance(content, list):
                text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            else:
                text = str(content)
            return _parse_model_json(text)
        except requests.HTTPError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            raise RuntimeError(f"OpenRouter HTTP error: {detail}") from exc
        except (RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(retry_delay_sec * (attempt + 1))

    raise RuntimeError(f"OpenRouter request failed after retries: {last_error}") from last_error


def query_llm(
    prompt: str,
    *,
    state: dict[str, Any] | None = None,
    failure_mode: str | None = None,
    backend: str = "mock",
    model: str = "openai/gpt-5.3-chat",
    api_base: str = "https://openrouter.ai/api/v1",
    api_key_env: str = "OPENROUTER_API_KEY",
    timeout_sec: float = 30.0,
) -> dict[str, Any]:
    llm_state = state if state is not None else _extract_state_from_prompt(prompt)
    if backend == "mock":
        return _mock_query(llm_state, failure_mode=failure_mode)
    if backend == "openrouter":
        if failure_mode == "timeout":
            raise TimeoutError("Injected timeout before OpenRouter request.")
        return _openrouter_query(
            prompt,
            model=model,
            api_base=api_base,
            api_key_env=api_key_env,
            timeout_sec=timeout_sec,
        )
    raise ValueError(f"Unsupported llm backend: {backend}")
