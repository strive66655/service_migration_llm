from __future__ import annotations

import json
import os
import time
from json import JSONDecodeError
from typing import Any

import requests
from requests import RequestException


_OPENROUTER_SCHEMAS = {
    "control": {
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
                    "minimum": 0.5,
                    "maximum": 0.99,
                    "description": "Discount factor for the lower-level MDP. Larger values emphasize longer-term consequences.",
                },
                "migration_weight": {
                    "type": "number",
                    "minimum": 0.5,
                    "maximum": 1.8,
                    "description": "Multiplicative scale applied to the baseline migration-cost terms. Values above 1 make migration more expensive; values below 1 make migration cheaper. This is not the paper's structural parameter -beta_l.",
                },
                "transmission_weight": {
                    "type": "number",
                    "minimum": 0.5,
                    "maximum": 1.8,
                    "description": "Multiplicative scale applied to the baseline transmission-cost terms. Values above 1 penalize service-user distance more strongly.",
                },
                "reason": {
                    "type": "string",
                    "description": "Short explanation of the control decision.",
                },
            },
            "required": ["objective_mode", "gamma", "migration_weight", "transmission_weight", "reason"],
            "additionalProperties": False,
        },
    },
    "forecast": {
        "name": "single_user_service_migration_forecast",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "distance_trend": {
                    "type": "string",
                    "enum": ["moving_away", "stable", "approaching"],
                },
                "mobility_level": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                },
                "stability_risk": {
                    "type": "string",
                    "enum": ["low", "high"],
                },
                "reason": {
                    "type": "string",
                    "description": "Short explanation of the mobility summary.",
                },
            },
            "required": ["distance_trend", "mobility_level", "stability_risk", "reason"],
            "additionalProperties": False,
        },
    },
    "policy_advice": {
        "name": "single_user_service_migration_policy_advice",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "objective_mode": {
                    "type": "string",
                    "enum": ["latency_first", "stability_first", "balanced"],
                },
                "gamma": {
                    "type": "number",
                    "minimum": 0.5,
                    "maximum": 0.99,
                    "description": "Discount factor for the lower-level MDP. Larger values emphasize longer-term consequences.",
                },
                "migration_weight": {
                    "type": "number",
                    "minimum": 0.5,
                    "maximum": 1.8,
                    "description": "Multiplicative scale applied to the baseline migration-cost terms. Values above 1 make migration more expensive; values below 1 make migration cheaper. This is not the paper's structural parameter -beta_l.",
                },
                "transmission_weight": {
                    "type": "number",
                    "minimum": 0.5,
                    "maximum": 1.8,
                    "description": "Multiplicative scale applied to the baseline transmission-cost terms. Values above 1 penalize service-user distance more strongly.",
                },
                "reason": {
                    "type": "string",
                    "description": "Short explanation of the control recommendation.",
                },
            },
            "required": ["objective_mode", "gamma", "migration_weight", "transmission_weight", "reason"],
            "additionalProperties": False,
        },
    },
}

_SYSTEM_PROMPTS = {
    "control": "You are an MDP meta-controller for single-user service migration. Return only valid JSON that matches the provided schema.",
    "forecast": "You summarize short-horizon mobility state for service migration. Keep labels coarse and conservative, and return only valid JSON that matches the provided schema.",
    "policy_advice": "You recommend safe lower-level MDP control parameters from a summarized state and forecast. Return only valid JSON that matches the provided schema.",
}


def _extract_state_from_prompt(prompt: str) -> dict[str, Any]:
    marker = "STATE JSON:\n"
    if marker not in prompt:
        raise ValueError("Prompt does not contain STATE JSON marker.")
    return json.loads(prompt.split(marker, 1)[1])


def _mock_forecast_query(llm_state: dict[str, Any], failure_mode: str | None = None) -> dict[str, Any]:
    current_state = llm_state.get("current_state", {})
    history = llm_state.get("recent_history", {})
    distance_delta = int(current_state.get("distance_delta", 0))
    recent_migrations = int(history.get("migration_count_recent", 0))

    if distance_delta > 0:
        distance_trend = "moving_away"
    elif distance_delta < 0:
        distance_trend = "approaching"
    else:
        distance_trend = "stable"

    if abs(distance_delta) >= 2:
        mobility_level = "high"
    elif abs(distance_delta) == 1 or recent_migrations >= 2:
        mobility_level = "medium"
    else:
        mobility_level = "low"

    stability_risk = "high" if recent_migrations >= 2 or mobility_level == "high" else "low"
    response = {
        "distance_trend": distance_trend,
        "mobility_level": mobility_level,
        "stability_risk": stability_risk,
        "reason": "mock mobility summary",
    }
    if failure_mode == "invalid_enum":
        response["mobility_level"] = "extreme"
    elif failure_mode == "missing_field":
        response.pop("stability_risk", None)
    elif failure_mode == "invalid_json":
        return {"payload": "{broken json}"}
    return response


def _normalize_mock_control_state(llm_state: dict[str, Any], schema_name: str) -> dict[str, Any]:
    if schema_name == "policy_advice" and "shared_control_state" in llm_state:
        shared = llm_state.get("shared_control_state", {})
        recent_history = shared.get("recent_history", {})
        return {
            "business_profile": shared.get("business_profile", "balanced"),
            "operator_text": shared.get("operator_text", ""),
            "current_state": shared.get("current_state", {}),
            "history": {
                "recent_service_distances": shared.get("recent_service_distances", []),
                "recent_migrations": shared.get("recent_migrations", []),
                "migration_count_recent": recent_history.get("migration_count_recent", 0),
                "average_service_distance_recent": recent_history.get("average_service_distance_recent", 0.0),
            },
            "forecast": llm_state.get("forecast", {}),
        }
    return llm_state


def _mock_control_like_query(llm_state: dict[str, Any], schema_name: str, failure_mode: str | None = None) -> dict[str, Any]:
    normalized_state = _normalize_mock_control_state(llm_state, schema_name)
    business_profile = str(normalized_state.get("business_profile", "balanced"))
    operator_text = str(normalized_state.get("operator_text", ""))
    current_state = normalized_state.get("current_state", {})
    history = normalized_state.get("history", normalized_state.get("recent_history", {}))
    forecast = normalized_state.get("forecast", {})

    latency_signal = any(
        token in operator_text
        for token in (
            "AR",
            "latency",
            "low latency",
            "latency-sensitive",
            "keep the service close",
            "distance-threshold violations",
        )
    )
    stability_signal = any(
        token in operator_text
        for token in (
            "stability",
            "service stability",
            "switching jitter",
            "avoid frequent",
            "temporary",
            "temporary shift",
        )
    )
    high_mobility = bool(current_state.get("high_mobility_hint", False))
    moving_away = bool(current_state.get("predicted_moving_away", False))
    recent_migrations = int(history.get("migration_count_recent", 0))
    average_distance_recent = float(history.get("average_service_distance_recent", current_state.get("distance_to_user", 0)))
    conservative_profile = business_profile in {"delay_tolerant", "migration_sensitive", "high_stability_required"}

    objective_mode = "balanced"
    gamma = 0.9
    migration_weight = 1.0
    transmission_weight = 1.0
    reason_parts = []

    if business_profile in {"latency_sensitive"} or latency_signal:
        objective_mode = "latency_first"
        gamma = 0.96 if high_mobility else 0.94
        migration_weight = 0.7
        transmission_weight = 1.6
        reason_parts.append("latency-oriented profile")
    if business_profile in {"migration_sensitive", "high_stability_required"} or stability_signal:
        if objective_mode == "latency_first":
            objective_mode = "balanced"
            gamma = 0.88
            migration_weight = 1.1
            transmission_weight = 1.1
            reason_parts.append("conflicting stability signal balanced the request")
        else:
            objective_mode = "stability_first"
            gamma = 0.82 if recent_migrations >= 2 else 0.86
            migration_weight = 1.5
            transmission_weight = 0.85
            reason_parts.append("stability-oriented profile")
    if business_profile in {"delay_tolerant"} and objective_mode == "balanced":
        gamma = 0.86
        migration_weight = 1.2
        transmission_weight = 0.98
        reason_parts.append("delay-tolerant profile")
    if schema_name == "policy_advice":
        forecast_trend = str(forecast.get("distance_trend", "stable"))
        forecast_mobility = str(forecast.get("mobility_level", "low"))
        forecast_stability = str(forecast.get("stability_risk", "low"))
        if forecast_trend == "moving_away":
            if conservative_profile and average_distance_recent < 1.5 and recent_migrations >= 2:
                reason_parts.append("moving-away signal acknowledged but migration cooldown preserved")
            else:
                transmission_weight = max(transmission_weight, 1.15 if conservative_profile else 1.45)
                gamma = max(gamma, 0.9 if conservative_profile else 0.94)
                reason_parts.append("forecast indicates user moving away")
        if forecast_mobility == "high":
            gamma = max(gamma, 0.93)
            reason_parts.append("forecast indicates high mobility")
        if forecast_stability == "high":
            if objective_mode == "latency_first":
                migration_weight = max(migration_weight, 0.95)
            else:
                migration_weight = max(migration_weight, 1.25 if conservative_profile else 1.2)
            reason_parts.append("forecast indicates elevated stability risk")

    if conservative_profile and recent_migrations >= 2 and average_distance_recent <= 1.5:
        gamma = min(gamma, 0.86 if business_profile == "delay_tolerant" else 0.87)
        migration_weight = max(migration_weight, 1.3 if business_profile == "delay_tolerant" else 1.45)
        transmission_weight = min(transmission_weight, 1.0 if business_profile == "delay_tolerant" else 0.95)
        if objective_mode == "latency_first":
            objective_mode = "balanced"
        reason_parts.append("recent migration cooldown favored stability")

    if business_profile == "high_stability_required" and average_distance_recent <= 1.5:
        objective_mode = "stability_first"
        gamma = min(gamma, 0.87)
        migration_weight = max(migration_weight, 1.4)
        transmission_weight = min(transmission_weight, 0.95)
        reason_parts.append("stability profile enforced stronger migration budget")

    if moving_away and objective_mode == "balanced":
        if conservative_profile and recent_migrations >= 2:
            reason_parts.append("drift observed but conservative profile kept cooldown active")
        else:
            gamma = 0.92
            transmission_weight = 1.2
            reason_parts.append("user drifting away from service")

    if not reason_parts:
        reason_parts.append("balanced baseline response")

    response = {
        "objective_mode": objective_mode,
        "gamma": gamma,
        "migration_weight": migration_weight,
        "transmission_weight": transmission_weight,
        "reason": "; ".join(reason_parts),
    }
    if failure_mode == "invalid_enum":
        response["objective_mode"] = "freeform"
    elif failure_mode == "missing_field":
        response.pop("gamma", None)
    elif failure_mode == "out_of_range":
        response["gamma"] = 1.4
        response["migration_weight"] = -3.0
    elif failure_mode == "invalid_json":
        return {"payload": "{broken json}"}
    return response


def _mock_query(llm_state: dict[str, Any], schema_name: str, failure_mode: str | None = None) -> dict[str, Any]:
    if failure_mode == "timeout":
        raise TimeoutError("Mock LLM timed out.")
    if schema_name == "forecast":
        return _mock_forecast_query(llm_state, failure_mode=failure_mode)
    if schema_name in {"control", "policy_advice"}:
        return _mock_control_like_query(llm_state, schema_name=schema_name, failure_mode=failure_mode)
    raise ValueError(f"Unsupported schema_name for mock backend: {schema_name}")


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
        elif ch == "{":
            depth += 1
        elif ch == "}":
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
    schema_name: str,
    max_retries: int = 2,
    retry_delay_sec: float = 1.0,
) -> dict[str, Any]:
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {api_key_env} is not set.")
    if schema_name not in _OPENROUTER_SCHEMAS:
        raise ValueError(f"Unsupported schema_name: {schema_name}")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": _SYSTEM_PROMPTS[schema_name],
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": _OPENROUTER_SCHEMAS[schema_name],
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
    model: str = "openai/gpt-5.4-mini",
    api_base: str = "https://openrouter.ai/api/v1",
    api_key_env: str = "OPENROUTER_API_KEY",
    timeout_sec: float = 30.0,
    schema_name: str = "control",
) -> dict[str, Any]:
    llm_state = state if state is not None else _extract_state_from_prompt(prompt)
    if backend == "mock":
        return _mock_query(llm_state, schema_name=schema_name, failure_mode=failure_mode)
    if backend == "openrouter":
        if failure_mode == "timeout":
            raise TimeoutError("Injected timeout before OpenRouter request.")
        return _openrouter_query(
            prompt,
            model=model,
            api_base=api_base,
            api_key_env=api_key_env,
            timeout_sec=timeout_sec,
            schema_name=schema_name,
        )
    raise ValueError(f"Unsupported llm backend: {backend}")
