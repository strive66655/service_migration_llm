from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

from .client import query_llm
from .schema import DEFAULT_SAFE_CONTROL, SafeControlParams
from .validator import validate_llm_output

_DISTANCE_TRENDS = {"moving_away", "stable", "approaching"}
_MOBILITY_LEVELS = {"low", "medium", "high"}
_STABILITY_RISKS = {"low", "high"}


@dataclass(slots=True)
class SharedControlState:
    current_state: dict[str, Any]
    recent_history: dict[str, Any]
    business_profile: str
    operator_text: str
    previous_solver_mode: str
    recent_service_distances: list[int]
    recent_migrations: list[int]


@dataclass(slots=True)
class ForecastOutput:
    distance_trend: str
    mobility_level: str
    stability_risk: str
    reason: str
    used_fallback: bool = False
    validation_notes: tuple[str, ...] = ()


@dataclass(slots=True)
class PolicyAdviceOutput:
    objective_mode: str
    gamma: float
    migration_weight: float
    transmission_weight: float
    solver_mode: str
    reason: str
    used_fallback: bool = False
    validation_notes: tuple[str, ...] = ()


def build_shared_control_state(state: dict[str, Any], history: dict[str, Any], business_profile: str, operator_text: str | None) -> dict[str, Any]:
    recent_service_distances = [int(x) for x in history.get("recent_service_distances", [])]
    recent_migrations = [int(bool(x)) for x in history.get("recent_migrations", [])]
    previous_distance = recent_service_distances[-1] if recent_service_distances else int(state["distance_to_user"])
    distance_delta = int(state["distance_to_user"]) - previous_distance
    recent_direction = int(state.get("recent_direction", 0))
    return asdict(
        SharedControlState(
            current_state={
                "state_index": int(state["state_index"]),
                "service_index": int(state["service_index"]),
                "distance_to_user": int(state["distance_to_user"]),
                "recent_direction": recent_direction,
                "distance_delta": distance_delta,
                "predicted_moving_away": distance_delta > 0,
                "high_mobility_hint": abs(distance_delta) >= 2 or abs(recent_direction) >= 2,
            },
            recent_history={
                "migration_count_recent": int(sum(recent_migrations)),
                "average_service_distance_recent": float(sum(recent_service_distances) / len(recent_service_distances)) if recent_service_distances else float(state["distance_to_user"]),
            },
            business_profile=business_profile,
            operator_text=operator_text or "",
            previous_solver_mode=str(history.get("previous_solver_mode", "mdp")),
            recent_service_distances=recent_service_distances,
            recent_migrations=recent_migrations,
        )
    )


def build_forecaster_prompt(shared_control_state: dict[str, Any]) -> str:
    required_schema = {
        "distance_trend": "moving_away | stable | approaching",
        "mobility_level": "low | medium | high",
        "stability_risk": "low | high",
        "reason": "short explanation",
    }
    return (
        "You summarize short-horizon mobility for service migration.\n"
        "Return only a JSON object that matches the required schema.\n"
        "Keep the labels coarse and conservative.\n"
        f"Required schema:\n{json.dumps(required_schema, ensure_ascii=False)}\n"
        f"STATE JSON:\n{json.dumps(shared_control_state, ensure_ascii=False)}"
    )


def build_policy_advisor_prompt(shared_control_state: dict[str, Any], forecast: dict[str, Any]) -> str:
    required_schema = {
        "objective_mode": "latency_first | stability_first | balanced",
        "gamma": "float in [0.7, 0.99]",
        "migration_weight": "float in [0.5, 1.8]",
        "transmission_weight": "float in [0.5, 1.8]",
        "solver_mode": "threshold | myopic | mdp",
        "reason": "short explanation",
    }
    return (
        "You recommend service migration control parameters.\n"
        "Use the mobility summary conservatively and return only a JSON object matching the required schema.\n"
        f"Required schema:\n{json.dumps(required_schema, ensure_ascii=False)}\n"
        f"STATE JSON:\n{json.dumps({'shared_control_state': shared_control_state, 'forecast': forecast}, ensure_ascii=False)}"
    )


def _validate_forecast_output(output: dict[str, Any] | None) -> ForecastOutput:
    notes: list[str] = []
    if output is None:
        return ForecastOutput("stable", "low", "low", "forecast fallback default", used_fallback=True, validation_notes=("default_fallback",))
    if "payload" in output and len(output) == 1:
        return ForecastOutput("stable", "low", "low", "forecast fallback invalid payload", used_fallback=True, validation_notes=("invalid_payload",))

    distance_trend = str(output.get("distance_trend", "stable"))
    if distance_trend not in _DISTANCE_TRENDS:
        notes.append("distance_trend_fallback")
        distance_trend = "stable"

    mobility_level = str(output.get("mobility_level", "low"))
    if mobility_level not in _MOBILITY_LEVELS:
        notes.append("mobility_level_fallback")
        mobility_level = "low"

    stability_risk = str(output.get("stability_risk", "low"))
    if stability_risk not in _STABILITY_RISKS:
        notes.append("stability_risk_fallback")
        stability_risk = "low"

    reason = str(output.get("reason", "forecast fallback partial")).strip() or "forecast fallback partial"
    return ForecastOutput(
        distance_trend=distance_trend,
        mobility_level=mobility_level,
        stability_risk=stability_risk,
        reason=reason,
        used_fallback=bool(notes),
        validation_notes=tuple(notes),
    )


def _coerce_policy_advice(output: dict[str, Any] | None) -> PolicyAdviceOutput:
    validated = validate_llm_output(output, DEFAULT_SAFE_CONTROL)
    return PolicyAdviceOutput(
        objective_mode=validated.objective_mode,
        gamma=validated.gamma,
        migration_weight=validated.migration_weight,
        transmission_weight=validated.transmission_weight,
        solver_mode=validated.solver_mode,
        reason=validated.reason,
        used_fallback=validated.used_fallback,
        validation_notes=validated.validation_notes,
    )


def _apply_forecast_rules(base: dict[str, Any], forecast: ForecastOutput) -> dict[str, Any]:
    draft = dict(base)
    if forecast.distance_trend == "moving_away":
        draft["transmission_weight"] = max(float(draft["transmission_weight"]), 1.15)
        draft["gamma"] = max(float(draft["gamma"]), 0.9)
        if forecast.mobility_level == "high":
            draft["solver_mode"] = "threshold"
    elif forecast.distance_trend == "approaching":
        draft["transmission_weight"] = min(float(draft["transmission_weight"]), 1.0)

    if forecast.mobility_level == "high":
        draft["gamma"] = max(float(draft["gamma"]), 0.93)
        draft["solver_mode"] = "threshold"
    elif forecast.mobility_level == "low" and draft["solver_mode"] == "threshold":
        draft["solver_mode"] = "mdp"

    if forecast.stability_risk == "high":
        draft["migration_weight"] = max(float(draft["migration_weight"]), 1.2)
        if draft["objective_mode"] == "latency_first":
            draft["objective_mode"] = "balanced"
        if draft["solver_mode"] == "myopic":
            draft["solver_mode"] = "threshold"

    return draft


def _agreement_level(forecast: ForecastOutput, advice: PolicyAdviceOutput) -> str:
    score = 0
    if forecast.distance_trend == "moving_away" and advice.transmission_weight >= 1.1:
        score += 1
    if forecast.stability_risk == "high" and advice.migration_weight >= 1.1:
        score += 1
    if forecast.mobility_level == "high" and advice.solver_mode == "threshold":
        score += 1
    if score >= 3:
        return "high"
    if score == 2:
        return "medium"
    return "low"


def _decision_source(forecast: ForecastOutput, policy_raw: dict[str, Any] | None, final_control: SafeControlParams) -> str:
    if policy_raw is None:
        return "fallback_default"
    if final_control.used_fallback or forecast.used_fallback:
        return "fallback_partial"
    return "multi_agent_merged"


def query_multi_agent_control(
    shared_control_state: dict[str, Any],
    *,
    failure_mode: str | None = None,
    backend: str = "mock",
    model: str = "openai/gpt-5.3-chat",
    api_base: str = "https://openrouter.ai/api/v1",
    api_key_env: str = "OPENROUTER_API_KEY",
    timeout_sec: float = 30.0,
    agent_models: dict[str, str] | None = None,
    agent_backends: dict[str, str] | None = None,
) -> dict[str, Any]:
    models = agent_models or {}
    backends = agent_backends or {}

    forecaster_prompt = build_forecaster_prompt(shared_control_state)
    forecaster_started = perf_counter()
    try:
        forecaster_raw = query_llm(
            forecaster_prompt,
            state=shared_control_state,
            failure_mode=failure_mode,
            backend=backends.get("forecaster", backend),
            model=models.get("forecaster", model),
            api_base=api_base,
            api_key_env=api_key_env,
            timeout_sec=timeout_sec,
            schema_name="forecast",
        )
    except TimeoutError:
        forecaster_raw = None
    forecaster_latency_ms = (perf_counter() - forecaster_started) * 1000.0
    forecast = _validate_forecast_output(forecaster_raw)

    policy_prompt = build_policy_advisor_prompt(shared_control_state, asdict(forecast))
    policy_started = perf_counter()
    try:
        policy_raw = query_llm(
            policy_prompt,
            state={
                "current_state": shared_control_state["current_state"],
                "history": {
                    "recent_service_distances": shared_control_state["recent_service_distances"],
                    "recent_migrations": shared_control_state["recent_migrations"],
                    "migration_count_recent": shared_control_state["recent_history"]["migration_count_recent"],
                    "average_service_distance_recent": shared_control_state["recent_history"]["average_service_distance_recent"],
                    "previous_solver_mode": shared_control_state["previous_solver_mode"],
                },
                "business_profile": shared_control_state["business_profile"],
                "operator_text": shared_control_state["operator_text"],
            },
            failure_mode=failure_mode,
            backend=backends.get("policy_advisor", backend),
            model=models.get("policy_advisor", model),
            api_base=api_base,
            api_key_env=api_key_env,
            timeout_sec=timeout_sec,
            schema_name="policy_advice",
        )
    except TimeoutError:
        policy_raw = None
    policy_latency_ms = (perf_counter() - policy_started) * 1000.0
    policy_advice = _coerce_policy_advice(policy_raw)

    draft_control = _apply_forecast_rules(
        {
            "objective_mode": policy_advice.objective_mode,
            "gamma": policy_advice.gamma,
            "migration_weight": policy_advice.migration_weight,
            "transmission_weight": policy_advice.transmission_weight,
            "solver_mode": policy_advice.solver_mode,
            "reason": policy_advice.reason,
        },
        forecast,
    )
    final_control = validate_llm_output(draft_control, DEFAULT_SAFE_CONTROL)
    fallback_notes = [*forecast.validation_notes, *policy_advice.validation_notes, *final_control.validation_notes]
    fallback_used = bool(fallback_notes)

    return {
        "shared_control_state": shared_control_state,
        "forecaster_raw_output": forecaster_raw,
        "forecaster_output": asdict(forecast),
        "policy_advisor_raw_output": policy_raw,
        "policy_advisor_output": asdict(policy_advice),
        "draft_control": draft_control,
        "final_safe_control": asdict(final_control),
        "validation_notes": list(final_control.validation_notes),
        "fallback_used": fallback_used,
        "fallback_reason": ", ".join(fallback_notes) if fallback_notes else "",
        "final_decision_source": _decision_source(forecast, policy_raw, final_control),
        "agent_agreement": _agreement_level(forecast, policy_advice),
        "agent_metrics": {
            "forecaster": {"latency_ms": forecaster_latency_ms, "call_count": 1},
            "policy_advisor": {"latency_ms": policy_latency_ms, "call_count": 1},
            "safety_arbiter": {"latency_ms": 0.0, "call_count": 1},
        },
    }
