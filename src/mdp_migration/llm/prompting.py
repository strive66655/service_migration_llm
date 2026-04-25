from __future__ import annotations

import json
from typing import Any


def build_decision_summary(state: dict[str, Any], recent_service_distances: list[int], recent_migrations: list[int]) -> dict[str, int]:
    distance_threshold = int(state.get("distance_threshold", 3))
    current_distance = int(state["distance_to_user"])
    if recent_service_distances:
        distance_trend_strength = current_distance - int(recent_service_distances[0])
    else:
        distance_trend_strength = 0

    consecutive_migrations = 0
    for migrated in reversed(recent_migrations):
        if not migrated:
            break
        consecutive_migrations += 1

    return {
        "distance_threshold": distance_threshold,
        "distance_violation_count_recent": int(sum(distance > distance_threshold for distance in recent_service_distances)),
        "consecutive_migrations_recent": consecutive_migrations,
        "distance_trend_strength": int(distance_trend_strength),
    }


def build_llm_state(state: dict[str, Any], history: dict[str, Any], business_profile: str, operator_text: str | None) -> dict[str, Any]:
    recent_service_distances = [int(x) for x in history.get("recent_service_distances", [])]
    recent_migrations = [int(bool(x)) for x in history.get("recent_migrations", [])]
    previous_distance = recent_service_distances[-1] if recent_service_distances else int(state["distance_to_user"])
    distance_delta = int(state["distance_to_user"]) - previous_distance
    moving_away = distance_delta > 0
    high_mobility = abs(distance_delta) >= 2 or abs(int(state.get("recent_direction", 0))) >= 2
    decision_summary = build_decision_summary(state, recent_service_distances, recent_migrations)
    return {
        "current_state": {
            "state_index": int(state["state_index"]),
            "service_index": int(state["service_index"]),
            "distance_to_user": int(state["distance_to_user"]),
            "recent_direction": int(state.get("recent_direction", 0)),
            "predicted_moving_away": moving_away,
            "high_mobility_hint": high_mobility,
        },
        "history": {
            "recent_service_distances": recent_service_distances,
            "recent_migrations": recent_migrations,
            "migration_count_recent": int(sum(recent_migrations)),
            "average_service_distance_recent": float(sum(recent_service_distances) / len(recent_service_distances)) if recent_service_distances else float(state["distance_to_user"]),
            **decision_summary,
        },
        "business_profile": business_profile,
        "operator_text": operator_text or "",
    }


def build_prompt(llm_state: dict[str, Any]) -> str:
    required_schema = {
        "objective_mode": "latency_first | stability_first | balanced",
        "gamma": "float in [0.5, 0.99]",
        "migration_weight": "float in [0.5, 1.8]",
        "transmission_weight": "float in [0.5, 1.8]",
        "reason": "short explanation",
    }
    return (
        "You are a meta-controller for an MDP-based single-user service migration system.\n"
        "You do not directly choose migration actions; recommend bounded control parameters used to re-solve the lower-level MDP.\n"
        "Parameter interpretation: gamma is the MDP discount factor in [0.5, 0.99]; larger values emphasize longer-term consequences.\n"
        "migration_weight is a multiplicative scale applied to the baseline migration-cost terms; values above 1 make migration more expensive and values below 1 make migration cheaper. This is not the paper's structural parameter -beta_l.\n"
        "transmission_weight is a multiplicative scale applied to the baseline transmission-cost terms; values above 1 penalize service-user distance more strongly.\n"
        "Business profile guidance: latency_sensitive prioritizes shorter service distance; delay_tolerant tolerates moderate service distance to reduce unnecessary migrations; migration_sensitive prioritizes reducing migration frequency and migration distance, even if moderate service distance must be tolerated; high_stability_required strongly avoids jitter and repeated migrations; balanced seeks a compromise.\n"
        "Read the state JSON and return only a JSON object matching the required schema. Do not output actions, formulas, or any extra text.\n"
        f"Required schema:\n{json.dumps(required_schema, ensure_ascii=False)}\n"
        f"STATE JSON:\n{json.dumps(llm_state, ensure_ascii=False)}"
    )
