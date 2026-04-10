from __future__ import annotations

import json
from typing import Any


def build_llm_state(state: dict[str, Any], history: dict[str, Any], business_profile: str, operator_text: str | None) -> dict[str, Any]:
    recent_service_distances = [int(x) for x in history.get("recent_service_distances", [])]
    recent_migrations = [int(bool(x)) for x in history.get("recent_migrations", [])]
    previous_distance = recent_service_distances[-1] if recent_service_distances else int(state["distance_to_user"])
    distance_delta = int(state["distance_to_user"]) - previous_distance
    moving_away = distance_delta > 0
    high_mobility = abs(distance_delta) >= 2 or abs(int(state.get("recent_direction", 0))) >= 2
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
            "previous_solver_mode": history.get("previous_solver_mode", "mdp"),
        },
        "business_profile": business_profile,
        "operator_text": operator_text or "",
    }


def build_prompt(llm_state: dict[str, Any]) -> str:
    required_schema = {
        "objective_mode": "latency_first | stability_first | balanced",
        "gamma": "float in [0.7, 0.99]",
        "migration_weight": "float in [0.5, 1.8]",
        "transmission_weight": "float in [0.5, 1.8]",
        "solver_mode": "threshold | myopic | mdp",
        "reason": "short explanation",
    }
    return (
        "You are a controller for single-user service migration.\n"
        "Read the state JSON and return only a JSON object matching the required schema.\n"
        "Do not output actions, formulas, or any extra text.\n"
        f"Required schema:\n{json.dumps(required_schema, ensure_ascii=False)}\n"
        f"STATE JSON:\n{json.dumps(llm_state, ensure_ascii=False)}"
    )
