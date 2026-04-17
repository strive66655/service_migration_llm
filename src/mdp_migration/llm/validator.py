from __future__ import annotations

from dataclasses import replace
from typing import Any

from .schema import DEFAULT_SAFE_CONTROL, FIXED_SOLVER_MODE, OBJECTIVE_MODES, SafeControlParams


def _clip(value: Any, lower: float, upper: float, fallback: float) -> tuple[float, str | None]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback, "non_numeric"
    clipped = min(max(parsed, lower), upper)
    if clipped != parsed:
        return clipped, "clipped"
    return clipped, None


def validate_llm_output(output: dict[str, Any] | None, fallback: SafeControlParams | None = None) -> SafeControlParams:
    safe_fallback = fallback if fallback is not None else DEFAULT_SAFE_CONTROL
    if output is None:
        return safe_fallback
    if "payload" in output and len(output) == 1:
        return replace(safe_fallback, validation_notes=safe_fallback.validation_notes + ("invalid_payload",))

    notes: list[str] = []
    objective_mode = str(output.get("objective_mode", safe_fallback.objective_mode))
    if objective_mode not in OBJECTIVE_MODES:
        notes.append("objective_mode_fallback")
        objective_mode = safe_fallback.objective_mode

    if "solver_mode" in output and str(output.get("solver_mode")) != FIXED_SOLVER_MODE:
        notes.append("solver_mode_locked")

    gamma, gamma_note = _clip(output.get("gamma", safe_fallback.gamma), 0.7, 0.99, safe_fallback.gamma)
    if gamma_note is not None:
        notes.append(f"gamma_{gamma_note}")

    migration_weight, migration_note = _clip(output.get("migration_weight", safe_fallback.migration_weight), 0.5, 1.8, safe_fallback.migration_weight)
    if migration_note is not None:
        notes.append(f"migration_weight_{migration_note}")

    transmission_weight, transmission_note = _clip(output.get("transmission_weight", safe_fallback.transmission_weight), 0.5, 1.8, safe_fallback.transmission_weight)
    if transmission_note is not None:
        notes.append(f"transmission_weight_{transmission_note}")

    reason = str(output.get("reason", safe_fallback.reason)).strip() or safe_fallback.reason
    used_fallback = bool(notes)
    return SafeControlParams(
        objective_mode=objective_mode,
        gamma=gamma,
        migration_weight=migration_weight,
        transmission_weight=transmission_weight,
        reason=reason,
        used_fallback=used_fallback,
        validation_notes=tuple(notes),
    )
