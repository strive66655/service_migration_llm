from __future__ import annotations

from dataclasses import dataclass


OBJECTIVE_MODES = ("latency_first", "stability_first", "balanced")
FIXED_SOLVER_MODE = "mdp"
SOLVER_MODES = (FIXED_SOLVER_MODE,)
BUSINESS_PROFILES = ("latency_sensitive", "delay_tolerant", "migration_sensitive", "high_stability_required", "balanced")


@dataclass(slots=True)
class LLMControlOutput:
    objective_mode: str
    gamma: float
    migration_weight: float
    transmission_weight: float
    reason: str


@dataclass(slots=True)
class SafeControlParams:
    objective_mode: str
    gamma: float
    migration_weight: float
    transmission_weight: float
    reason: str
    used_fallback: bool = False
    validation_notes: tuple[str, ...] = ()


DEFAULT_SAFE_CONTROL = SafeControlParams(
    objective_mode="balanced",
    gamma=0.9,
    migration_weight=1.0,
    transmission_weight=1.0,
    reason="Fallback to baseline MDP configuration.",
    used_fallback=True,
    validation_notes=("default_fallback",),
)
