from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .core import CostParams, PolicyResult, build_1d_transition_matrix, build_random_walk_2d_transition_matrix, evaluate_policy, hex_grid_coordinates, hop_distance_matrix, map_threshold_actions_to_2d, reduced_chain_from_stay_probability
from .llm import DEFAULT_SAFE_CONTROL, apply_control_params, build_llm_state, build_prompt, build_shared_control_state, query_llm, query_multi_agent_control, validate_llm_output
from .policies import AlwaysMigratePolicy, ModifiedPolicyIterationPolicy, MyopicPolicy, NeverMigratePolicy, PolicyContext, PolicyIterationPolicy


@dataclass(slots=True)
class SingleUserLLMConfig:
    use_2d: bool = True
    sim_seed: int = 1
    gamma: float = 0.9
    migrate_proportional: float = 1.0
    power_factor: float = 0.8
    num_steps: int = 60
    llm_refresh_interval: int = 5
    failure_mode: str | None = None
    llm_backend: str = "mock"
    llm_model: str = "openai/gpt-5.3-chat"
    llm_api_base: str = "https://openrouter.ai/api/v1"
    llm_api_key_env: str = "OPENROUTER_API_KEY"
    llm_timeout_sec: float = 30.0
    controller_mode: str = "single_agent"
    business_profile: str = "balanced"
    operator_text: str = ""
    agent_models: dict[str, str] | None = None
    agent_backends: dict[str, str] | None = None
    num_states_left: int = 0
    num_states_right: int = 10
    num_states_2d: int = 6
    cell_dist: float = 0.005
    center_coordinate: tuple[float, float] = (37.762, -122.43)
    evaluation_weights: tuple[float, float, float, float] = (1.0, 0.75, 0.5, 1.25)
    distance_threshold: int = 3


@dataclass(slots=True)
class _EnvironmentSpec:
    transition_matrix: np.ndarray
    reduced_transition_matrix: np.ndarray | None
    zero_state_index: int
    coordinates: np.ndarray | None
    cell_dist: float | None
    hop_distances: np.ndarray | None
    allowed_actions: list[np.ndarray] | None
    threshold_action_mode: str
    num_states_2d: int | None = None
    ring_starts: np.ndarray | None = None


def _cost_params(gamma: float, migrate_proportional: float, power_factor: float) -> CostParams:
    proportional_factor_migrate = -migrate_proportional
    proportional_factor_trans = -1.0
    const_factor_migrate = 1.0 - proportional_factor_migrate
    const_factor_trans = -proportional_factor_trans
    return CostParams(gamma, power_factor, const_factor_migrate, proportional_factor_migrate, const_factor_trans, proportional_factor_trans)


def _standard_allowed_actions_2d(num_states_total: int, outer_ring_start: int) -> list[np.ndarray]:
    allowed = []
    for state in range(1, num_states_total + 1):
        if state >= outer_ring_start:
            allowed.append(np.arange(1, outer_ring_start, dtype=int))
        else:
            allowed.append(np.arange(1, num_states_total + 1, dtype=int))
    return allowed


def _build_environment(config: SingleUserLLMConfig) -> tuple[_EnvironmentSpec, CostParams]:
    base_cost_params = _cost_params(config.gamma, config.migrate_proportional, config.power_factor)
    rng = np.random.default_rng(config.sim_seed)
    if config.use_2d:
        coordinates, ring_starts = hex_grid_coordinates(config.num_states_2d, config.cell_dist, config.center_coordinate)
        p_2d = float(rng.random() / 6.0)
        transition_matrix = build_random_walk_2d_transition_matrix(config.num_states_2d, ring_starts, p_2d)
        reduced_p, _ = reduced_chain_from_stay_probability(float(transition_matrix[0, 0]), config.num_states_2d)
        hop_distances = hop_distance_matrix(coordinates, config.cell_dist)
        return (
            _EnvironmentSpec(
                transition_matrix=transition_matrix,
                reduced_transition_matrix=reduced_p,
                zero_state_index=1,
                coordinates=coordinates,
                cell_dist=config.cell_dist,
                hop_distances=hop_distances,
                allowed_actions=_standard_allowed_actions_2d(transition_matrix.shape[0], ring_starts[config.num_states_2d]),
                threshold_action_mode="distance",
                num_states_2d=config.num_states_2d,
                ring_starts=ring_starts,
            ),
            base_cost_params,
        )

    p_forward = float(rng.random())
    p_back = float(rng.random() * (1 - p_forward))
    p_same = 1 - p_forward - p_back
    p_out_state_first = float(rng.random()) if config.num_states_left == 0 else 0.0
    num_states = config.num_states_left + config.num_states_right + 1
    zero_state_index = config.num_states_left + 1
    transition_matrix = build_1d_transition_matrix(num_states, p_forward, p_back, p_same, p_out_state_first, 0.0)
    return (
        _EnvironmentSpec(
            transition_matrix=transition_matrix,
            reduced_transition_matrix=None,
            zero_state_index=zero_state_index,
            coordinates=None,
            cell_dist=None,
            hop_distances=None,
            allowed_actions=None,
            threshold_action_mode="standard",
        ),
        base_cost_params,
    )


def _distance(env: _EnvironmentSpec, src: int, dst: int) -> int:
    if env.hop_distances is not None:
        return int(env.hop_distances[src - 1, dst - 1])
    return abs(src - dst)


def _immediate_cost(env: _EnvironmentSpec, cost_params: CostParams, state_index: int, action: int) -> float:
    migrate_distance = _distance(env, state_index, action)
    trans_distance = _distance(env, env.zero_state_index, action)
    migrate_cost = (cost_params.const_factor_migrate + cost_params.proportional_factor_migrate * cost_params.power_factor**migrate_distance) * int(action != state_index)
    trans_cost = (cost_params.const_factor_trans + cost_params.proportional_factor_trans * cost_params.power_factor**trans_distance) * int(action != env.zero_state_index)
    return float(migrate_cost + trans_cost)


def _sample_next_state(row: np.ndarray, draw: float) -> int:
    cdf = np.cumsum(row)
    idx = int(np.searchsorted(cdf, draw, side="right"))
    return min(idx + 1, len(row))


def _standard_context(env: _EnvironmentSpec, cost_params: CostParams) -> PolicyContext:
    return PolicyContext(
        env.transition_matrix,
        cost_params,
        env.zero_state_index,
        env.coordinates,
        env.cell_dist,
        env.hop_distances,
        action_mode="standard",
        allowed_actions=env.allowed_actions,
    )


def _threshold_context(env: _EnvironmentSpec, cost_params: CostParams) -> PolicyContext:
    matrix = env.reduced_transition_matrix if env.reduced_transition_matrix is not None else env.transition_matrix
    return PolicyContext(matrix, cost_params, env.zero_state_index, action_mode=env.threshold_action_mode)


def _solve_policy_actions(env: _EnvironmentSpec, cost_params: CostParams, solver_mode: str) -> PolicyResult:
    context = _standard_context(env, cost_params)
    if solver_mode == "myopic":
        result = MyopicPolicy().solve(context)
        result.metadata["solver_mode"] = "myopic"
        return result
    if solver_mode == "threshold":
        threshold_result = ModifiedPolicyIterationPolicy().solve(_threshold_context(env, cost_params))
        if env.reduced_transition_matrix is not None:
            assert env.num_states_2d is not None and env.ring_starts is not None and env.coordinates is not None and env.cell_dist is not None
            actions = map_threshold_actions_to_2d(threshold_result.actions, env.num_states_2d, env.ring_starts, env.coordinates, env.cell_dist, env.hop_distances)
            values = evaluate_policy(actions, env.transition_matrix, cost_params, env.zero_state_index, env.coordinates, env.cell_dist, env.hop_distances)
            return PolicyResult(actions=actions, state_values=values, runtime_sec=threshold_result.runtime_sec, metadata={"solver_mode": "threshold"})
        threshold_result.metadata["solver_mode"] = "threshold"
        return threshold_result
    result = PolicyIterationPolicy().solve(context)
    result.metadata["solver_mode"] = "mdp"
    return result


def _collect_summary(trace: dict[str, list[float] | list[int]]) -> dict[str, Any]:
    return {
        "avg_service_distance": float(np.mean(np.asarray(trace["service_distance"], dtype=float))),
        "distance_violation_ratio": float(np.mean(np.asarray(trace["distance_violation"], dtype=float))),
        "avg_migration_distance": float(np.mean(np.asarray(trace["migration_distance"], dtype=float))),
        "avg_migration_count": float(np.mean(np.asarray(trace["migration_flag"], dtype=float))),
        "jitter_ratio": float(np.mean(np.asarray(trace["instability_flag"], dtype=float))),
        "evaluation_cost": float(np.mean(np.asarray(trace["evaluation_cost"], dtype=float))),
        "run_cost": float(np.mean(np.asarray(trace["run_cost"], dtype=float))),
    }


def _build_shared_state_for_step(env: _EnvironmentSpec, current_state: int, history: dict[str, Any], config: SingleUserLLMConfig) -> dict[str, Any]:
    return build_shared_control_state(
        {
            "state_index": current_state,
            "service_index": current_state,
            "distance_to_user": _distance(env, current_state, env.zero_state_index),
            "recent_direction": 0 if len(history["recent_service_distances"]) < 2 else int(history["recent_service_distances"][-1] - history["recent_service_distances"][-2]),
        },
        history,
        config.business_profile,
        config.operator_text,
    )


def _single_agent_diagnostics(
    shared_control_state: dict[str, Any],
    llm_raw: dict[str, Any] | None,
    llm_control: Any,
) -> dict[str, Any]:
    validation_notes = list(llm_control.validation_notes)
    if llm_raw is None:
        final_decision_source = "fallback_default"
    elif llm_control.used_fallback:
        final_decision_source = "fallback_partial"
    else:
        final_decision_source = "single_agent_direct"
    return {
        "shared_control_state": shared_control_state,
        "forecaster_raw_output": None,
        "forecaster_output": None,
        "policy_advisor_raw_output": None,
        "policy_advisor_output": None,
        "draft_control": llm_raw,
        "final_safe_control": asdict(llm_control),
        "validation_notes": validation_notes,
        "fallback_used": bool(validation_notes),
        "fallback_reason": ", ".join(validation_notes) if validation_notes else "",
        "final_decision_source": final_decision_source,
        "agent_agreement": "high",
        "agent_metrics": {
            "single_agent": {"latency_ms": 0.0, "call_count": 1},
        },
    }


def run_single_user_llm_loop(config: SingleUserLLMConfig) -> dict[str, Any]:
    env, base_cost_params = _build_environment(config)
    standard_context = _standard_context(env, base_cost_params)
    baseline_actions = {
        "never_migrate": NeverMigratePolicy().solve(standard_context).actions,
        "always_migrate": AlwaysMigratePolicy().solve(standard_context).actions,
        "myopic": MyopicPolicy().solve(standard_context).actions,
        "mdp_baseline": _solve_policy_actions(env, base_cost_params, "mdp").actions,
    }

    methods = {
        name: {
            "state": env.zero_state_index,
            "trace": {
                "service_distance": [],
                "migration_distance": [],
                "migration_flag": [],
                "instability_flag": [],
                "distance_violation": [],
                "evaluation_cost": [],
                "run_cost": [],
                "actions": [],
                "states": [],
            },
            "prev_migration_flag": 0,
        }
        for name in [*baseline_actions.keys(), "llm_meta_mdp"]
    }

    llm_control = DEFAULT_SAFE_CONTROL
    llm_cost_params = apply_control_params(base_cost_params, llm_control)
    llm_policy_actions = _solve_policy_actions(env, llm_cost_params, llm_control.solver_mode).actions
    llm_history: dict[str, Any] = {
        "recent_service_distances": [],
        "recent_migrations": [],
        "previous_solver_mode": llm_control.solver_mode,
    }
    llm_decisions: list[dict[str, Any]] = []
    draws = np.random.default_rng(config.sim_seed + 97).random(config.num_steps)
    weights = config.evaluation_weights

    for step in range(config.num_steps):
        if step % max(config.llm_refresh_interval, 1) == 0:
            current_state = int(methods["llm_meta_mdp"]["state"])
            shared_control_state = _build_shared_state_for_step(env, current_state, llm_history, config)
            if config.controller_mode == "multi_agent":
                diagnostics = query_multi_agent_control(
                    shared_control_state,
                    failure_mode=config.failure_mode if step == 0 else None,
                    backend=config.llm_backend,
                    model=config.llm_model,
                    api_base=config.llm_api_base,
                    api_key_env=config.llm_api_key_env,
                    timeout_sec=config.llm_timeout_sec,
                    agent_models=config.agent_models,
                    agent_backends=config.agent_backends,
                )
                llm_control = validate_llm_output(diagnostics["final_safe_control"], DEFAULT_SAFE_CONTROL)
            else:
                llm_state = build_llm_state(
                    {
                        "state_index": current_state,
                        "service_index": current_state,
                        "distance_to_user": shared_control_state["current_state"]["distance_to_user"],
                        "recent_direction": shared_control_state["current_state"]["recent_direction"],
                    },
                    {
                        "recent_service_distances": shared_control_state["recent_service_distances"],
                        "recent_migrations": shared_control_state["recent_migrations"],
                        "previous_solver_mode": shared_control_state["previous_solver_mode"],
                    },
                    config.business_profile,
                    config.operator_text,
                )
                prompt = build_prompt(llm_state)
                try:
                    llm_raw = query_llm(
                        prompt,
                        state=llm_state,
                        failure_mode=config.failure_mode if step == 0 else None,
                        backend=config.llm_backend,
                        model=config.llm_model,
                        api_base=config.llm_api_base,
                        api_key_env=config.llm_api_key_env,
                        timeout_sec=config.llm_timeout_sec,
                    )
                except TimeoutError:
                    llm_raw = None
                llm_control = validate_llm_output(llm_raw, DEFAULT_SAFE_CONTROL)
                diagnostics = _single_agent_diagnostics(shared_control_state, llm_raw, llm_control)
            llm_cost_params = apply_control_params(base_cost_params, llm_control)
            llm_policy_actions = _solve_policy_actions(env, llm_cost_params, llm_control.solver_mode).actions
            llm_history["previous_solver_mode"] = llm_control.solver_mode
            llm_decisions.append(
                {
                    "step": step,
                    "controller_mode": config.controller_mode,
                    **diagnostics,
                    "validated_control": asdict(llm_control),
                }
            )

        for name, method in methods.items():
            current_state = int(method["state"])
            action = int(llm_policy_actions[current_state - 1]) if name == "llm_meta_mdp" else int(baseline_actions[name][current_state - 1])
            service_distance = _distance(env, action, env.zero_state_index)
            migration_distance = _distance(env, current_state, action)
            migration_flag = int(action != current_state)
            instability_flag = int(migration_flag and method["prev_migration_flag"])
            evaluation_cost = (
                weights[0] * service_distance
                + weights[1] * migration_distance
                + weights[2] * migration_flag
                + weights[3] * instability_flag
            )
            run_cost_params = llm_cost_params if name == "llm_meta_mdp" else base_cost_params
            run_cost = _immediate_cost(env, run_cost_params, current_state, action)

            trace = method["trace"]
            trace["service_distance"].append(service_distance)
            trace["migration_distance"].append(migration_distance)
            trace["migration_flag"].append(migration_flag)
            trace["instability_flag"].append(instability_flag)
            trace["distance_violation"].append(int(service_distance > config.distance_threshold))
            trace["evaluation_cost"].append(float(evaluation_cost))
            trace["run_cost"].append(run_cost)
            trace["actions"].append(action)
            trace["states"].append(current_state)

            method["prev_migration_flag"] = migration_flag
            method["state"] = _sample_next_state(env.transition_matrix[action - 1], float(draws[step]))

            if name == "llm_meta_mdp":
                llm_history["recent_service_distances"] = (llm_history["recent_service_distances"] + [service_distance])[-5:]
                llm_history["recent_migrations"] = (llm_history["recent_migrations"] + [migration_flag])[-5:]

    return {
        "config": asdict(config),
        "llm_decisions": llm_decisions,
        "method_summaries": {name: _collect_summary(methods[name]["trace"]) for name in methods},
        "method_traces": {name: methods[name]["trace"] for name in methods},
        "evaluation_metric_definition": {
            "weights": {
                "service_distance": weights[0],
                "migration_distance": weights[1],
                "migration_count": weights[2],
                "instability": weights[3],
            },
            "distance_threshold": config.distance_threshold,
            "notes": "evaluation_cost is fixed across all methods and is separate from run_cost.",
        },
    }
