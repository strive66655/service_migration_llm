from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict

import numpy as np

from .core import CostParams, RandomWalkConfig, build_1d_transition_matrix, build_random_walk_2d_transition_matrix, evaluate_policy, hex_grid_coordinates, hop_distance_matrix, map_threshold_actions_to_2d, reduced_chain_from_stay_probability
from .policies import AlwaysMigratePolicy, ModifiedPolicyIterationPolicy, MyopicPolicy, NeverMigratePolicy, PolicyContext, PolicyIterationPolicy, ValueIterationPolicy


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


def _run_random_walk_case(args: tuple[int, int, float, float, RandomWalkConfig, np.ndarray | None, np.ndarray | None, np.ndarray | None]) -> tuple[int, int, dict[str, float]]:
    g_idx, m_idx, gamma, migrate_proportional, config, coords, ring_starts, hop_distances = args
    totals = {
        key: 0.0
        for key in [
            "time_value",
            "time_policy",
            "time_th_policy",
            "value_error",
            "value_policy",
            "value_value_result",
            "value_value_actual",
            "value_never",
            "value_always",
            "value_myopic",
            "different_action_pct",
            "value_th_policy_actual",
        ]
    }

    for sim_seed in config.sim_seed_vector:
        rng = np.random.default_rng(sim_seed)
        cost_params = _cost_params(gamma, migrate_proportional, config.power_factor)
        skip_standard = (gamma == 0.5 and migrate_proportional > 2) or (gamma == 0.9 and migrate_proportional > 10)

        if config.use_2d:
            assert coords is not None and ring_starts is not None
            p_2d = rng.random() / 6.0
            p_standard = build_random_walk_2d_transition_matrix(config.num_states_2d, ring_starts, p_2d)
            reduced_p, _ = reduced_chain_from_stay_probability(float(p_standard[0, 0]), config.num_states_2d)
            allowed_actions = _standard_allowed_actions_2d(p_standard.shape[0], ring_starts[config.num_states_2d])
            standard_context = PolicyContext(
                p_standard,
                cost_params,
                1,
                coords,
                config.cell_dist,
                hop_distances,
                action_mode="standard",
                allowed_actions=allowed_actions,
            )
            threshold_context = PolicyContext(reduced_p, cost_params, 1, action_mode="distance")
        else:
            p_forward = float(rng.random())
            p_back = float(rng.random() * (1 - p_forward))
            p_same = 1 - p_forward - p_back
            p_out_state_first = float(rng.random()) if config.num_states_left == 0 else 0.0
            p_out_state_last = 0.0
            num_states = config.num_states_left + config.num_states_right + 1
            zero_state_index = config.num_states_left + 1
            p_standard = build_1d_transition_matrix(num_states, p_forward, p_back, p_same, p_out_state_first, p_out_state_last)
            standard_context = PolicyContext(p_standard, cost_params, zero_state_index, action_mode="standard")
            threshold_context = PolicyContext(p_standard, cost_params, zero_state_index, action_mode="standard")

        value_result = None
        policy_result = None
        if not skip_standard:
            value_result = ValueIterationPolicy().solve(standard_context)
            policy_result = PolicyIterationPolicy().solve(standard_context)
            value_actual = evaluate_policy(
                value_result.actions,
                p_standard,
                cost_params,
                standard_context.zero_state_index,
                coords if config.use_2d else None,
                config.cell_dist if config.use_2d else None,
                hop_distances if config.use_2d else None,
            )
            totals["time_value"] += value_result.runtime_sec
            totals["time_policy"] += policy_result.runtime_sec
            totals["value_policy"] += float(np.mean(policy_result.state_values))
            totals["value_value_result"] += float(np.mean(value_result.state_values))
            totals["value_value_actual"] += float(np.mean(value_actual))
            totals["value_error"] += float(np.sum(np.abs(policy_result.state_values - value_result.state_values)) / len(policy_result.state_values))

        threshold_result = ModifiedPolicyIterationPolicy().solve(threshold_context)
        if config.use_2d:
            threshold_actions = map_threshold_actions_to_2d(threshold_result.actions, config.num_states_2d, ring_starts, coords, config.cell_dist, hop_distances)
            threshold_values = evaluate_policy(threshold_actions, p_standard, cost_params, 1, coords, config.cell_dist, hop_distances)
        else:
            threshold_actions = threshold_result.actions
            threshold_values = evaluate_policy(threshold_actions, p_standard, cost_params, standard_context.zero_state_index)
            if policy_result is not None:
                different_action_pct = float(np.mean(threshold_actions != policy_result.actions))
                totals["different_action_pct"] += different_action_pct
                if different_action_pct > 0.0:
                    raise ValueError("Modified policy iteration actions differ from policy iteration actions in 1D.")

        never_result = NeverMigratePolicy().solve(standard_context)
        always_result = AlwaysMigratePolicy().solve(standard_context)
        myopic_result = MyopicPolicy().solve(standard_context)
        totals["time_th_policy"] += threshold_result.runtime_sec
        totals["value_never"] += float(np.mean(never_result.state_values))
        totals["value_always"] += float(np.mean(always_result.state_values))
        totals["value_myopic"] += float(np.mean(myopic_result.state_values))
        totals["value_th_policy_actual"] += float(np.mean(threshold_values))

    denom = len(config.sim_seed_vector)
    averages = {key: value / denom for key, value in totals.items()}
    return g_idx, m_idx, averages


def run_random_walk(config: RandomWalkConfig) -> dict:
    coords = None
    ring_starts = None
    distances = None
    if config.use_2d:
        coords, ring_starts = hex_grid_coordinates(config.num_states_2d, config.cell_dist, config.center_coordinate)
        distances = hop_distance_matrix(coords, config.cell_dist)

    shape = (len(config.gamma_vector), len(config.migrate_proportional_vector))
    stores = {
        key: np.zeros(shape)
        for key in [
            "time_value",
            "time_policy",
            "time_th_policy",
            "value_error",
            "value_policy",
            "value_value_result",
            "value_value_actual",
            "value_never",
            "value_always",
            "value_myopic",
            "different_action_pct",
            "value_th_policy_actual",
        ]
    }

    case_args = [
        (g_idx, m_idx, gamma, migrate_proportional, config, coords, ring_starts, distances)
        for g_idx, gamma in enumerate(config.gamma_vector)
        for m_idx, migrate_proportional in enumerate(config.migrate_proportional_vector)
    ]

    if config.num_workers > 1 and len(case_args) > 1:
        try:
            with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
                results_iter = executor.map(_run_random_walk_case, case_args)
                for g_idx, m_idx, averages in results_iter:
                    for key, value in averages.items():
                        stores[key][g_idx, m_idx] = value
        except (OSError, PermissionError):
            with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
                results_iter = executor.map(_run_random_walk_case, case_args)
                for g_idx, m_idx, averages in results_iter:
                    for key, value in averages.items():
                        stores[key][g_idx, m_idx] = value
    else:
        for case in case_args:
            g_idx, m_idx, averages = _run_random_walk_case(case)
            for key, value in averages.items():
                stores[key][g_idx, m_idx] = value

    return {
        "config": asdict(config),
        "sim_param_vector": list(config.migrate_proportional_vector),
        "time_value": stores["time_value"].tolist(),
        "time_policy": stores["time_policy"].tolist(),
        "time_th_policy": stores["time_th_policy"].tolist(),
        "value_error": stores["value_error"].reshape(-1).tolist(),
        "value_policy": stores["value_policy"].tolist(),
        "value_value_result": stores["value_value_result"].tolist(),
        "value_value_actual": stores["value_value_actual"].tolist(),
        "value_never": stores["value_never"].tolist(),
        "value_always": stores["value_always"].tolist(),
        "value_myopic": stores["value_myopic"].tolist(),
        "different_action_pct": stores["different_action_pct"].reshape(-1).tolist(),
        "value_th_policy": stores["value_th_policy_actual"].tolist(),
    }
