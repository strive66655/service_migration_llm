from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .core import CostParams, RealTraceConfig, matlab_round_or_ceil, nearest_cloud_index, nearest_state_index, reduced_chain_from_stay_probability
from .io import load_trace_data, scalar
from .policies import ModifiedPolicyIterationPolicy, MyopicPolicy, PolicyContext


def _build_cost_params(total_users: np.ndarray, timeslot: int, config: RealTraceConfig) -> CostParams:
    max_users = int(np.max(total_users))
    available_trans = config.avail_resource_trans_factor * max_users
    available_migrate = config.avail_resource_migration_factor * max_users
    proportional_migrate = -1 / (1 - total_users[timeslot] / available_trans)
    proportional_trans = -1 / (1 - total_users[timeslot] / available_trans)
    const_migrate = 1 / (1 - total_users[timeslot] / available_migrate) - proportional_migrate
    const_trans = -proportional_trans
    return CostParams(config.gamma, config.power_factor, float(const_migrate), float(proportional_migrate), float(const_trans), float(proportional_trans))


def _state_distance(coordinates: np.ndarray, a: int, b: int, cell_dist: float) -> int:
    distance = np.linalg.norm(coordinates[a - 1] - coordinates[b - 1]) / cell_dist
    return int(matlab_round_or_ceil(distance))


def _candidate_clouds(coordinates: np.ndarray, user_cell: int, cloud_indexes: np.ndarray, num_states_2d: int, cell_dist: float) -> np.ndarray:
    distances = np.array([_state_distance(coordinates, user_cell, idx, cell_dist) for idx in cloud_indexes], dtype=int)
    subset = cloud_indexes[distances < num_states_2d]
    return subset if len(subset) > 0 else cloud_indexes


def _instant_cost(cost_params: CostParams, prev_location: int, new_location: int, user_cell: int, coordinates: np.ndarray, cell_dist: float) -> float:
    migrate_distance = 0 if prev_location == 0 else _state_distance(coordinates, prev_location, new_location, cell_dist)
    trans_distance = _state_distance(coordinates, user_cell, new_location, cell_dist)
    migrate_cost = (cost_params.const_factor_migrate + cost_params.proportional_factor_migrate * cost_params.power_factor**migrate_distance) * int(new_location != prev_location)
    trans_cost = (cost_params.const_factor_trans + cost_params.proportional_factor_trans * cost_params.power_factor**trans_distance) * int(user_cell != new_location)
    return float(migrate_cost + trans_cost)


def _rank_reassignment_candidates(key: str, user_idx: int, timeslot: int, proposals: dict[str, np.ndarray], service_locations: dict[str, np.ndarray], origins: dict[str, np.ndarray], cell_of_users: np.ndarray, cloud_indexes: np.ndarray, coordinates: np.ndarray, cell_dist: float, config: RealTraceConfig, cost_params: CostParams, p_each_slot: np.ndarray, values_threshold_each_timeslot: np.ndarray) -> np.ndarray:
    user_cell = int(cell_of_users[timeslot, user_idx])
    if key in {"never", "always"}:
        target = origins[key][user_idx]
        return cloud_indexes[np.argsort(np.sum((coordinates[cloud_indexes - 1] - coordinates[target - 1]) ** 2, axis=1))]

    if key == "myopic":
        candidates = _candidate_clouds(coordinates, user_cell, cloud_indexes, config.num_states_2d, cell_dist)
        prev = int(service_locations[key][user_idx])
        scores = []
        for candidate in candidates:
            migrate_distance = 0 if prev == 0 else _state_distance(coordinates, prev, int(candidate), cell_dist)
            trans_distance = _state_distance(coordinates, user_cell, int(candidate), cell_dist)
            migrate_cost = (cost_params.const_factor_migrate + cost_params.proportional_factor_migrate * cost_params.power_factor**migrate_distance) * int(candidate != prev)
            trans_cost = (cost_params.const_factor_trans + cost_params.proportional_factor_trans * cost_params.power_factor**trans_distance) * int(user_cell != candidate)
            scores.append(float(migrate_cost + trans_cost))
        return candidates[np.argsort(np.asarray(scores))]

    candidates = _candidate_clouds(coordinates, user_cell, cloud_indexes, config.num_states_2d, cell_dist)
    prev = int(service_locations[key][user_idx])
    scores = []
    for candidate in candidates:
        migrate_distance = 0 if prev == 0 else _state_distance(coordinates, prev, int(candidate), cell_dist)
        trans_distance = min(_state_distance(coordinates, user_cell, int(candidate), cell_dist), config.num_states_2d)
        migrate_cost = (cost_params.const_factor_migrate + cost_params.proportional_factor_migrate * cost_params.power_factor**migrate_distance) * int(candidate != prev)
        trans_cost = (cost_params.const_factor_trans + cost_params.proportional_factor_trans * cost_params.power_factor**trans_distance) * int(user_cell != candidate)
        future = config.gamma * float(p_each_slot[trans_distance, :, timeslot] @ values_threshold_each_timeslot[:, timeslot])
        scores.append(float(migrate_cost + trans_cost + future))
    return candidates[np.argsort(np.asarray(scores))]


def run_real_trace(config: RealTraceConfig) -> dict:
    data = load_trace_data(config.data_path)
    coordinates = np.asarray(data["coordinatesCells2D"], dtype=float)
    total_users = np.asarray(data["totalUsers"], dtype=int).reshape(-1)
    cell_of_users = np.asarray(data["cellOfUsers"], dtype=int)
    cell_dist = float(scalar(data["cellDist"]))
    num_states_2d_total = int(scalar(data["numStates2DTotal"]))
    time_min = int(scalar(data["timeMin"])) if "timeMin" in data else None
    time_max = int(scalar(data["timeMax"])) if "timeMax" in data else None
    update_time_step = int(scalar(data["updateTimeStep"])) if "updateTimeStep" in data else None

    total_user_each_slot = np.count_nonzero(cell_of_users, axis=1)
    left_user_each_slot = np.zeros_like(total_user_each_slot)
    for user in range(cell_of_users.shape[1]):
        prev_cell = 0
        for slot in range(cell_of_users.shape[0] - 1, -1, -1):
            current = cell_of_users[slot, user]
            if prev_cell != 0 and slot < cell_of_users.shape[0] - 1 and current != prev_cell:
                left_user_each_slot[slot + 1] += 1
            prev_cell = current

    actions_myopic_each_timeslot = np.zeros((cell_of_users.shape[0], num_states_2d_total), dtype=int)
    actions_threshold_each_timeslot = np.zeros((cell_of_users.shape[0], config.num_states_2d + 1), dtype=int)
    values_threshold_each_timeslot = np.zeros((config.num_states_2d + 1, cell_of_users.shape[0]), dtype=float)
    p_each_slot = np.zeros((config.num_states_2d + 1, config.num_states_2d + 1, cell_of_users.shape[0]), dtype=float)
    cost_params_by_timeslot: dict[int, CostParams] = {}

    for timeslot in range(cell_of_users.shape[0] - 1, -1, -1):
        stay_probability = 1.0 if total_user_each_slot[timeslot] == 0 else 1 - left_user_each_slot[timeslot] / total_user_each_slot[timeslot]
        reduced_p, _ = reduced_chain_from_stay_probability(stay_probability, config.num_states_2d)
        cost_params = _build_cost_params(total_users, timeslot, config)
        cost_params_by_timeslot[timeslot] = cost_params
        myopic_result = MyopicPolicy().solve(PolicyContext(np.eye(num_states_2d_total), cost_params, 1, coordinates, cell_dist, action_mode="standard"))
        threshold_result = ModifiedPolicyIterationPolicy().solve(PolicyContext(reduced_p, cost_params, 1, action_mode="distance"))
        actions_myopic_each_timeslot[timeslot] = myopic_result.actions
        actions_threshold_each_timeslot[timeslot] = threshold_result.actions
        values_threshold_each_timeslot[:, timeslot] = threshold_result.state_values
        p_each_slot[:, :, timeslot] = reduced_p

    step = coordinates.shape[0] / config.num_cells_with_cloud
    cloud_mask = np.zeros(coordinates.shape[0], dtype=int)
    idx = step
    while round(idx) <= coordinates.shape[0]:
        cloud_mask[round(idx) - 1] = 1
        idx += step
    cloud_indexes = np.where(cloud_mask == 1)[0] + 1

    policy_names = ["never", "always", "myopic", "threshold"]
    service_locations = {key: np.zeros(cell_of_users.shape[1], dtype=int) for key in policy_names}
    trace_costs = {key: np.zeros((cell_of_users.shape[0], cell_of_users.shape[1]), dtype=float) for key in policy_names}

    for timeslot in range(cell_of_users.shape[0] - 1, -1, -1):
        cost_params = cost_params_by_timeslot[timeslot]
        proposals = {key: np.zeros(cell_of_users.shape[1], dtype=int) for key in policy_names}
        origins = {key: np.zeros(cell_of_users.shape[1], dtype=int) for key in policy_names}
        for user_idx in range(cell_of_users.shape[1]):
            user_cell = int(cell_of_users[timeslot, user_idx])
            if user_cell == 0:
                continue

            prev_never = int(service_locations["never"][user_idx])
            raw_never = user_cell if prev_never == 0 or _state_distance(coordinates, user_cell, prev_never, cell_dist) >= config.num_states_2d else prev_never
            origins["never"][user_idx] = raw_never
            proposals["never"][user_idx] = nearest_cloud_index(coordinates, cloud_indexes, raw_never)

            origins["always"][user_idx] = user_cell
            proposals["always"][user_idx] = nearest_cloud_index(coordinates, cloud_indexes, user_cell)

            prev_myopic = int(service_locations["myopic"][user_idx])
            if prev_myopic == 0:
                raw_myopic = user_cell
            else:
                query_point = np.asarray(config.center_coordinate) + coordinates[user_cell - 1] - coordinates[prev_myopic - 1]
                offset = nearest_state_index(coordinates, query_point)
                if offset == 1:
                    new_point = coordinates[user_cell - 1]
                else:
                    new_point = coordinates[user_cell - 1] - (coordinates[actions_myopic_each_timeslot[timeslot, offset - 1] - 1] - np.asarray(config.center_coordinate))
                raw_myopic = nearest_state_index(coordinates, new_point)
            origins["myopic"][user_idx] = raw_myopic
            candidates = _candidate_clouds(coordinates, user_cell, cloud_indexes, config.num_states_2d, cell_dist)
            myopic_costs = []
            for candidate in candidates:
                migrate_distance = 0 if prev_myopic == 0 else _state_distance(coordinates, prev_myopic, int(candidate), cell_dist)
                trans_distance = _state_distance(coordinates, user_cell, int(candidate), cell_dist)
                migrate_cost = (cost_params.const_factor_migrate + cost_params.proportional_factor_migrate * cost_params.power_factor**migrate_distance) * int(candidate != prev_myopic)
                trans_cost = (cost_params.const_factor_trans + cost_params.proportional_factor_trans * cost_params.power_factor**trans_distance) * int(user_cell != candidate)
                myopic_costs.append(float(migrate_cost + trans_cost))
            proposals["myopic"][user_idx] = int(candidates[int(np.argmin(myopic_costs))])

            prev_threshold = int(service_locations["threshold"][user_idx])
            if prev_threshold == 0:
                raw_threshold = user_cell
            else:
                distance = _state_distance(coordinates, user_cell, prev_threshold, cell_dist)
                if distance == 0:
                    new_point = coordinates[user_cell - 1]
                else:
                    target_distance = int(actions_threshold_each_timeslot[timeslot, min(distance, config.num_states_2d)] - 1)
                    new_point = (coordinates[user_cell - 1] - coordinates[prev_threshold - 1]) * (1 - target_distance / distance) + coordinates[prev_threshold - 1]
                raw_threshold = nearest_state_index(coordinates, new_point)
            origins["threshold"][user_idx] = raw_threshold
            threshold_costs = []
            for candidate in cloud_indexes:
                migrate_distance = 0 if prev_threshold == 0 else _state_distance(coordinates, prev_threshold, int(candidate), cell_dist)
                trans_distance = min(_state_distance(coordinates, user_cell, int(candidate), cell_dist), config.num_states_2d)
                migrate_cost = (cost_params.const_factor_migrate + cost_params.proportional_factor_migrate * cost_params.power_factor**migrate_distance) * int(candidate != prev_threshold)
                trans_cost = (cost_params.const_factor_trans + cost_params.proportional_factor_trans * cost_params.power_factor**trans_distance) * int(user_cell != candidate)
                future = config.gamma * float(p_each_slot[trans_distance, :, timeslot] @ values_threshold_each_timeslot[:, timeslot])
                threshold_costs.append(float(migrate_cost + trans_cost + future))
            proposals["threshold"][user_idx] = int(cloud_indexes[int(np.argmin(threshold_costs))])

        for key in policy_names:
            counts = np.bincount(proposals[key], minlength=coordinates.shape[0] + 1)
            overloaded = np.where(counts > config.max_user_each_cloud)[0]
            for cloud in overloaded:
                if cloud == 0:
                    continue
                users = np.where(proposals[key] == cloud)[0]
                costs = np.array([
                    _instant_cost(cost_params, int(service_locations[key][u]), int(proposals[key][u]), int(cell_of_users[timeslot, u]), coordinates, cell_dist)
                    for u in users
                ])
                ranked_users = users[np.argsort(costs)[::-1]]
                for user_idx in ranked_users[: counts[cloud] - config.max_user_each_cloud]:
                    ranked_clouds = _rank_reassignment_candidates(
                        key,
                        int(user_idx),
                        timeslot,
                        proposals,
                        service_locations,
                        origins,
                        cell_of_users,
                        cloud_indexes,
                        coordinates,
                        cell_dist,
                        config,
                        cost_params,
                        p_each_slot,
                        values_threshold_each_timeslot,
                    )
                    for candidate in ranked_clouds:
                        if counts[int(candidate)] < config.max_user_each_cloud:
                            counts[cloud] -= 1
                            counts[int(candidate)] += 1
                            proposals[key][user_idx] = int(candidate)
                            break

        for user_idx in range(cell_of_users.shape[1]):
            user_cell = int(cell_of_users[timeslot, user_idx])
            if user_cell == 0:
                for key in policy_names:
                    service_locations[key][user_idx] = 0
                    trace_costs[key][timeslot, user_idx] = 0.0
                continue
            for key in policy_names:
                prev = int(service_locations[key][user_idx])
                new = int(proposals[key][user_idx])
                trace_costs[key][timeslot, user_idx] = _instant_cost(cost_params, prev, new, user_cell, coordinates, cell_dist)
                service_locations[key][user_idx] = new

    total_users_safe = np.where(total_users == 0, 1, total_users)
    instantaneous = {key: np.sum(trace_costs[key], axis=1) / total_users_safe for key in policy_names}
    summary = {key: float(np.mean(instantaneous[key])) for key in policy_names}
    std_summary = {key: float(np.std(instantaneous[key])) for key in policy_names}

    gains = {
        "gain_over_never": np.divide(np.sum(trace_costs["threshold"], axis=1), np.sum(trace_costs["never"], axis=1), out=np.full(cell_of_users.shape[0], np.nan), where=np.sum(trace_costs["never"], axis=1) != 0),
        "gain_over_always": np.divide(np.sum(trace_costs["threshold"], axis=1), np.sum(trace_costs["always"], axis=1), out=np.full(cell_of_users.shape[0], np.nan), where=np.sum(trace_costs["always"], axis=1) != 0),
        "gain_over_myopic": np.divide(np.sum(trace_costs["threshold"], axis=1), np.sum(trace_costs["myopic"], axis=1), out=np.full(cell_of_users.shape[0], np.nan), where=np.sum(trace_costs["myopic"], axis=1) != 0),
    }
    gain_stats = {}
    for key, values in gains.items():
        valid = values[~np.isnan(values)]
        gain_stats[key] = {
            "mean": float(np.mean(valid)) if len(valid) else float("nan"),
            "std": float(np.std(valid)) if len(valid) else float("nan"),
            "min": float(np.min(valid)) if len(valid) else float("nan"),
            "max": float(np.max(valid)) if len(valid) else float("nan"),
        }

    first_migrate = []
    for timeslot in range(actions_threshold_each_timeslot.shape[0]):
        found = np.where(actions_threshold_each_timeslot[timeslot] < np.arange(1, config.num_states_2d + 2))[0]
        first_migrate.append(int(found[0]) if len(found) else config.num_states_2d)
    first_migrate = np.asarray(first_migrate)
    first_migrate_stats = {
        "mean": float(np.mean(first_migrate)),
        "std": float(np.std(first_migrate)),
        "min": int(np.min(first_migrate)),
        "max": int(np.max(first_migrate)),
    }

    time_axis = None
    if time_min is not None and time_max is not None and update_time_step is not None:
        time_axis = list(range(time_max, time_min - 1, -update_time_step))[: cell_of_users.shape[0]]

    return {
        "config": asdict(config),
        "summary": summary,
        "std_summary": std_summary,
        "avg_cost_series": {key: instantaneous[key].tolist() for key in policy_names},
        "gain_stats": gain_stats,
        "first_migrate_stats": first_migrate_stats,
        "time_axis": time_axis,
    }
