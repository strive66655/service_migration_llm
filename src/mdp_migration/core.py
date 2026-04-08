from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

EPS = 1e-8


@dataclass(slots=True)
class CostParams:
    gamma: float
    power_factor: float
    const_factor_migrate: float
    proportional_factor_migrate: float
    const_factor_trans: float
    proportional_factor_trans: float


@dataclass(slots=True)
class RandomWalkConfig:
    use_2d: bool = True
    gamma_vector: tuple[float, ...] = (0.5, 0.9, 0.99)
    migrate_proportional_vector: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 4.0, 6.0, 10.0, 15.0, 20.0)
    sim_seed_vector: tuple[int, ...] = tuple(range(1, 51))
    num_workers: int = 1
    num_states_left: int = 0
    num_states_right: int = 10
    num_states_2d: int = 10
    cell_dist: float = 0.005
    center_coordinate: tuple[float, float] = (37.762, -122.43)
    power_factor: float = 0.8


@dataclass(slots=True)
class RealTraceConfig:
    data_path: str = "traceRealCellLocations.mat"
    gamma: float = 0.9
    power_factor: float = 0.8
    avail_resource_trans_factor: float = 1.5
    avail_resource_migration_factor: float = 1.5
    max_user_each_cloud: int = 50
    num_cells_with_cloud: int = 100
    num_states_2d: int = 20
    center_coordinate: tuple[float, float] = (37.762, -122.43)


@dataclass(slots=True)
class PolicyResult:
    actions: np.ndarray
    state_values: np.ndarray | None
    runtime_sec: float
    metadata: dict[str, Any] = field(default_factory=dict)


def matlab_round_or_ceil(values: np.ndarray | float) -> np.ndarray | int:
    array = np.asarray(values, dtype=float)
    rounded = np.round(array)
    result = np.where(np.abs(array - rounded) < EPS, rounded, np.ceil(array))
    if np.isscalar(values):
        return int(result.item())
    return result.astype(int)


def hex_grid_coordinates(num_states_2d: int, cell_dist: float, center_coordinate: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    num_states_total = 1
    ring_starts = np.zeros(num_states_2d + 2, dtype=int)
    for ring in range(1, num_states_2d + 1):
        ring_starts[ring] = num_states_total + 1
        num_states_total += ring * 6
    ring_starts[num_states_2d + 1] = num_states_total + 1

    coords = np.zeros((num_states_total, 2), dtype=float)
    for ring in range(1, num_states_2d + 1):
        ring_start = ring_starts[ring] - 1
        for side in range(6):
            idx = ring_start + side * ring
            coords[idx, 0] = cell_dist * ring * np.cos(2 * np.pi / 6 * side)
            coords[idx, 1] = cell_dist * ring * np.sin(2 * np.pi / 6 * side)
        for side in range(6):
            prev_corner = ring_start + (side % 6) * ring
            next_corner = ring_start + ((side + 1) % 6) * ring
            for point in range(prev_corner + 1, prev_corner + ring):
                coords[point] = coords[prev_corner] + ((coords[next_corner] - coords[prev_corner]) / ring * (point - prev_corner))

    coords[:, 0] += center_coordinate[0]
    coords[:, 1] += center_coordinate[1]
    return coords, ring_starts


def hex_neighbor_matrix(coordinates: np.ndarray, cell_dist: float) -> np.ndarray:
    num_states_total = coordinates.shape[0]
    neighbors = np.zeros((num_states_total, 7), dtype=int)
    neighbors[:, 0] = np.arange(1, num_states_total + 1)
    diffs = coordinates[:, None, :] - coordinates[None, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=2))
    for i in range(num_states_total):
        adjacent = np.where(np.abs(dists[i] - cell_dist) < 1e-12)[0]
        for j in adjacent:
            delta = np.round((coordinates[j] - coordinates[i]) * 1e8) / 1e8
            if delta[0] > 0 and delta[1] == 0:
                neighbors[i, 1] = j + 1
            elif delta[0] > 0 and delta[1] > 0:
                neighbors[i, 2] = j + 1
            elif delta[0] < 0 and delta[1] > 0:
                neighbors[i, 3] = j + 1
            elif delta[0] < 0 and delta[1] == 0:
                neighbors[i, 4] = j + 1
            elif delta[0] < 0 and delta[1] < 0:
                neighbors[i, 5] = j + 1
            elif delta[0] > 0 and delta[1] < 0:
                neighbors[i, 6] = j + 1
    return neighbors


def build_random_walk_2d_transition_matrix(num_states_2d: int, ring_starts: np.ndarray, p_2d: float) -> np.ndarray:
    num_states_total = ring_starts[num_states_2d + 1] - 1
    p = np.zeros((num_states_total, num_states_total), dtype=float)
    for ring in range(1, num_states_2d + 1):
        ring_start = ring_starts[ring] - 1
        ring_end = ring_starts[ring + 1] - 1
        for j in range(ring_start, ring_end):
            nxt = ring_start if j + 1 >= ring_end else j + 1
            p[j, nxt] = p_2d
            p[nxt, j] = p_2d
    for i in range(6):
        p[0, i + 1] = p_2d
        p[i + 1, 0] = p_2d
    for ring in range(1, num_states_2d):
        for j in range(1, ring * 6 + 1):
            lower = int(np.ceil((ring + 1) / ring * (j - 1)))
            upper = int(np.ceil((ring + 1) / ring * j))
            for k in range(lower, upper + 1):
                tmp = (ring + 1) * 6 if k == 0 else k
                src = ring_starts[ring] - 1 + j - 1
                dst = ring_starts[ring + 1] - 1 + tmp - 1
                p[src, dst] = p_2d
                p[dst, src] = p_2d
    np.fill_diagonal(p, 1.0 - p.sum(axis=1))
    return p


def build_1d_transition_matrix(num_states: int, p_forward: float, p_back: float, p_same: float, p_out_state_first: float, p_out_state_last: float) -> np.ndarray:
    p = np.zeros((num_states, num_states), dtype=float)
    for i in range(num_states):
        for j in range(num_states):
            if i == j:
                if i == 0:
                    p[i, j] = 1 - p_out_state_first
                elif i == num_states - 1:
                    p[i, j] = 1 - p_out_state_last
                else:
                    p[i, j] = p_same
            elif j == i + 1:
                p[i, j] = p_out_state_first if i == 0 else p_forward
            elif j == i - 1:
                p[i, j] = p_out_state_last if i == num_states - 1 else p_back
    return p


def reduced_chain_from_stay_probability(stay_probability: float, num_states_2d: int) -> tuple[np.ndarray, dict[str, float]]:
    p_forward = (1 - stay_probability) / 6 * 2.5
    p_back = (1 - stay_probability) / 6 * 1.5
    p_same = 1 - p_forward - p_back
    p_out_state_first = 1 - stay_probability
    p_out_state_last = (1 - stay_probability) / 6
    p = build_1d_transition_matrix(num_states_2d + 1, p_forward, p_back, p_same, p_out_state_first, p_out_state_last)
    return p, {
        "p_forward": p_forward,
        "p_back": p_back,
        "p_same": p_same,
        "p_out_state_first": p_out_state_first,
        "p_out_state_last": p_out_state_last,
    }


def hop_distance_2d(coordinates: np.ndarray, src: int, dst: int, cell_dist: float) -> int:
    distance = np.linalg.norm(coordinates[src - 1] - coordinates[dst - 1]) / cell_dist
    return int(matlab_round_or_ceil(distance))


def hop_distance_matrix(coordinates: np.ndarray, cell_dist: float) -> np.ndarray:
    diffs = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.sqrt(np.sum(diffs * diffs, axis=2)) / cell_dist
    return matlab_round_or_ceil(distances).astype(int)


def policy_cost_vector(actions: np.ndarray, transition_matrix: np.ndarray, cost_params: CostParams, zero_state_index: int, coordinates: np.ndarray | None = None, cell_dist: float | None = None, hop_distances: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    num_states = len(actions)
    modified = np.zeros_like(transition_matrix, dtype=float)
    ck = np.zeros(num_states, dtype=float)
    for idx in range(num_states):
        action = int(actions[idx])
        modified[idx] = transition_matrix[action - 1]
        if coordinates is None:
            migrate_distance = abs(action - (idx + 1))
            trans_distance = abs(action - zero_state_index)
        else:
            assert cell_dist is not None
            if hop_distances is None:
                migrate_distance = hop_distance_2d(coordinates, idx + 1, action, cell_dist)
                trans_distance = hop_distance_2d(coordinates, zero_state_index, action, cell_dist)
            else:
                migrate_distance = int(hop_distances[idx, action - 1])
                trans_distance = int(hop_distances[zero_state_index - 1, action - 1])
        migrate_cost = (cost_params.const_factor_migrate + cost_params.proportional_factor_migrate * cost_params.power_factor**migrate_distance) * int(action != idx + 1)
        trans_cost = (cost_params.const_factor_trans + cost_params.proportional_factor_trans * cost_params.power_factor**trans_distance) * int(action != zero_state_index)
        ck[idx] = migrate_cost + trans_cost
    return modified, ck


def evaluate_policy(actions: np.ndarray, transition_matrix: np.ndarray, cost_params: CostParams, zero_state_index: int, coordinates: np.ndarray | None = None, cell_dist: float | None = None, hop_distances: np.ndarray | None = None) -> np.ndarray:
    modified, ck = policy_cost_vector(actions, transition_matrix, cost_params, zero_state_index, coordinates, cell_dist, hop_distances)
    return np.linalg.solve(np.eye(modified.shape[0]) - cost_params.gamma * modified, ck)


def nearest_state_index(coordinates: np.ndarray, point: np.ndarray) -> int:
    diffs = coordinates - point
    dists = np.sum(diffs * diffs, axis=1)
    return int(np.argmin(dists) + 1)


def nearest_cloud_index(coordinates: np.ndarray, cloud_indexes: np.ndarray, point_index: int) -> int:
    target = coordinates[point_index - 1]
    cloud_coords = coordinates[cloud_indexes - 1]
    dists = np.sum((cloud_coords - target) ** 2, axis=1)
    return int(cloud_indexes[np.argmin(dists)])


def map_threshold_actions_to_2d(actions_by_distance: np.ndarray, num_states_2d: int, ring_starts: np.ndarray, coordinates: np.ndarray, cell_dist: float, hop_distances: np.ndarray | None = None) -> np.ndarray:
    num_states_total = coordinates.shape[0]
    mapped = np.ones(num_states_total, dtype=int)
    for state in range(2, num_states_2d + 1):
        target_state = int(actions_by_distance[state - 1])
        ring_start = ring_starts[state - 1]
        ring_end = ring_starts[state] - 1
        for idx in range(ring_start, ring_end + 1):
            if target_state == state:
                mapped[idx - 1] = idx
            elif target_state == 1:
                mapped[idx - 1] = 1
            else:
                target_ring_start = ring_starts[target_state - 1]
                target_ring_end = ring_starts[target_state] - 1
                best_index = target_ring_start
                best_distance = float("inf")
                for candidate in range(target_ring_start, target_ring_end + 1):
                    if hop_distances is None:
                        distance = hop_distance_2d(coordinates, idx, candidate, cell_dist)
                    else:
                        distance = int(hop_distances[idx - 1, candidate - 1])
                    if distance < best_distance:
                        best_distance = distance
                        best_index = candidate
                mapped[idx - 1] = best_index
    if ring_starts[num_states_2d] <= num_states_total:
        mapped[ring_starts[num_states_2d] - 1 :] = 1
    return mapped
