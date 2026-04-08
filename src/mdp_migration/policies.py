from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from time import perf_counter

import numpy as np

from .core import CostParams, PolicyResult, evaluate_policy, hop_distance_2d


@dataclass(slots=True)
class PolicyContext:
    transition_matrix: np.ndarray
    cost_params: CostParams
    zero_state_index: int
    coordinates: np.ndarray | None = None
    cell_dist: float | None = None
    hop_distances: np.ndarray | None = None
    num_value_iteration: int = 5
    epsilon_value_iteration: float = 0.1
    action_mode: str = "standard"
    allowed_actions: list[np.ndarray] | None = None


class Policy:
    name = "policy"

    def solve(self, context: PolicyContext) -> PolicyResult:
        raise NotImplementedError


def _default_action_candidates(state_index: int, num_states: int, zero_state_index: int, action_mode: str) -> np.ndarray:
    one_based = state_index + 1
    if action_mode == "distance":
        if one_based == 1:
            start = 2 if zero_state_index > 1 else 1
            return np.arange(start, one_based + 1, dtype=int)
        if one_based == num_states:
            return np.arange(1, one_based, dtype=int)
        return np.arange(1, one_based + 1, dtype=int)
    if one_based == 1:
        if zero_state_index > 1:
            return np.arange(2, num_states + 1, dtype=int)
        return np.arange(1, num_states + 1, dtype=int)
    if one_based == num_states:
        return np.arange(1, num_states, dtype=int)
    return np.arange(1, num_states + 1, dtype=int)


def _context_action_candidates(state_index: int, context: PolicyContext) -> np.ndarray:
    if context.allowed_actions is not None:
        return context.allowed_actions[state_index]
    return _default_action_candidates(
        state_index,
        context.transition_matrix.shape[0],
        context.zero_state_index,
        context.action_mode,
    )


def _immediate_cost(state_index: int, action: int, context: PolicyContext) -> float:
    if context.coordinates is None:
        migrate_distance = abs(action - (state_index + 1))
        trans_distance = abs(context.zero_state_index - action)
    else:
        assert context.cell_dist is not None
        if context.hop_distances is None:
            migrate_distance = hop_distance_2d(context.coordinates, state_index + 1, action, context.cell_dist)
            trans_distance = hop_distance_2d(context.coordinates, context.zero_state_index, action, context.cell_dist)
        else:
            migrate_distance = int(context.hop_distances[state_index, action - 1])
            trans_distance = int(context.hop_distances[context.zero_state_index - 1, action - 1])
    migrate_cost = (context.cost_params.const_factor_migrate + context.cost_params.proportional_factor_migrate * context.cost_params.power_factor**migrate_distance) * int(action != state_index + 1)
    trans_cost = (context.cost_params.const_factor_trans + context.cost_params.proportional_factor_trans * context.cost_params.power_factor**trans_distance) * int(action != context.zero_state_index)
    return float(migrate_cost + trans_cost)


def _first_less(actions: np.ndarray, start_one_based: int = 1) -> int | None:
    for i in range(start_one_based - 1, len(actions)):
        if actions[i] < i + 1:
            return i + 1
    return None


def _last_greater(actions: np.ndarray, end_one_based: int | None = None) -> int | None:
    end = len(actions) if end_one_based is None else end_one_based
    for i in range(end - 1, -1, -1):
        if actions[i] > i + 1:
            return i + 1
    return None


class ValueIterationPolicy(Policy):
    name = "value_iteration"

    def solve(self, context: PolicyContext) -> PolicyResult:
        start = perf_counter()
        num_states = context.transition_matrix.shape[0]
        values = np.zeros(num_states, dtype=float)
        actions = np.full(num_states, context.zero_state_index, dtype=int)
        threshold = context.epsilon_value_iteration * (1 - context.cost_params.gamma) / (2 * context.cost_params.gamma)
        for _ in range(context.num_value_iteration):
            prev = values.copy()
            for state_index in range(num_states):
                best_value = float("inf")
                for action in _context_action_candidates(state_index, context):
                    q = _immediate_cost(state_index, int(action), context) + context.cost_params.gamma * float(context.transition_matrix[int(action) - 1] @ values)
                    if q < best_value:
                        best_value = q
                        actions[state_index] = int(action)
                values[state_index] = best_value
            if np.all(np.abs(prev - values) < threshold):
                break
        return PolicyResult(actions=actions, state_values=values, runtime_sec=perf_counter() - start, metadata={})


class PolicyIterationPolicy(Policy):
    name = "policy_iteration"

    def solve(self, context: PolicyContext) -> PolicyResult:
        start = perf_counter()
        num_states = context.transition_matrix.shape[0]
        actions = np.full(num_states, context.zero_state_index, dtype=int)
        values = np.zeros(num_states, dtype=float)
        while True:
            prev_actions = actions.copy()
            values = evaluate_policy(actions, context.transition_matrix, context.cost_params, context.zero_state_index, context.coordinates, context.cell_dist, context.hop_distances)
            for state_index in range(num_states):
                best_value = float("inf")
                for action in _context_action_candidates(state_index, context):
                    q = _immediate_cost(state_index, int(action), context) + context.cost_params.gamma * float(context.transition_matrix[int(action) - 1] @ values)
                    if q < best_value:
                        best_value = q
                        actions[state_index] = int(action)
            if np.array_equal(actions, prev_actions):
                break
        return PolicyResult(actions=actions, state_values=values, runtime_sec=perf_counter() - start, metadata={})


class ModifiedPolicyIterationPolicy(Policy):
    name = "modified_policy_iteration"

    def solve(self, context: PolicyContext) -> PolicyResult:
        start = perf_counter()
        p = context.transition_matrix
        num_states = p.shape[0]
        zero_state_index = context.zero_state_index
        actions = np.full(num_states, zero_state_index, dtype=int)
        values = np.zeros(num_states, dtype=float)

        p_out_state_first = float(p[0, 1]) if num_states > 1 else 0.0
        p_forward = float(p[1, 2]) if num_states > 2 else 0.0
        p_back = float(p[1, 0]) if num_states > 1 else 0.0
        gamma = context.cost_params.gamma
        power_factor = context.cost_params.power_factor
        const_factor_migrate = context.cost_params.const_factor_migrate
        proportional_factor_migrate = context.cost_params.proportional_factor_migrate
        const_factor_trans = context.cost_params.const_factor_trans
        proportional_factor_trans = context.cost_params.proportional_factor_trans

        alpha1 = gamma * p_back / (1 - gamma * (1 - p_forward - p_back))
        alpha2 = gamma * p_forward / (1 - gamma * (1 - p_forward - p_back))
        alpha3 = const_factor_trans / (1 - gamma * (1 - p_forward - p_back))
        alpha4 = proportional_factor_trans / (1 - gamma * (1 - p_forward - p_back))

        disc = sqrt(max(0.0, 1 - 4 * alpha1 * alpha2))
        m1 = (1 + disc) / (2 * alpha2) if alpha2 != 0 else float("inf")
        m2 = (1 - disc) / (2 * alpha2) if alpha2 != 0 else 0.0
        D = alpha3 / (1 - alpha1 - alpha2)
        B = alpha4 / (1 - alpha1 / power_factor - alpha2 * power_factor)

        if zero_state_index == 1:
            alpha0 = (gamma * p_out_state_first) / (1 - gamma * (1 - p_out_state_first))
            while True:
                prev_actions = actions.copy()
                highest = _first_less(actions)
                if highest is None:
                    highest = num_states
                    actions[-1] = num_states - 1 if num_states > 1 else 1
                right_map = int(actions[highest - 1])
                a = np.array([
                    [1 - alpha0 * m1, 1 - alpha0 * m2],
                    [m1 ** (highest - 1) - m1 ** (right_map - 1), m2 ** (highest - 1) - m2 ** (right_map - 1)],
                ], dtype=float)
                const = np.array([
                    D * (alpha0 - 1) + B * (alpha0 * power_factor - 1),
                    const_factor_migrate + proportional_factor_migrate * power_factor ** (highest - right_map) - B * (power_factor ** (highest - 1) - power_factor ** abs(right_map - 1)),
                ], dtype=float)
                a1, a2 = np.linalg.solve(a, const)
                for i in range(1, highest + 1):
                    values[i - 1] = a1 * m1 ** (i - 1) + a2 * m2 ** (i - 1) + D + B * power_factor ** (i - 1)

                while True:
                    prev_highest = highest
                    highest = _first_less(actions, prev_highest + 1)
                    if highest is None:
                        break
                    right_map = int(actions[highest - 1])
                    if right_map <= prev_highest:
                        a = np.array([
                            [m1 ** (prev_highest - 1), m2 ** (prev_highest - 1)],
                            [m1 ** (highest - 1), m2 ** (highest - 1)],
                        ], dtype=float)
                        const = np.array([
                            values[prev_highest - 1] - D - B * power_factor ** abs(prev_highest - 1),
                            const_factor_migrate + proportional_factor_migrate * power_factor ** (highest - right_map) + values[right_map - 1] - D - B * power_factor ** (highest - 1),
                        ], dtype=float)
                    else:
                        a = np.array([
                            [m1 ** (prev_highest - 1), m2 ** (prev_highest - 1)],
                            [m1 ** (highest - 1) - m1 ** abs(right_map - 1), m2 ** (highest - 1) - m2 ** abs(right_map - 1)],
                        ], dtype=float)
                        const = np.array([
                            values[prev_highest - 1] - D - B * power_factor ** (prev_highest - 1),
                            const_factor_migrate + proportional_factor_migrate * power_factor ** (highest - right_map) - B * (power_factor ** (highest - 1) - power_factor ** abs(right_map - 1)),
                        ], dtype=float)
                    a1, a2 = np.linalg.solve(a, const)
                    for i in range(prev_highest + 1, highest + 1):
                        values[i - 1] = a1 * m1 ** abs(i - 1) + a2 * m2 ** abs(i - 1) + D + B * power_factor ** abs(i - 1)

                for s_mobile in range(1, num_states + 1):
                    best_value = float("inf")
                    for action in _context_action_candidates(s_mobile - 1, context):
                        q = (const_factor_migrate + proportional_factor_migrate * power_factor ** abs(int(action) - s_mobile)) * int(s_mobile != int(action))
                        q += (const_factor_trans + proportional_factor_trans * power_factor ** abs(zero_state_index - int(action))) * int(zero_state_index != int(action))
                        q += gamma * float(p[int(action) - 1] @ values)
                        if q < best_value:
                            best_value = q
                            actions[s_mobile - 1] = int(action)
                if np.array_equal(actions, prev_actions):
                    break
        else:
            m1r = (1 + disc) / (2 * alpha1) if alpha1 != 0 else float("inf")
            m2r = (1 - disc) / (2 * alpha1) if alpha1 != 0 else 0.0
            Br = alpha4 / (1 - alpha2 / power_factor - alpha1 * power_factor)
            H = (alpha1 * (D * (1 - m2r) + Br * (power_factor - m2r)) + alpha2 * (D * (1 - m2) + B * (power_factor - m2))) / disc
            while True:
                prev_actions = actions.copy()
                lowest = _last_greater(actions)
                highest = _first_less(actions)
                if lowest is None:
                    lowest = 1
                if highest is None:
                    highest = num_states
                left_map = int(actions[lowest - 1])
                right_map = int(actions[highest - 1])

                abs_m = zero_state_index - lowest
                abs_am = abs(left_map - zero_state_index)
                if left_map <= zero_state_index:
                    v0_left = m1r ** abs_m - m1r ** abs_am
                    a1_left = m2r ** abs_m - m1r ** abs_m + m1r ** abs_am - m2r ** abs_am
                    const_left = const_factor_migrate + proportional_factor_migrate * power_factor ** (left_map - lowest) + Br * (power_factor ** abs_am - power_factor ** abs_m) + (D + Br - H) * (m2r ** abs_m - m2r ** abs_am) + H * (m1r ** abs_m - m1r ** abs_am)
                else:
                    v0_left = m1r ** abs_m - m2 ** abs_am
                    a1_left = m2r ** abs_m - m1r ** abs_m + m2 ** abs_am - m1 ** abs_am
                    const_left = const_factor_migrate + proportional_factor_migrate * power_factor ** (left_map - lowest) + B * power_factor ** abs_am - Br * power_factor ** abs_m + (D + Br - H) * (m2r ** abs_m) + H * m1r ** abs_m - (D + B) * m2 ** abs_am

                abs_n = highest - zero_state_index
                abs_an = abs(right_map - zero_state_index)
                if right_map <= zero_state_index:
                    v0_right = m2 ** abs_n - m1r ** abs_an
                    a1_right = m1 ** abs_n - m2 ** abs_n + m1r ** abs_an - m2r ** abs_an
                    const_right = const_factor_migrate + proportional_factor_migrate * power_factor ** (highest - right_map) + Br * power_factor ** abs_an - B * power_factor ** abs_n + (H - D - Br) * m2r ** abs_an - H * m1r ** abs_an + (D + B) * m2 ** abs_n
                else:
                    v0_right = m2 ** abs_n - m2 ** abs_an
                    a1_right = m1 ** abs_n - m2 ** abs_n + m2 ** abs_an - m1 ** abs_an
                    const_right = const_factor_migrate + proportional_factor_migrate * power_factor ** (highest - right_map) + B * (power_factor ** abs_an - power_factor ** abs_n) + (D + B) * (m2 ** abs_n - m2 ** abs_an)

                coeff = np.array([[v0_left, a1_left], [v0_right, a1_right]], dtype=float)
                rhs = np.array([const_left, const_right], dtype=float)
                v0, a1 = np.linalg.solve(coeff, rhs)
                a2 = v0 - a1 - D - B
                a1r = v0 - a1 - H
                a2r = v0 - a1r - D - Br
                for i in range(lowest, highest + 1):
                    if i <= zero_state_index:
                        values[i - 1] = a1r * m1r ** abs(i - zero_state_index) + a2r * m2r ** abs(i - zero_state_index) + D + Br * power_factor ** abs(i - zero_state_index)
                    else:
                        values[i - 1] = a1 * m1 ** (i - zero_state_index) + a2 * m2 ** (i - zero_state_index) + D + B * power_factor ** (i - zero_state_index)

                while True:
                    prev_lowest = lowest
                    lowest = _last_greater(actions, prev_lowest - 1)
                    if lowest is None:
                        break
                    left_map = int(actions[lowest - 1])
                    if left_map >= prev_lowest:
                        a = np.array([
                            [m1r ** abs(prev_lowest - zero_state_index), m2r ** abs(prev_lowest - zero_state_index)],
                            [m1r ** abs(lowest - zero_state_index), m2r ** abs(lowest - zero_state_index)],
                        ], dtype=float)
                        const = np.array([
                            values[prev_lowest - 1] - D - Br * power_factor ** abs(prev_lowest - zero_state_index),
                            const_factor_migrate + proportional_factor_migrate * power_factor ** (left_map - lowest) + values[left_map - 1] - D - Br * power_factor ** abs(lowest - zero_state_index),
                        ], dtype=float)
                    else:
                        a = np.array([
                            [m1r ** abs(prev_lowest - zero_state_index), m2r ** abs(prev_lowest - zero_state_index)],
                            [m1r ** abs(lowest - zero_state_index) - m1r ** abs(left_map - zero_state_index), m2r ** abs(lowest - zero_state_index) - m2r ** abs(left_map - zero_state_index)],
                        ], dtype=float)
                        const = np.array([
                            values[prev_lowest - 1] - D - Br * power_factor ** abs(prev_lowest - zero_state_index),
                            const_factor_migrate + proportional_factor_migrate * power_factor ** (left_map - lowest) - Br * (power_factor ** abs(lowest - zero_state_index) - power_factor ** abs(left_map - zero_state_index)),
                        ], dtype=float)
                    a1r_seg, a2r_seg = np.linalg.solve(a, const)
                    for i in range(lowest, prev_lowest):
                        values[i - 1] = a1r_seg * m1r ** abs(i - zero_state_index) + a2r_seg * m2r ** abs(i - zero_state_index) + D + Br * power_factor ** abs(i - zero_state_index)

                while True:
                    prev_highest = highest
                    highest = _first_less(actions, prev_highest + 1)
                    if highest is None:
                        break
                    right_map = int(actions[highest - 1])
                    if right_map <= prev_highest:
                        a = np.array([
                            [m1 ** (prev_highest - zero_state_index), m2 ** (prev_highest - zero_state_index)],
                            [m1 ** (highest - zero_state_index), m2 ** (highest - zero_state_index)],
                        ], dtype=float)
                        const = np.array([
                            values[prev_highest - 1] - D - B * power_factor ** abs(prev_highest - zero_state_index),
                            const_factor_migrate + proportional_factor_migrate * power_factor ** (highest - right_map) + values[right_map - 1] - D - B * power_factor ** (highest - zero_state_index),
                        ], dtype=float)
                    else:
                        a = np.array([
                            [m1 ** (prev_highest - zero_state_index), m2 ** (prev_highest - zero_state_index)],
                            [m1 ** (highest - zero_state_index) - m1 ** abs(right_map - zero_state_index), m2 ** (highest - zero_state_index) - m2 ** abs(right_map - zero_state_index)],
                        ], dtype=float)
                        const = np.array([
                            values[prev_highest - 1] - D - B * power_factor ** (prev_highest - zero_state_index),
                            const_factor_migrate + proportional_factor_migrate * power_factor ** (highest - right_map) - B * (power_factor ** (highest - zero_state_index) - power_factor ** abs(right_map - zero_state_index)),
                        ], dtype=float)
                    a1_seg, a2_seg = np.linalg.solve(a, const)
                    for i in range(prev_highest + 1, highest + 1):
                        values[i - 1] = a1_seg * m1 ** abs(i - zero_state_index) + a2_seg * m2 ** abs(i - zero_state_index) + D + B * power_factor ** abs(i - zero_state_index)

                for s_mobile in range(1, num_states + 1):
                    best_value = float("inf")
                    for action in _context_action_candidates(s_mobile - 1, context):
                        q = (const_factor_migrate + proportional_factor_migrate * power_factor ** abs(int(action) - s_mobile)) * int(s_mobile != int(action))
                        q += (const_factor_trans + proportional_factor_trans * power_factor ** abs(zero_state_index - int(action))) * int(zero_state_index != int(action))
                        q += gamma * float(p[int(action) - 1] @ values)
                        if q < best_value:
                            best_value = q
                            actions[s_mobile - 1] = int(action)
                if np.array_equal(actions, prev_actions):
                    break

        return PolicyResult(actions=actions, state_values=values, runtime_sec=perf_counter() - start, metadata={"implementation": "difference_equations"})


class NeverMigratePolicy(Policy):
    name = "never_migrate"

    def solve(self, context: PolicyContext) -> PolicyResult:
        start = perf_counter()
        num_states = context.transition_matrix.shape[0]
        actions = np.arange(1, num_states + 1, dtype=int)
        if context.allowed_actions is not None:
            for idx, allowed in enumerate(context.allowed_actions):
                if actions[idx] not in allowed:
                    actions[idx] = context.zero_state_index if context.zero_state_index in allowed else int(allowed[0])
        elif context.coordinates is None:
            actions[0] = context.zero_state_index
            actions[-1] = context.zero_state_index
        values = evaluate_policy(actions, context.transition_matrix, context.cost_params, context.zero_state_index, context.coordinates, context.cell_dist, context.hop_distances)
        return PolicyResult(actions=actions, state_values=values, runtime_sec=perf_counter() - start, metadata={})


class AlwaysMigratePolicy(Policy):
    name = "always_migrate"

    def solve(self, context: PolicyContext) -> PolicyResult:
        start = perf_counter()
        actions = np.full(context.transition_matrix.shape[0], context.zero_state_index, dtype=int)
        values = evaluate_policy(actions, context.transition_matrix, context.cost_params, context.zero_state_index, context.coordinates, context.cell_dist, context.hop_distances)
        return PolicyResult(actions=actions, state_values=values, runtime_sec=perf_counter() - start, metadata={})


class MyopicPolicy(Policy):
    name = "myopic"

    def solve(self, context: PolicyContext) -> PolicyResult:
        start = perf_counter()
        num_states = context.transition_matrix.shape[0]
        actions = np.zeros(num_states, dtype=int)
        for idx in range(num_states):
            if context.coordinates is None:
                distance = abs((idx + 1) - context.zero_state_index)
            else:
                assert context.cell_dist is not None
                if context.hop_distances is None:
                    distance = hop_distance_2d(context.coordinates, idx + 1, context.zero_state_index, context.cell_dist)
                else:
                    distance = int(context.hop_distances[idx, context.zero_state_index - 1])
            if distance == 0:
                actions[idx] = idx + 1
                continue
            migrate_cost = context.cost_params.const_factor_migrate + context.cost_params.proportional_factor_migrate * context.cost_params.power_factor**distance
            trans_cost = context.cost_params.const_factor_trans + context.cost_params.proportional_factor_trans * context.cost_params.power_factor**distance
            actions[idx] = context.zero_state_index if migrate_cost < trans_cost else idx + 1
        if context.coordinates is not None and context.allowed_actions is not None:
            for idx, allowed in enumerate(context.allowed_actions):
                if actions[idx] not in allowed:
                    actions[idx] = idx + 1 if (idx + 1) in allowed else int(allowed[0])
        if context.coordinates is None:
            actions[0] = context.zero_state_index
            actions[-1] = context.zero_state_index
        values = evaluate_policy(actions, context.transition_matrix, context.cost_params, context.zero_state_index, context.coordinates, context.cell_dist, context.hop_distances)
        return PolicyResult(actions=actions, state_values=values, runtime_sec=perf_counter() - start, metadata={})
