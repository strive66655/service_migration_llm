from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from mdp_migration.core import CostParams, build_1d_transition_matrix, evaluate_policy, hex_grid_coordinates, hex_neighbor_matrix, map_threshold_actions_to_2d
from mdp_migration.policies import ModifiedPolicyIterationPolicy, PolicyContext, PolicyIterationPolicy
from mdp_migration.random_walk import RandomWalkConfig, run_random_walk
import mdp_migration.real_trace as real_trace_module
from mdp_migration.real_trace import RealTraceConfig, run_real_trace


class CoreTests(unittest.TestCase):
    def test_hex_geometry(self) -> None:
        coords, ring_starts = hex_grid_coordinates(2, 1.0, (0.0, 0.0))
        self.assertEqual(coords.shape[0], 19)
        self.assertEqual(ring_starts[1], 2)
        self.assertEqual(ring_starts[2], 8)
        neighbors = hex_neighbor_matrix(coords, 1.0)
        self.assertEqual(int(neighbors[0, 0]), 1)

    def test_policy_iteration_matches_modified_on_small_chain(self) -> None:
        p = build_1d_transition_matrix(4, 0.2, 0.1, 0.7, 0.3, 0.1)
        cost = CostParams(0.9, 0.8, 2.0, -1.0, 1.0, -1.0)
        context = PolicyContext(p, cost, 1, action_mode="distance")
        standard = PolicyIterationPolicy().solve(context)
        modified = ModifiedPolicyIterationPolicy().solve(context)
        self.assertTrue(np.array_equal(standard.actions, modified.actions))

    def test_threshold_mapping(self) -> None:
        coords, ring_starts = hex_grid_coordinates(3, 1.0, (0.0, 0.0))
        mapped = map_threshold_actions_to_2d(np.array([1, 1, 2, 3]), 3, ring_starts, coords, 1.0)
        self.assertEqual(mapped[0], 1)
        self.assertTrue(np.all(mapped[ring_starts[3] - 1 :] == 1))

    def test_policy_evaluation(self) -> None:
        p = np.eye(3)
        cost = CostParams(0.5, 0.8, 1.0, -0.2, 1.0, -0.1)
        values = evaluate_policy(np.array([1, 1, 1]), p, cost, 1)
        self.assertEqual(values.shape, (3,))

    def test_random_walk_1d_returns_matlab_style_metrics(self) -> None:
        result = run_random_walk(
            RandomWalkConfig(
                use_2d=False,
                gamma_vector=(0.9,),
                migrate_proportional_vector=(0.0,),
                sim_seed_vector=(1,),
                num_states_left=2,
                num_states_right=4,
            )
        )
        self.assertIn("value_error", result)
        self.assertIn("different_action_pct", result)
        self.assertEqual(len(result["value_th_policy"]), 1)

    def test_random_walk_2d_returns_matlab_style_metrics(self) -> None:
        result = run_random_walk(
            RandomWalkConfig(
                use_2d=True,
                gamma_vector=(0.9,),
                migrate_proportional_vector=(0.0,),
                sim_seed_vector=(1,),
                num_states_2d=4,
            )
        )
        self.assertIn("value_th_policy", result)
        self.assertGreater(result["value_th_policy"][0][0], 0.0)

    def test_real_trace_smoke(self) -> None:
        original_loader = real_trace_module.load_trace_data

        def limited_load(path):
            data = original_loader(path)
            data["cellOfUsers"] = data["cellOfUsers"][:40, :20]
            data["totalUsers"] = data["totalUsers"][:40]
            return data

        try:
            real_trace_module.load_trace_data = limited_load
            result = run_real_trace(RealTraceConfig())
        finally:
            real_trace_module.load_trace_data = original_loader

        self.assertIn("summary", result)
        self.assertIn("gain_stats", result)
        self.assertEqual(len(result["avg_cost_series"]["threshold"]), 40)


if __name__ == "__main__":
    unittest.main()
