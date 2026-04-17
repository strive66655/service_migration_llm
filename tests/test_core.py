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
from mdp_migration.llm import DEFAULT_SAFE_CONTROL, apply_control_params, build_llm_state, build_prompt, build_shared_control_state, query_llm, query_multi_agent_control, validate_llm_output
from mdp_migration.policies import ModifiedPolicyIterationPolicy, PolicyContext, PolicyIterationPolicy
from mdp_migration.random_walk import RandomWalkConfig, run_random_walk
import mdp_migration.real_trace as real_trace_module
from mdp_migration.real_trace import RealTraceConfig, run_real_trace
from mdp_migration.plotting import plot_single_user_llm_multi_agent_diagnostics
from mdp_migration.single_user_llm import SingleUserLLMConfig, run_single_user_llm_loop


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

    def test_llm_validation_clips_and_falls_back(self) -> None:
        validated = validate_llm_output(
            {
                "objective_mode": "latency_first",
                "gamma": 1.4,
                "migration_weight": -2,
                "transmission_weight": 3,
                "reason": "test",
            },
            DEFAULT_SAFE_CONTROL,
        )
        self.assertAlmostEqual(validated.gamma, 0.99)
        self.assertAlmostEqual(validated.migration_weight, 0.5)
        self.assertTrue(validated.used_fallback)

    def test_llm_prompt_and_query(self) -> None:
        llm_state = build_llm_state(
            {"state_index": 3, "service_index": 3, "distance_to_user": 2, "recent_direction": 1},
            {"recent_service_distances": [1, 2], "recent_migrations": [0, 1]},
            "latency_sensitive",
            "当前业务对时延敏感，可接受必要迁移",
        )
        prompt = build_prompt(llm_state)
        raw = query_llm(prompt, state=llm_state)
        self.assertEqual(raw["objective_mode"], "latency_first")
        self.assertNotIn("solver_mode", raw)

    def test_multi_agent_forecast_schema_and_merge(self) -> None:
        shared_state = build_shared_control_state(
            {"state_index": 3, "service_index": 3, "distance_to_user": 3, "recent_direction": 2},
            {"recent_service_distances": [1, 1], "recent_migrations": [1, 1]},
            "latency_sensitive",
            "latency critical",
        )
        result = query_multi_agent_control(shared_state, backend="mock")
        self.assertEqual(result["forecaster_output"]["distance_trend"], "moving_away")
        self.assertIn(result["forecaster_output"]["mobility_level"], {"medium", "high"})
        self.assertIn(result["final_decision_source"], {"multi_agent_merged", "fallback_partial"})
        self.assertIn("final_safe_control", result)

    def test_multi_agent_forecast_invalid_enum_partially_falls_back(self) -> None:
        shared_state = build_shared_control_state(
            {"state_index": 2, "service_index": 2, "distance_to_user": 2, "recent_direction": 0},
            {"recent_service_distances": [2], "recent_migrations": [0]},
            "balanced",
            "",
        )
        result = query_multi_agent_control(shared_state, backend="mock", failure_mode="invalid_enum")
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["final_decision_source"], "fallback_partial")
        self.assertEqual(result["forecaster_output"]["mobility_level"], "low")
        self.assertTrue(result["validation_notes"])
        self.assertIn("mobility_level_fallback", result["validation_notes"])

    def test_policy_advisor_mock_accepts_shared_state_protocol(self) -> None:
        shared_state = build_shared_control_state(
            {"state_index": 3, "service_index": 3, "distance_to_user": 3, "recent_direction": 2},
            {"recent_service_distances": [1, 1], "recent_migrations": [1, 1]},
            "latency_sensitive",
            "latency critical",
        )
        result = query_multi_agent_control(shared_state, backend="mock")
        self.assertGreaterEqual(result["policy_advisor_output"]["transmission_weight"], 1.45)
        self.assertEqual(result["final_safe_control"]["solver_mode"], "mdp")

    def test_latency_profile_is_not_pulled_back_by_stability_rule(self) -> None:
        shared_state = build_shared_control_state(
            {"state_index": 4, "service_index": 4, "distance_to_user": 4, "recent_direction": 2},
            {"recent_service_distances": [1, 2, 3], "recent_migrations": [1, 1, 1]},
            "latency_sensitive",
            "latency critical AR traffic",
        )
        result = query_multi_agent_control(shared_state, backend="mock")
        self.assertEqual(result["final_safe_control"]["objective_mode"], "latency_first")
        self.assertGreaterEqual(result["final_safe_control"]["transmission_weight"], 1.45)
        self.assertLessEqual(result["final_safe_control"]["migration_weight"], 0.95)

    def test_apply_control_params_changes_weights(self) -> None:
        base = CostParams(0.9, 0.8, 2.0, -1.0, 1.0, -0.5)
        controlled = apply_control_params(
            base,
            validate_llm_output(
                {
                    "objective_mode": "stability_first",
                    "gamma": 0.8,
                    "migration_weight": 1.5,
                    "transmission_weight": 0.75,
                    "reason": "stable",
                }
            ),
        )
        self.assertAlmostEqual(controlled.const_factor_migrate, 3.0)
        self.assertAlmostEqual(controlled.const_factor_trans, 0.75)
        self.assertAlmostEqual(controlled.gamma, 0.8)

    def test_single_user_llm_loop_reports_fixed_evaluation_metrics(self) -> None:
        result = run_single_user_llm_loop(
            SingleUserLLMConfig(
                use_2d=False,
                sim_seed=1,
                num_steps=12,
                llm_refresh_interval=3,
                num_states_left=2,
                num_states_right=4,
                business_profile="latency_sensitive",
                operator_text="当前业务对时延敏感，可接受必要迁移",
            )
        )
        self.assertIn("llm_meta_mdp", result["method_summaries"])
        self.assertIn("evaluation_metric_definition", result)
        self.assertIn("evaluation_cost", result["method_summaries"]["llm_meta_mdp"])
        self.assertGreater(len(result["llm_decisions"]), 0)
        self.assertIn("forecaster", result["llm_decisions"][0]["agent_metrics"])
        self.assertIn("single_agent", result["llm_decisions"][0]["agent_metrics"])

    def test_single_user_llm_timeout_falls_back(self) -> None:
        result = run_single_user_llm_loop(
            SingleUserLLMConfig(
                use_2d=False,
                sim_seed=2,
                num_steps=6,
                llm_refresh_interval=2,
                num_states_left=1,
                num_states_right=3,
                failure_mode="timeout",
            )
        )
        self.assertTrue(result["llm_decisions"][0]["validated_control"]["used_fallback"])

    def test_single_user_llm_multi_agent_loop_reports_diagnostics(self) -> None:
        result = run_single_user_llm_loop(
            SingleUserLLMConfig(
                use_2d=False,
                sim_seed=3,
                num_steps=6,
                llm_refresh_interval=2,
                num_states_left=1,
                num_states_right=3,
                controller_mode="multi_agent",
                business_profile="latency_sensitive",
                operator_text="latency critical",
            )
        )
        first_decision = result["llm_decisions"][0]
        self.assertEqual(first_decision["controller_mode"], "multi_agent")
        self.assertIn("shared_control_state", first_decision)
        self.assertIn("final_decision_source", first_decision)
        self.assertIn("agent_agreement", first_decision)

    def test_multi_agent_diagnostics_plot_is_generated(self) -> None:
        result = run_single_user_llm_loop(
            SingleUserLLMConfig(
                use_2d=False,
                sim_seed=4,
                num_steps=6,
                llm_refresh_interval=2,
                num_states_left=1,
                num_states_right=3,
                controller_mode="multi_agent",
            )
        )
        output_dir = ROOT / "outputs" / "test_multi_agent_plot"
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_single_user_llm_multi_agent_diagnostics(result, str(output_dir))
        self.assertTrue((output_dir / "single_user_llm_multi_agent_diagnostics.png").exists())


if __name__ == "__main__":
    unittest.main()
