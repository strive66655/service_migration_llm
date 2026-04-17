from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdp_migration.io import save_json
from mdp_migration.plotting import plot_single_user_llm_multi_agent_diagnostics, plot_single_user_llm_parameter_trace, plot_single_user_llm_results
from mdp_migration.single_user_llm import SingleUserLLMConfig, run_single_user_llm_loop


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--use-1d", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--llm-refresh-interval", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--migrate-proportional", type=float, default=1.0)
    parser.add_argument("--llm-backend", default="mock")
    parser.add_argument("--llm-model", default="openai/gpt-5.4-mini")
    parser.add_argument("--llm-api-base", default="https://openrouter.ai/api/v1")
    parser.add_argument("--llm-api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--llm-timeout-sec", type=float, default=30.0)
    parser.add_argument("--controller-mode", choices=["single_agent", "multi_agent"], default="single_agent")
    parser.add_argument("--business-profile", default="balanced")
    parser.add_argument("--operator-text", default="")
    parser.add_argument("--failure-mode", default=None)
    parser.add_argument("--num-states-left", type=int, default=0)
    parser.add_argument("--num-states-right", type=int, default=10)
    parser.add_argument("--num-states-2d", type=int, default=6)
    args = parser.parse_args()

    config = SingleUserLLMConfig(
        use_2d=not args.use_1d,
        sim_seed=args.seed,
        gamma=args.gamma,
        migrate_proportional=args.migrate_proportional,
        num_steps=args.steps,
        llm_refresh_interval=args.llm_refresh_interval,
        failure_mode=args.failure_mode,
        llm_backend=args.llm_backend,
        llm_model=args.llm_model,
        llm_api_base=args.llm_api_base,
        llm_api_key_env=args.llm_api_key_env,
        llm_timeout_sec=args.llm_timeout_sec,
        controller_mode=args.controller_mode,
        show_progress=args.show_progress,
        business_profile=args.business_profile,
        operator_text=args.operator_text,
        num_states_left=args.num_states_left,
        num_states_right=args.num_states_right,
        num_states_2d=args.num_states_2d,
    )
    results = run_single_user_llm_loop(config)
    print(
        json.dumps(
            {
                "method_summaries": results["method_summaries"],
                "evaluation_metric_definition": results["evaluation_metric_definition"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        save_json(Path(args.save_dir) / "single_user_llm_results.json", results)
    if args.plot:
        plot_output_dir = args.save_dir
        plot_single_user_llm_results(results, plot_output_dir)
        plot_single_user_llm_parameter_trace(results, plot_output_dir)
        plot_single_user_llm_multi_agent_diagnostics(results, plot_output_dir)


if __name__ == "__main__":
    main()
