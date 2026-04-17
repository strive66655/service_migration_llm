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
from mdp_migration.plotting import (
    plot_single_user_llm_batch_results,
    plot_single_user_llm_multi_agent_diagnostics,
    plot_single_user_llm_parameter_trace,
    plot_single_user_llm_results,
    plot_single_user_llm_tradeoff,
)
from mdp_migration.single_user_llm import SingleUserLLMConfig, run_single_user_llm_loop

PRIMARY_METRICS = ["evaluation_cost", "avg_service_distance", "avg_migration_count", "jitter_ratio"]
AUX_METRICS = ["distance_violation_ratio", "avg_migration_distance"]
EXPLANATORY_METRICS = ["run_cost"]
SCENARIO_REVIEW_GUIDES = {
    "balanced": {
        "primary_metrics": ["avg_service_distance", "avg_migration_count"],
        "aux_metrics": ["evaluation_cost"],
        "judgement_focus": "Check whether service distance and migration count reach a reasonable trade-off.",
    },
    "latency": {
        "primary_metrics": ["avg_service_distance", "distance_violation_ratio"],
        "aux_metrics": ["avg_migration_count"],
        "judgement_focus": "Check whether service stays close to the user and distance violations remain low; extra migrations are acceptable but should stay controlled.",
    },
    "stability": {
        "primary_metrics": ["jitter_ratio", "avg_migration_count"],
        "aux_metrics": ["avg_service_distance"],
        "judgement_focus": "Check whether switching jitter and migration frequency are suppressed; service distance can worsen slightly but not too much.",
    },
    "delay_tolerant": {
        "primary_metrics": ["avg_migration_count", "run_cost"],
        "aux_metrics": ["avg_service_distance"],
        "judgement_focus": "Check whether migration count and system cost are reduced without over-migrating just to keep the service closer.",
    },
    "conflict": {
        "primary_metrics": ["avg_service_distance", "avg_migration_count", "jitter_ratio"],
        "aux_metrics": ["evaluation_cost"],
        "judgement_focus": "Check whether distance, migration count, and jitter achieve a balanced compromise rather than optimizing only one metric.",
    },
}


def _print_batch_progress(
    *,
    completed_runs: int,
    total_runs: int,
    scenario_name: str,
    seed: int,
) -> None:
    width = 24
    ratio = completed_runs / max(total_runs, 1)
    filled = int(ratio * width)
    bar = "#" * filled + "-" * (width - filled)
    print(
        f"\r[{bar}] batch {completed_runs}/{total_runs} | scenario={scenario_name} | seed={seed}",
        end="",
        file=sys.stderr,
        flush=True,
    )


def _scenario_configs(
    use_2d: bool,
    steps: int,
    seeds: list[int],
    llm_refresh_interval: int,
    controller_mode: str,
    llm_backend: str,
    llm_model: str,
    llm_api_base: str,
    llm_api_key_env: str,
    llm_timeout_sec: float,
    scenario_names: list[str] | None = None,
):
    base = {
        "use_2d": use_2d,
        "num_steps": steps,
        "llm_refresh_interval": llm_refresh_interval,
        "controller_mode": controller_mode,
        "llm_backend": llm_backend,
        "llm_model": llm_model,
        "llm_api_base": llm_api_base,
        "llm_api_key_env": llm_api_key_env,
        "llm_timeout_sec": llm_timeout_sec,
    }
    scenarios = {
        "balanced": {
            "business_profile": "balanced",
            "operator_text": "Maintain overall service quality while balancing shorter service distance with controlled migration frequency.",
            "failure_mode": None,
        },
        "latency": {
            "business_profile": "latency_sensitive",
            "operator_text": "This is an AR navigation session. Prioritize low latency, keep the service close to the user, and keep distance-threshold violations low. Necessary migrations are acceptable but must remain controlled.",
            "failure_mode": None,
        },
        "stability": {
            "business_profile": "high_stability_required",
            "operator_text": "The current position shift may be temporary. Prioritize service stability, suppress switching jitter and migration frequency, and allow a modest distance trade-off if needed.",
            "failure_mode": None,
        },
        "delay_tolerant": {
            "business_profile": "delay_tolerant",
            "operator_text": "This workload is delay tolerant. Prioritize fewer migrations and lower system cost, and avoid over-migrating just to keep the service slightly closer.",
            "failure_mode": None,
        },
        "conflict": {
            "business_profile": "balanced",
            "operator_text": "This scenario contains conflicting goals: keep service distance low, avoid too many migrations, and suppress switching jitter. Seek a balanced compromise across all three.",
            "failure_mode": None,
        },
        "timeout_fallback": {
            "business_profile": "latency_sensitive",
            "operator_text": "This is an AR navigation session. Prioritize low latency, keep the service close to the user, and keep distance-threshold violations low. Necessary migrations are acceptable but must remain controlled.",
            "failure_mode": "timeout",
        },
    }
    if scenario_names is not None:
        requested = set(scenario_names)
        unknown = sorted(requested - set(scenarios))
        if unknown:
            raise ValueError(f"Unknown scenarios: {unknown}")
        scenarios = {name: scenarios[name] for name in scenarios if name in requested}

    expanded = {}
    for name, override in scenarios.items():
        review_guide = SCENARIO_REVIEW_GUIDES.get(name)
        if review_guide is not None:
            override["review_guide"] = review_guide
        config_override = {key: value for key, value in override.items() if key != "review_guide"}
        expanded[name] = [SingleUserLLMConfig(sim_seed=seed, **base, **config_override) for seed in seeds]
    return expanded, scenarios


def _average_summaries(runs: list[dict]) -> dict:
    methods = list(runs[0]["method_summaries"].keys())
    averaged = {}
    for method in methods:
        keys = list(runs[0]["method_summaries"][method].keys())
        averaged[method] = {
            key: float(sum(run["method_summaries"][method][key] for run in runs) / len(runs)) for key in keys
        }
    return averaged


def _summary_std(runs: list[dict]) -> dict:
    methods = list(runs[0]["method_summaries"].keys())
    deviations = {}
    for method in methods:
        keys = list(runs[0]["method_summaries"][method].keys())
        deviations[method] = {}
        for key in keys:
            values = [run["method_summaries"][method][key] for run in runs]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            deviations[method][key] = variance ** 0.5
    return deviations


def _format_metric_cell(mean: float, std: float) -> str:
    return f"{mean:.4f} +- {std:.4f}"


def _write_markdown_tables(aggregated: dict, output_path: Path) -> None:
    scenarios = aggregated["scenarios"]
    methods = list(next(iter(scenarios.values()))["method_summaries"].keys())

    metrics_md = ["# Single-User LLM Metrics", ""]
    for metric_group_name, metric_names in [
        ("Primary Metrics", PRIMARY_METRICS),
        ("Auxiliary Metrics", AUX_METRICS),
        ("Explanatory Metrics", EXPLANATORY_METRICS),
    ]:
        metrics_md.append(f"## {metric_group_name}")
        metrics_md.append("")
        for metric in metric_names:
            metrics_md.append(f"### {metric}")
            metrics_md.append("")
            metrics_md.append("| Scenario | " + " | ".join(methods) + " |")
            metrics_md.append("| --- | " + " | ".join(["---"] * len(methods)) + " |")
            for scenario_name, scenario_result in scenarios.items():
                row = [scenario_name]
                for method in methods:
                    row.append(_format_metric_cell(
                        scenario_result["method_summaries"][method][metric],
                        scenario_result["method_summary_std"][method][metric],
                    ))
                metrics_md.append("| " + " | ".join(row) + " |")
            metrics_md.append("")

    improvement_md = [
        "# LLM Improvement Over MDP Baseline",
        "",
        "| Scenario | Evaluation Cost d% | Service Distance d% | Migration Count d% | Jitter Ratio d% |",
        "| --- | --- | --- | --- | --- |",
    ]
    for scenario_name, scenario_result in scenarios.items():
        baseline = scenario_result["method_summaries"]["mdp_baseline"]
        llm = scenario_result["method_summaries"]["llm_meta_mdp"]

        def pct(metric: str) -> float:
            base = baseline[metric]
            if abs(base) < 1e-12:
                return 0.0
            return (llm[metric] - base) / base * 100.0

        improvement_md.append(
            f"| {scenario_name} | {pct('evaluation_cost'):.2f}% | {pct('avg_service_distance'):.2f}% | {pct('avg_migration_count'):.2f}% | {pct('jitter_ratio'):.2f}% |"
        )

    scenario_md = [
        "# Scenario Definitions",
        "",
        "| Scenario | Business Profile | Primary Metrics | Auxiliary Metrics | Judgement Focus | Operator Text | Failure Mode |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for scenario_name, definition in aggregated["scenario_definitions"].items():
        review_guide = definition.get("review_guide", {})
        scenario_md.append(
            f"| {scenario_name} | {definition['business_profile']} | {', '.join(review_guide.get('primary_metrics', []))} | {', '.join(review_guide.get('aux_metrics', []))} | {review_guide.get('judgement_focus', '')} | {definition['operator_text']} | {definition['failure_mode'] or 'none'} |"
        )

    (output_path / "single_user_llm_metrics_table.md").write_text("\n".join(metrics_md), encoding="utf-8")
    (output_path / "single_user_llm_improvement_table.md").write_text("\n".join(improvement_md), encoding="utf-8")
    (output_path / "single_user_llm_scenarios.md").write_text("\n".join(scenario_md), encoding="utf-8")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _merge_results(base: dict, updates: dict) -> dict:
    merged = dict(base)
    merged_config = dict(base.get("config", {}))
    merged_config.update(updates.get("config", {}))
    merged["config"] = merged_config

    merged_definitions = dict(base.get("scenario_definitions", {}))
    merged_definitions.update(updates.get("scenario_definitions", {}))
    merged["scenario_definitions"] = merged_definitions

    merged_scenarios = dict(base.get("scenarios", {}))
    merged_scenarios.update(updates.get("scenarios", {}))
    merged["scenarios"] = merged_scenarios
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", default="outputs/single_user_llm_batch")
    parser.add_argument("--use-1d", action="store_true")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--llm-refresh-interval", type=int, default=10)
    parser.add_argument("--controller-mode", choices=["single_agent", "multi_agent"], default="single_agent")
    parser.add_argument("--llm-backend", default="mock")
    parser.add_argument("--llm-model", default="openai/gpt-5.4-mini")
    parser.add_argument("--llm-api-base", default="https://openrouter.ai/api/v1")
    parser.add_argument("--llm-api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--llm-timeout-sec", type=float, default=30.0)
    parser.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--merge-into", default=None)
    args = parser.parse_args()

    scenario_runs, scenario_definitions = _scenario_configs(
        not args.use_1d,
        args.steps,
        list(args.seeds),
        args.llm_refresh_interval,
        args.controller_mode,
        args.llm_backend,
        args.llm_model,
        args.llm_api_base,
        args.llm_api_key_env,
        args.llm_timeout_sec,
        list(args.scenarios) if args.scenarios else None,
    )
    aggregated = {
        "config": {
            "use_2d": not args.use_1d,
            "steps": args.steps,
            "llm_refresh_interval": args.llm_refresh_interval,
            "controller_mode": args.controller_mode,
            "llm_backend": args.llm_backend,
            "llm_model": args.llm_model,
            "seeds": list(args.seeds),
        },
        "scenario_definitions": scenario_definitions,
        "scenarios": {},
    }
    representative_result = None
    total_runs = sum(len(configs) for configs in scenario_runs.values())
    completed_runs = 0
    for name, configs in scenario_runs.items():
        runs = []
        for config in configs:
            config.show_progress = args.show_progress
            run = run_single_user_llm_loop(config)
            runs.append(run)
            completed_runs += 1
            if args.show_progress:
                _print_batch_progress(
                    completed_runs=completed_runs,
                    total_runs=total_runs,
                    scenario_name=name,
                    seed=config.sim_seed,
                )
        aggregated["scenarios"][name] = {
            "method_summaries": _average_summaries(runs),
            "method_summary_std": _summary_std(runs),
            "evaluation_metric_definition": runs[0]["evaluation_metric_definition"],
        }
        if representative_result is None or name == "latency":
            representative_result = runs[0]
    if args.show_progress:
        print(file=sys.stderr)

    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    if args.merge_into:
        merge_path = Path(args.merge_into)
        merged = _merge_results(_load_json(merge_path), aggregated)
        aggregated = merged
        representative_path = save_path / "single_user_llm_representative.json"
        if representative_result is None and representative_path.exists():
            representative_result = _load_json(representative_path)
    save_json(save_path / "single_user_llm_batch_results.json", aggregated)
    _write_markdown_tables(aggregated, save_path)
    if representative_result is not None:
        save_json(save_path / "single_user_llm_representative.json", representative_result)
    print(json.dumps(aggregated, indent=2, ensure_ascii=False))
    if args.plot:
        plot_single_user_llm_batch_results(aggregated, str(save_path))
        plot_single_user_llm_tradeoff(aggregated, str(save_path))
        if representative_result is not None:
            plot_single_user_llm_results(representative_result, str(save_path))
            plot_single_user_llm_parameter_trace(representative_result, str(save_path))
            plot_single_user_llm_multi_agent_diagnostics(representative_result, str(save_path))


if __name__ == "__main__":
    main()
