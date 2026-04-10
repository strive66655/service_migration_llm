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
    plot_single_user_llm_parameter_trace,
    plot_single_user_llm_results,
    plot_single_user_llm_tradeoff,
)
from mdp_migration.single_user_llm import SingleUserLLMConfig, run_single_user_llm_loop

PRIMARY_METRICS = ["evaluation_cost", "avg_service_distance", "avg_migration_count", "jitter_ratio"]
AUX_METRICS = ["distance_violation_ratio", "avg_migration_distance"]
EXPLANATORY_METRICS = ["run_cost"]


def _scenario_configs(
    use_2d: bool,
    steps: int,
    seeds: list[int],
    llm_refresh_interval: int,
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
        "llm_backend": llm_backend,
        "llm_model": llm_model,
        "llm_api_base": llm_api_base,
        "llm_api_key_env": llm_api_key_env,
        "llm_timeout_sec": llm_timeout_sec,
    }
    scenarios = {
        "balanced": {
            "business_profile": "balanced",
            "operator_text": "\u5728\u4fdd\u8bc1\u603b\u4f53\u670d\u52a1\u8d28\u91cf\u7684\u524d\u63d0\u4e0b\uff0c\u517c\u987e\u670d\u52a1\u8ddd\u79bb\u4e0e\u8fc1\u79fb\u5f00\u9500\u3002",
            "failure_mode": None,
        },
        "latency": {
            "business_profile": "latency_sensitive",
            "operator_text": "\u5f53\u524d\u4e1a\u52a1\u4e3a AR \u5bfc\u822a\u4f1a\u8bdd\uff0c\u4f18\u5148\u4fdd\u8bc1\u4f4e\u65f6\u5ef6\uff0c\u53ef\u63a5\u53d7\u5fc5\u8981\u8fc1\u79fb\u3002",
            "failure_mode": None,
        },
        "stability": {
            "business_profile": "high_stability_required",
            "operator_text": "\u5f53\u524d\u4f4d\u7f6e\u53ef\u80fd\u53ea\u662f\u77ed\u6682\u504f\u79fb\uff0c\u4f18\u5148\u4fdd\u6301\u670d\u52a1\u7a33\u5b9a\uff0c\u907f\u514d\u9891\u7e41\u8fc1\u79fb\u3002",
            "failure_mode": None,
        },
        "delay_tolerant": {
            "business_profile": "delay_tolerant",
            "operator_text": "\u5f53\u524d\u4e1a\u52a1\u5bf9\u65f6\u5ef6\u4e0d\u654f\u611f\uff0c\u4f18\u5148\u51cf\u5c11\u8fc1\u79fb\u5e26\u6765\u7684\u989d\u5916\u5f00\u9500\u3002",
            "failure_mode": None,
        },
        "conflict": {
            "business_profile": "latency_sensitive",
            "operator_text": "\u5f53\u524d\u4e1a\u52a1\u5bf9\u65f6\u5ef6\u654f\u611f\uff0c\u4f46\u82e5\u4f4d\u7f6e\u504f\u79fb\u53ea\u662f\u77ed\u65f6\u73b0\u8c61\uff0c\u5e94\u907f\u514d\u65e0\u610f\u4e49\u8fc1\u79fb\u3002",
            "failure_mode": None,
        },
        "timeout_fallback": {
            "business_profile": "latency_sensitive",
            "operator_text": "\u5f53\u524d\u4e1a\u52a1\u4e3a AR \u5bfc\u822a\u4f1a\u8bdd\uff0c\u4f18\u5148\u4fdd\u8bc1\u4f4e\u65f6\u5ef6\uff0c\u53ef\u63a5\u53d7\u5fc5\u8981\u8fc1\u79fb\u3002",
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
        expanded[name] = [SingleUserLLMConfig(sim_seed=seed, **base, **override) for seed in seeds]
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
        "| Scenario | Business Profile | Operator Text | Failure Mode |",
        "| --- | --- | --- | --- |",
    ]
    for scenario_name, definition in aggregated["scenario_definitions"].items():
        scenario_md.append(
            f"| {scenario_name} | {definition['business_profile']} | {definition['operator_text']} | {definition['failure_mode'] or 'none'} |"
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
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--llm-refresh-interval", type=int, default=5)
    parser.add_argument("--llm-backend", default="mock")
    parser.add_argument("--llm-model", default="openai/gpt-5.3-chat")
    parser.add_argument("--llm-api-base", default="https://openrouter.ai/api/v1")
    parser.add_argument("--llm-api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--llm-timeout-sec", type=float, default=30.0)
    parser.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--merge-into", default=None)
    args = parser.parse_args()

    scenario_runs, scenario_definitions = _scenario_configs(
        not args.use_1d,
        args.steps,
        list(args.seeds),
        args.llm_refresh_interval,
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
            "llm_backend": args.llm_backend,
            "llm_model": args.llm_model,
            "seeds": list(args.seeds),
        },
        "scenario_definitions": scenario_definitions,
        "scenarios": {},
    }
    representative_result = None
    for name, configs in scenario_runs.items():
        runs = [run_single_user_llm_loop(config) for config in configs]
        aggregated["scenarios"][name] = {
            "method_summaries": _average_summaries(runs),
            "method_summary_std": _summary_std(runs),
            "evaluation_metric_definition": runs[0]["evaluation_metric_definition"],
        }
        if name == "latency":
            representative_result = runs[0]

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


if __name__ == "__main__":
    main()
