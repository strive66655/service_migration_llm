from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def _finite_series_values(series: list[np.ndarray]) -> np.ndarray:
    values = np.concatenate([np.asarray(s, dtype=float).reshape(-1) for s in series])
    return values[np.isfinite(values)]


def _filter_series_by_x_limits(
    sim_param_vector: np.ndarray,
    series: list[np.ndarray],
    x_limits: tuple[float, float] | None,
) -> list[np.ndarray]:
    if x_limits is None:
        return [np.asarray(s, dtype=float) for s in series]
    xmin, xmax = x_limits
    sim_param_vector = np.asarray(sim_param_vector, dtype=float)
    mask = (sim_param_vector >= xmin) & (sim_param_vector < xmax)
    if np.any(np.isclose(sim_param_vector, xmax)):
        mask = mask | np.isclose(sim_param_vector, xmax)
    else:
        right_candidates = np.flatnonzero(sim_param_vector > xmax)
        if right_candidates.size > 0:
            mask[right_candidates[0]] = True
    return [np.asarray(s, dtype=float)[mask] for s in series]


def _auto_ylim(series: list[np.ndarray]) -> tuple[float, float] | None:
    finite = _finite_series_values(series)
    if finite.size == 0:
        return None
    data_min = float(np.min(finite))
    data_max = float(np.max(finite))
    data_range = data_max - data_min
    if data_range == 0:
        pad = max(abs(data_max) * 0.05, 0.1)
    else:
        pad = data_range * 0.08
    lower = min(0.0, data_min - pad)
    upper = data_max + pad
    if lower == upper:
        upper = lower + 1.0
    return lower, upper


def plot_random_walk_results(results: dict, output_dir: str | None = None) -> None:
    sim_param_vector = np.asarray(results["sim_param_vector"], dtype=float)
    x_full_limits = (float(np.min(sim_param_vector)), float(np.max(sim_param_vector)))
    figures = [
        ("figure1.png", 0, (0, 2)),
        ("figure2.png", 1, (0, 8)),
        ("figure3.png", 2, x_full_limits),
    ]
    rendered = []
    for filename, gamma_idx, x_limits in figures:
        if gamma_idx >= len(results["time_th_policy"]):
            continue
        fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True, constrained_layout=True)
        axes[0].semilogy(sim_param_vector, results["time_th_policy"][gamma_idx], "-ko", label="Proposed")
        axes[0].semilogy(sim_param_vector, results["time_policy"][gamma_idx], "-.bx", label="Policy iteration")
        axes[0].semilogy(sim_param_vector, results["time_value"][gamma_idx], "--r^", label="Value iteration")
        axes[0].set_xlabel(r"$-\beta_l$")
        axes[0].set_ylabel("Computation time (s)")
        axes[0].legend()

        axes[1].plot(sim_param_vector, results["value_th_policy"][gamma_idx], "-ko", label="Proposed")
        axes[1].plot(sim_param_vector, results["value_policy"][gamma_idx], ":mx", label="Optimal")
        axes[1].plot(sim_param_vector, results["value_never"][gamma_idx], "--g^", label="Never migrate")
        axes[1].plot(sim_param_vector, results["value_always"][gamma_idx], "-.bv", label="Always migrate")
        axes[1].plot(sim_param_vector, results["value_myopic"][gamma_idx], ":rs", label="Myopic")
        axes[1].set_xlabel(r"$-\beta_l$")
        axes[1].set_ylabel("Discounted sum cost")
        axes[1].legend()
        if x_limits is not None:
            axes[1].set_xlim(*x_limits)
        visible_series = _filter_series_by_x_limits(sim_param_vector, [
            results["value_th_policy"][gamma_idx],
            results["value_policy"][gamma_idx],
            results["value_never"][gamma_idx],
            results["value_always"][gamma_idx],
            results["value_myopic"][gamma_idx],
        ], x_limits)
        y_limits = _auto_ylim(visible_series)
        if y_limits is not None:
            axes[1].set_ylim(*y_limits)
            axes[1].yaxis.set_major_locator(MaxNLocator(nbins=6))
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            fig.savefig(Path(output_dir) / filename, dpi=200, bbox_inches="tight", pad_inches=0.08)
            plt.close(fig)
        else:
            rendered.append(fig)
    if not output_dir and rendered:
        plt.show()


def plot_real_trace_results(results: dict, output_dir: str | None = None) -> None:
    x = np.asarray(results.get("time_axis") or np.arange(len(results["avg_cost_series"]["never"])))
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(x, results["avg_cost_series"]["never"], "k", label="Never Migrate (A)")
    ax1.plot(x, results["avg_cost_series"]["always"], "g", label="Always Migrate (B)")
    ax1.plot(x, results["avg_cost_series"]["myopic"], "b", label="Myopic (C)")
    ax1.plot(x, results["avg_cost_series"]["threshold"], "r", label="Proposed (D)")
    ax1.plot(x, np.full_like(x, results["summary"]["never"], dtype=float), "--k")
    ax1.plot(x, np.full_like(x, results["summary"]["always"], dtype=float), "--g")
    ax1.plot(x, np.full_like(x, results["summary"]["myopic"], dtype=float), "--b")
    ax1.plot(x, np.full_like(x, results["summary"]["threshold"], dtype=float), "--r")
    ax1.set_xlabel("Time" if results.get("time_axis") else "Timeslot")
    ax1.set_ylabel("Instantaneous cost")
    ax1.legend()
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot([x[0], x[-1]], [results["summary"]["never"]] * 2, "--k", label="Never")
    ax2.plot([x[0], x[-1]], [results["summary"]["always"]] * 2, "--g", label="Always")
    ax2.plot([x[0], x[-1]], [results["summary"]["myopic"]] * 2, "--b", label="Myopic")
    ax2.plot([x[0], x[-1]], [results["summary"]["threshold"]] * 2, "--r", label="Proposed")
    ax2.set_xlabel("Time" if results.get("time_axis") else "Timeslot")
    ax2.set_ylabel("Average instantaneous cost")
    ax2.legend()
    fig2.tight_layout()

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig1.savefig(Path(output_dir) / "real_trace_costs.png", dpi=200)
        fig2.savefig(Path(output_dir) / "real_trace_averages.png", dpi=200)
        plt.close(fig1)
        plt.close(fig2)
    else:
        plt.show()

def plot_single_user_llm_results(results: dict, output_dir: str | None = None) -> None:
    method_summaries = results["method_summaries"]
    methods = list(method_summaries.keys())
    labels = [m.replace("_", "\n") for m in methods]
    eval_cost = [method_summaries[m]["evaluation_cost"] for m in methods]
    service_distance = [method_summaries[m]["avg_service_distance"] for m in methods]
    migration_count = [method_summaries[m]["avg_migration_count"] for m in methods]
    jitter_ratio = [method_summaries[m]["jitter_ratio"] for m in methods]

    fig1, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes = axes.reshape(-1)
    series = [
        ("Evaluation Cost", eval_cost, "#3b82f6"),
        ("Avg Service Distance", service_distance, "#10b981"),
        ("Avg Migration Count", migration_count, "#f59e0b"),
        ("Jitter Ratio", jitter_ratio, "#ef4444"),
    ]
    x = np.arange(len(methods))
    for ax, (title, values, color) in zip(axes, series):
        ax.bar(x, values, color=color)
        ax.set_title(title)
        ax.set_xticks(x, labels)
        ax.tick_params(axis="x", labelrotation=20)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    llm_trace = results["method_traces"]["llm_meta_mdp"]
    baseline_trace = results["method_traces"]["mdp_baseline"]
    steps = np.arange(len(llm_trace["service_distance"]))
    fig2, axes2 = plt.subplots(2, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    axes2[0].plot(steps, baseline_trace["service_distance"], label="MDP baseline", color="#6366f1")
    axes2[0].plot(steps, llm_trace["service_distance"], label="LLM-Meta-MDP", color="#dc2626")
    axes2[0].set_ylabel("Service Distance")
    axes2[0].legend()

    axes2[1].step(steps, baseline_trace["migration_flag"], where="mid", label="MDP baseline", color="#6366f1")
    axes2[1].step(steps, llm_trace["migration_flag"], where="mid", label="LLM-Meta-MDP", color="#dc2626")
    axes2[1].set_xlabel("Step")
    axes2[1].set_ylabel("Migration Flag")
    axes2[1].legend()

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig1.savefig(Path(output_dir) / "single_user_llm_summary.png", dpi=200, bbox_inches="tight")
        fig2.savefig(Path(output_dir) / "single_user_llm_trace.png", dpi=200, bbox_inches="tight")
        plt.close(fig1)
        plt.close(fig2)
    else:
        plt.show()


def plot_single_user_llm_batch_results(results: dict, output_dir: str | None = None) -> None:
    scenario_names = list(results["scenarios"].keys())
    methods = list(next(iter(results["scenarios"].values()))["method_summaries"].keys())
    metrics = ["evaluation_cost", "avg_service_distance", "avg_migration_count", "jitter_ratio"]
    titles = {
        "evaluation_cost": "Evaluation Cost",
        "avg_service_distance": "Avg Service Distance",
        "avg_migration_count": "Avg Migration Count",
        "jitter_ratio": "Jitter Ratio",
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes = axes.reshape(-1)
    width = 0.14
    x = np.arange(len(scenario_names))
    for ax, metric in zip(axes, metrics):
        for idx, method in enumerate(methods):
            values = [results["scenarios"][scenario]["method_summaries"][method][metric] for scenario in scenario_names]
            errors = [results["scenarios"][scenario]["method_summary_std"][method][metric] for scenario in scenario_names]
            ax.bar(
                x + (idx - (len(methods) - 1) / 2) * width,
                values,
                width=width,
                yerr=errors,
                capsize=3,
                label=method if metric == metrics[0] else None,
            )
        ax.set_title(titles[metric])
        ax.set_xticks(x, scenario_names)
        ax.tick_params(axis="x", labelrotation=20)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    axes[0].legend()

    fig2, ax2 = plt.subplots(figsize=(12, 5), constrained_layout=True)
    llm_values = [results["scenarios"][scenario]["method_summaries"]["llm_meta_mdp"]["evaluation_cost"] for scenario in scenario_names]
    llm_errors = [results["scenarios"][scenario]["method_summary_std"]["llm_meta_mdp"]["evaluation_cost"] for scenario in scenario_names]
    baseline_values = [results["scenarios"][scenario]["method_summaries"]["mdp_baseline"]["evaluation_cost"] for scenario in scenario_names]
    baseline_errors = [results["scenarios"][scenario]["method_summary_std"]["mdp_baseline"]["evaluation_cost"] for scenario in scenario_names]
    ax2.errorbar(scenario_names, baseline_values, yerr=baseline_errors, marker="o", capsize=4, label="MDP baseline", color="#6366f1")
    ax2.errorbar(scenario_names, llm_values, yerr=llm_errors, marker="o", capsize=4, label="LLM-Meta-MDP", color="#dc2626")
    ax2.set_ylabel("Evaluation Cost")
    ax2.set_title("LLM vs MDP Across Scenarios")
    ax2.legend()
    ax2.tick_params(axis="x", labelrotation=20)

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(output_dir) / "single_user_llm_batch_summary.png", dpi=200, bbox_inches="tight")
        fig2.savefig(Path(output_dir) / "single_user_llm_batch_compare.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        plt.close(fig2)
    else:
        plt.show()


def plot_single_user_llm_tradeoff(results: dict, output_dir: str | None = None) -> None:
    scenario_names = list(results["scenarios"].keys())
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    baseline_x = [results["scenarios"][scenario]["method_summaries"]["mdp_baseline"]["avg_migration_count"] for scenario in scenario_names]
    baseline_y = [results["scenarios"][scenario]["method_summaries"]["mdp_baseline"]["avg_service_distance"] for scenario in scenario_names]
    llm_x = [results["scenarios"][scenario]["method_summaries"]["llm_meta_mdp"]["avg_migration_count"] for scenario in scenario_names]
    llm_y = [results["scenarios"][scenario]["method_summaries"]["llm_meta_mdp"]["avg_service_distance"] for scenario in scenario_names]
    ax.scatter(baseline_x, baseline_y, color="#6366f1", label="MDP baseline", s=80)
    ax.scatter(llm_x, llm_y, color="#dc2626", label="LLM-Meta-MDP", s=80)
    for idx, scenario in enumerate(scenario_names):
        ax.annotate(scenario, (llm_x[idx], llm_y[idx]), textcoords="offset points", xytext=(6, 6))
    ax.set_xlabel("Average Migration Count")
    ax.set_ylabel("Average Service Distance")
    ax.set_title("Trade-off Between Migration Frequency and Service Distance")
    ax.legend()
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(output_dir) / "single_user_llm_tradeoff.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_single_user_llm_parameter_trace(results: dict, output_dir: str | None = None) -> None:
    decisions = results.get("llm_decisions", [])
    if not decisions:
        return
    steps = [decision["step"] for decision in decisions]
    gamma = [decision["validated_control"]["gamma"] for decision in decisions]
    migration_weight = [decision["validated_control"]["migration_weight"] for decision in decisions]
    transmission_weight = [decision["validated_control"]["transmission_weight"] for decision in decisions]
    solver_mode_map = {"myopic": 0, "threshold": 1, "mdp": 2}
    solver_values = [solver_mode_map.get(decision["validated_control"]["solver_mode"], -1) for decision in decisions]

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    axes[0].plot(steps, gamma, marker="o", color="#2563eb")
    axes[0].set_ylabel("gamma")
    axes[1].plot(steps, migration_weight, marker="o", color="#d97706")
    axes[1].set_ylabel("migration weight")
    axes[2].plot(steps, transmission_weight, marker="o", color="#059669")
    axes[2].set_ylabel("transmission weight")
    axes[3].step(steps, solver_values, where="mid", color="#7c3aed")
    axes[3].set_yticks([0, 1, 2], ["myopic", "threshold", "mdp"])
    axes[3].set_ylabel("solver")
    axes[3].set_xlabel("Decision Refresh Step")
    fig.suptitle("LLM Control Parameter Evolution")
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(output_dir) / "single_user_llm_parameter_trace.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
