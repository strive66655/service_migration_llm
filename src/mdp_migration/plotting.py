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
