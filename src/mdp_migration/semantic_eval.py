from __future__ import annotations

from typing import Any

SEMANTIC_CLIP_MIN = -1.0
SEMANTIC_CLIP_MAX = 1.0
SEMANTIC_POSITIVE_THRESHOLD = 0.05
SEMANTIC_NEGATIVE_THRESHOLD = -0.05
_EPSILON = 1e-12


def relative_improvement(baseline: float, method: float, *, epsilon: float = _EPSILON) -> float:
    baseline_value = float(baseline)
    method_value = float(method)
    if abs(baseline_value) < epsilon:
        if abs(method_value) < epsilon:
            return 0.0
        # Metrics are cost-like, so lower is better. When the baseline is zero,
        # use a finite sentinel instead of an epsilon denominator explosion.
        return 1.0 if method_value < baseline_value else -1.0
    return (baseline_value - method_value) / abs(baseline_value)


def clipped_relative_improvement(raw_improvement: float) -> float:
    return max(SEMANTIC_CLIP_MIN, min(SEMANTIC_CLIP_MAX, float(raw_improvement)))


def semantic_alignment_label(clipped_values: list[float], semantic_score: float) -> str:
    negatives = sum(1 for value in clipped_values if value < 0.0)
    non_negatives = len(clipped_values) - negatives
    has_positive = any(value > 0.0 for value in clipped_values)
    has_negative = negatives > 0
    if clipped_values and all(value >= 0.0 for value in clipped_values) and semantic_score > SEMANTIC_POSITIVE_THRESHOLD:
        return "improved"
    if semantic_score < SEMANTIC_NEGATIVE_THRESHOLD and negatives > non_negatives:
        return "not_aligned"
    if SEMANTIC_NEGATIVE_THRESHOLD <= semantic_score <= SEMANTIC_POSITIVE_THRESHOLD or (has_positive and has_negative):
        return "mixed"
    return "mixed"


def build_semantic_review(method_summaries: dict[str, dict[str, float]], primary_metrics: list[str]) -> dict[str, Any]:
    baseline = method_summaries["mdp_baseline"]
    raw_by_method: dict[str, dict[str, float]] = {}
    clipped_by_method: dict[str, dict[str, float]] = {}
    scores: dict[str, float] = {}
    labels: dict[str, str] = {}

    for method, summary in method_summaries.items():
        raw_improvements = {
            metric: relative_improvement(baseline[metric], summary[metric])
            for metric in primary_metrics
        }
        clipped_improvements = {
            metric: clipped_relative_improvement(value)
            for metric, value in raw_improvements.items()
        }
        clipped_values = list(clipped_improvements.values())
        score = sum(clipped_values) / len(clipped_values) if clipped_values else 0.0
        raw_by_method[method] = raw_improvements
        clipped_by_method[method] = clipped_improvements
        scores[method] = score
        labels[method] = semantic_alignment_label(clipped_values, score)

    return {
        "primary_metrics": list(primary_metrics),
        "semantic_primary_improvements_raw": raw_by_method,
        "semantic_primary_improvements_clipped": clipped_by_method,
        "semantic_scores": scores,
        "semantic_alignment_labels": labels,
        "methods": {
            method: {
                "semantic_primary_improvements_raw": raw_by_method[method],
                "semantic_primary_improvements_clipped": clipped_by_method[method],
                "semantic_consistency_score": scores[method],
                "semantic_alignment_label": labels[method],
            }
            for method in method_summaries
        },
    }
