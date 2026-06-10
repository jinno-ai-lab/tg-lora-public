"""Multi-seed statistical analysis for TG-LoRA paper experiments.

Provides confidence intervals, paired t-tests, Cohen's d effect sizes,
and aggregate multi-seed analysis from experiment summaries.
"""
from __future__ import annotations

import math
from typing import Any

from scipy import stats as scipy_stats


def confidence_interval(
    data: list[float], confidence: float = 0.95
) -> tuple[float, float, float]:
    """Compute confidence interval for a sample.

    Returns (mean, lower_bound, upper_bound).
    For n=1, returns (value, value, value).
    Raises ValueError for empty input.
    """
    if not data:
        raise ValueError("data must not be empty")
    n = len(data)
    mean = sum(data) / n
    if n == 1:
        return (mean, mean, mean)

    std_err = float(scipy_stats.sem(data))
    if std_err == 0.0:
        return (mean, mean, mean)

    lower, upper = scipy_stats.t.interval(
        confidence,
        df=n - 1,
        loc=mean,
        scale=std_err,
    )

    return (mean, float(lower), float(upper))


def paired_t_test(
    baseline: list[float], treatment: list[float]
) -> tuple[float, float]:
    """Perform a paired t-test.

    Returns (t_statistic, p_value).
    Raises ValueError if lengths differ or fewer than 2 pairs.
    """
    if len(baseline) != len(treatment):
        raise ValueError("baseline and treatment must have same length")
    n = len(baseline)
    if n < 2:
        raise ValueError("need at least 2 pairs for paired t-test")

    diffs = [t - b for b, t in zip(baseline, treatment)]
    mean_diff = sum(diffs) / n
    var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)

    if var_diff == 0:
        if mean_diff == 0:
            return (0.0, 1.0)
        return (math.copysign(float("inf"), mean_diff), 0.0)

    result = scipy_stats.ttest_rel(treatment, baseline)
    t_stat = float(result.statistic)
    p_value = float(result.pvalue)

    return (t_stat, max(0.0, min(1.0, p_value)))


def cohens_d(baseline: list[float], treatment: list[float]) -> float:
    """Compute Cohen's d effect size between two independent samples.

    Positive d means treatment > baseline.
    """
    if not baseline or not treatment:
        raise ValueError("both groups must be non-empty")

    n1, n2 = len(baseline), len(treatment)
    mean1 = sum(baseline) / n1
    mean2 = sum(treatment) / n2

    var1 = sum((x - mean1) ** 2 for x in baseline) / max(n1 - 1, 1)
    var2 = sum((x - mean2) ** 2 for x in treatment) / max(n2 - 1, 1)

    if n1 + n2 > 2:
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    else:
        pooled_var = (var1 + var2) / 2

    pooled_std = math.sqrt(pooled_var)
    if pooled_std == 0:
        return 0.0

    return (mean2 - mean1) / pooled_std


def analyze_multi_seed(aggregate_summary: dict[str, Any]) -> dict[str, Any]:
    """Analyze multi-seed experiment results with statistical summaries.

    Expects aggregate_summary with 'per_seed' and 'aggregate' keys.
    Returns dict with per-metric statistics including CI, std, and mean.
    """
    per_seed = aggregate_summary.get("per_seed", {})

    if not per_seed:
        return {"metrics": {}, "seed_count": 0}

    metric_keys = _extract_metric_keys(per_seed)
    seed_count = len(per_seed)
    metrics_stats: dict[str, Any] = {}

    for key in metric_keys:
        values = []
        for seed_data in per_seed.values():
            val = seed_data.get(key)
            if val is not None:
                values.append(float(val))

        if not values:
            continue

        entry: dict[str, Any] = {
            "n": len(values),
            "mean": sum(values) / len(values),
        }

        if len(values) >= 2:
            ci = confidence_interval(values)
            entry["ci_lower"] = ci[1]
            entry["ci_upper"] = ci[2]
            var = sum((v - entry["mean"]) ** 2 for v in values) / (len(values) - 1)
            entry["std"] = math.sqrt(var)

        metrics_stats[key] = entry

    return {
        "metrics": metrics_stats,
        "seed_count": seed_count,
    }


def _extract_metric_keys(per_seed: dict[str, Any]) -> list[str]:
    keys: set[str] = set()
    for seed_data in per_seed.values():
        if isinstance(seed_data, dict):
            keys.update(
                k for k, v in seed_data.items() if isinstance(v, (int, float))
            )
    return sorted(keys)

