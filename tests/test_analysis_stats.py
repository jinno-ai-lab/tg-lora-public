"""Tests for src/analysis/stats.py — multi-seed statistical analysis."""
from __future__ import annotations

import math

import pytest

from src.analysis.stats import (
    analyze_multi_seed,
    confidence_interval,
    cohens_d,
    paired_t_test,
)


# ---------------------------------------------------------------------------
# confidence_interval
# ---------------------------------------------------------------------------


class TestConfidenceInterval:
    def test_known_95_ci(self):
        data = [2.0, 4.0, 6.0, 8.0, 10.0]
        mean, lo, hi = confidence_interval(data, 0.95)
        assert mean == pytest.approx(6.0)
        # t(4, 0.025) ≈ 2.776, std = sqrt(10/5*4) wait let me compute:
        # variance = sum((x-6)^2 for x in data)/(n-1) = (16+4+0+4+16)/4 = 10
        # std_err = sqrt(10/5) = sqrt(2) ≈ 1.414
        # t_crit ≈ 2.776
        # margin ≈ 2.776 * 1.414 ≈ 3.926
        assert lo < mean
        assert hi > mean
        margin = hi - lo
        expected_margin = 2 * 2.776 * math.sqrt(2)
        assert margin == pytest.approx(expected_margin, rel=0.05)

    def test_single_value_degenerate(self):
        mean, lo, hi = confidence_interval([42.0])
        assert mean == 42.0
        assert lo == 42.0
        assert hi == 42.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            confidence_interval([])

    def test_two_values(self):
        mean, lo, hi = confidence_interval([1.0, 3.0])
        assert mean == pytest.approx(2.0)
        assert lo < mean
        assert hi > mean

    def test_99_ci_wider_than_95(self):
        data = [5.0, 7.0, 9.0, 11.0]
        _, lo95, hi95 = confidence_interval(data, 0.95)
        _, lo99, hi99 = confidence_interval(data, 0.99)
        assert (hi99 - lo99) > (hi95 - lo95)

    def test_identical_values_zero_width(self):
        mean, lo, hi = confidence_interval([5.0, 5.0, 5.0, 5.0])
        assert mean == 5.0
        assert lo == pytest.approx(5.0)
        assert hi == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# paired_t_test
# ---------------------------------------------------------------------------


class TestPairedTTest:
    def test_identical_data_t_zero_p_one(self):
        t, p = paired_t_test([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert t == pytest.approx(0.0)
        assert p == pytest.approx(1.0)

    def test_clear_positive_effect(self):
        baseline = [10.0, 12.0, 11.0, 13.0, 10.5]
        treatment = [15.0, 17.0, 16.0, 18.0, 14.5]
        t, p = paired_t_test(baseline, treatment)
        assert t > 0
        assert p < 0.05

    def test_clear_negative_effect(self):
        baseline = [15.0, 17.0, 16.0, 18.0, 14.5]
        treatment = [10.0, 12.0, 11.0, 13.0, 10.5]
        t, p = paired_t_test(baseline, treatment)
        assert t < 0
        assert p < 0.05

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            paired_t_test([1.0, 2.0], [1.0])

    def test_fewer_than_two_pairs_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            paired_t_test([1.0], [2.0])

    def test_p_value_bounded(self):
        t, p = paired_t_test([1.0, 100.0], [50.0, 2.0])
        assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# cohens_d
# ---------------------------------------------------------------------------


class TestCohensD:
    def test_identical_groups_d_zero(self):
        d = cohens_d([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert d == pytest.approx(0.0)

    def test_positive_effect(self):
        d = cohens_d([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        assert d > 0

    def test_negative_effect(self):
        d = cohens_d([4.0, 5.0, 6.0], [1.0, 2.0, 3.0])
        assert d < 0

    def test_known_effect_size(self):
        # Two groups with mean difference 2, pooled std ≈ 1
        # Group 1: [0,1,2], Group 2: [2,3,4]
        # mean1=1, mean2=3, var1=1, var2=1, pooled_std=1
        # d = (3-1)/1 = 2.0
        d = cohens_d([0.0, 1.0, 2.0], [2.0, 3.0, 4.0])
        assert d == pytest.approx(2.0)

    def test_empty_baseline_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            cohens_d([], [1.0])

    def test_empty_treatment_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            cohens_d([1.0], [])

    def test_zero_variance_returns_zero(self):
        d = cohens_d([5.0, 5.0], [5.0, 5.0])
        assert d == pytest.approx(0.0)

    def test_large_effect_classification(self):
        # Cohen's convention: |d| > 0.8 = large
        d = cohens_d([0.0, 1.0, 2.0], [10.0, 11.0, 12.0])
        assert abs(d) > 0.8  # Large effect


# ---------------------------------------------------------------------------
# analyze_multi_seed
# ---------------------------------------------------------------------------


class TestAnalyzeMultiSeed:
    def _make_summary(
        self, seeds: dict[str, dict[str, float]]
    ) -> dict:
        return {
            "per_seed": seeds,
            "aggregate": {},
        }

    def test_basic_metrics(self):
        summary = self._make_summary({
            "seed_42": {"loss": 1.0, "accuracy": 0.9},
            "seed_123": {"loss": 1.2, "accuracy": 0.88},
            "seed_456": {"loss": 0.8, "accuracy": 0.92},
        })
        result = analyze_multi_seed(summary)
        assert result["seed_count"] == 3
        assert "loss" in result["metrics"]
        assert "accuracy" in result["metrics"]

    def test_ci_computed_for_two_plus_seeds(self):
        summary = self._make_summary({
            "seed_42": {"loss": 1.0},
            "seed_123": {"loss": 1.2},
        })
        result = analyze_multi_seed(summary)
        loss_stats = result["metrics"]["loss"]
        assert "ci_lower" in loss_stats
        assert "ci_upper" in loss_stats
        assert "std" in loss_stats
        assert loss_stats["n"] == 2

    def test_mean_calculation(self):
        summary = self._make_summary({
            "seed_1": {"loss": 1.0},
            "seed_2": {"loss": 3.0},
        })
        result = analyze_multi_seed(summary)
        assert result["metrics"]["loss"]["mean"] == pytest.approx(2.0)

    def test_empty_per_seed(self):
        result = analyze_multi_seed({"per_seed": {}, "aggregate": {}})
        assert result["seed_count"] == 0
        assert result["metrics"] == {}

    def test_missing_per_seed_key(self):
        result = analyze_multi_seed({"aggregate": {}})
        assert result["seed_count"] == 0

    def test_skips_none_values(self):
        summary = self._make_summary({
            "seed_42": {"loss": 1.0},
            "seed_123": {"loss": None},
        })
        result = analyze_multi_seed(summary)
        loss_stats = result["metrics"]["loss"]
        assert loss_stats["n"] == 1
        assert loss_stats["mean"] == pytest.approx(1.0)

    def test_skips_non_numeric_values(self):
        summary = self._make_summary({
            "seed_42": {"loss": 1.0, "name": "run_42"},
            "seed_123": {"loss": 2.0, "name": "run_123"},
        })
        result = analyze_multi_seed(summary)
        assert "loss" in result["metrics"]
        assert "name" not in result["metrics"]

    def test_single_seed_no_ci(self):
        summary = self._make_summary({
            "seed_42": {"loss": 1.0},
        })
        result = analyze_multi_seed(summary)
        loss_stats = result["metrics"]["loss"]
        assert "ci_lower" not in loss_stats
        assert "ci_upper" not in loss_stats
        assert loss_stats["mean"] == pytest.approx(1.0)
