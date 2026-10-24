"""
Unit Tests for Evaluation Protocols.
"""

import pytest
import numpy as np

from mm_flex_merge.protocols import (
    ContinualMergingProtocol,
    StatisticalSignificanceProtocol,
)


# ---------------------------------------------------------------------------
# ContinualMergingProtocol
# ---------------------------------------------------------------------------

class TestContinualMergingProtocol:
    def test_evaluate_predictions_perfect(self):
        targets = np.array([0, 1, 2, 0, 1])
        preds = np.array([0, 1, 2, 0, 1])
        acc, f1 = ContinualMergingProtocol.evaluate_predictions(targets, preds)
        assert acc == 100.0
        assert f1 == 100.0

    def test_evaluate_predictions_worst(self):
        targets = np.array([0, 0, 0])
        preds = np.array([1, 1, 1])
        acc, f1 = ContinualMergingProtocol.evaluate_predictions(targets, preds)
        assert acc == 0.0

    def test_bwt_positive_transfer(self):
        """When final merged acc > single-task acc → BWT > 0."""
        single = [80.0, 70.0, 60.0]
        final = [85.0, 75.0, 65.0]
        bwt = ContinualMergingProtocol.compute_bwt(single, final)
        assert bwt > 0

    def test_bwt_catastrophic_forgetting(self):
        """When final merged acc < single-task acc → BWT < 0."""
        single = [90.0, 85.0, 80.0]
        final = [70.0, 65.0, 80.0]
        bwt = ContinualMergingProtocol.compute_bwt(single, final)
        assert bwt < 0

    def test_bwt_single_task(self):
        """BWT with a single task should be 0."""
        bwt = ContinualMergingProtocol.compute_bwt([90.0], [85.0])
        assert bwt == 0.0

    def test_forgetting_non_negative(self):
        single = [90.0, 85.0]
        final = [95.0, 90.0]
        forgetting = ContinualMergingProtocol.compute_forgetting(single, final)
        assert forgetting >= 0.0

    def test_plasticity_ratio(self):
        single = [80.0, 70.0, 90.0]
        final = [80.0, 70.0, 85.0]
        plasticity = ContinualMergingProtocol.compute_plasticity(single, final)
        assert abs(plasticity - 85.0 / 90.0) < 1e-5


# ---------------------------------------------------------------------------
# StatisticalSignificanceProtocol
# ---------------------------------------------------------------------------

class TestStatisticalSignificanceProtocol:
    def test_aggregate_runs(self):
        values = [90.0, 91.0, 89.0, 90.5, 90.0]
        mean, std = StatisticalSignificanceProtocol.aggregate_runs(values)
        assert abs(mean - np.mean(values)) < 1e-6
        assert std > 0

    def test_paired_t_test_same_values(self):
        """Identical distributions should yield finite p = 1.0 (> 0.05) and t = 0.0."""
        vals = [90.0, 91.0, 89.0, 90.5, 90.0]
        result = StatisticalSignificanceProtocol.paired_t_test(vals, vals)
        assert result["t_statistic"] == 0.0
        assert result["p_value"] == 1.0
        assert result["cohens_d"] == 0.0
        assert not result["statistically_significant_p05"]

    def test_paired_t_test_different_values(self):
        """Clearly different distributions should yield p < 0.05."""
        proposed = [95.0, 96.0, 94.0, 95.5, 95.0]
        baseline = [70.0, 71.0, 69.0, 70.5, 70.0]
        result = StatisticalSignificanceProtocol.paired_t_test(proposed, baseline)
        assert result["statistically_significant_p05"]
        assert result["cohens_d"] > 0

    def test_aggregate_single_value(self):
        mean, std = StatisticalSignificanceProtocol.aggregate_runs([85.0])
        assert mean == 85.0
        assert std == 0.0
