"""Unit Tests for Clinical Evaluation Metrics, Calibration, and Threshold Optimization."""

import numpy as np
import pytest
from ml.loss import (
    calculate_clinical_metrics,
    ThresholdOptimizer,
    continuous_to_probabilities,
)
from ml.calibration import compute_ece


def test_calculate_clinical_metrics_perfect_predictions():
    """Perfect predictions should yield QWK = 1.0, Accuracy = 1.0, and Ref Sens/Spec = 1.0."""
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    y_pred = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

    metrics = calculate_clinical_metrics(y_true, y_pred)
    assert metrics["quadratic_weighted_kappa"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["referable_dr"]["sensitivity"] == 1.0
    assert metrics["referable_dr"]["specificity"] == 1.0


def test_referable_dr_metrics():
    """Verify sensitivity & specificity calculations for Grade >= 2."""
    # Grades: [0, 1, 2, 3] -> Binary: [0, 0, 1, 1]
    y_true = np.array([0, 1, 2, 3])
    # Predicted: [0, 2, 2, 1] -> Binary: [0, 1, 1, 0] (1 FP on grade 1, 1 FN on grade 3)
    y_pred = np.array([0, 2, 2, 1])

    metrics = calculate_clinical_metrics(y_true, y_pred)
    # TP = 1 (grade 2), FN = 1 (grade 3), Sensitivity = 1/2 = 0.5
    assert metrics["referable_dr"]["sensitivity"] == 0.5
    # TN = 1 (grade 0), FP = 1 (grade 1), Specificity = 1/2 = 0.5
    assert metrics["referable_dr"]["specificity"] == 0.5


def test_threshold_optimizer_discretization():
    optimizer = ThresholdOptimizer(initial_thresholds=[0.5, 1.5, 2.5, 3.5])
    preds = np.array([-0.2, 0.4, 0.5, 1.49, 1.51, 2.8, 3.7])
    discrete = optimizer.discretize(preds)
    expected = np.array([0, 0, 1, 1, 2, 3, 4])
    np.testing.assert_array_equal(discrete, expected)


def test_continuous_to_probabilities():
    # Value close to 2.0 should have peak probability at index 2
    probs = continuous_to_probabilities(2.0, sigma=0.65)
    assert len(probs) == 5
    assert np.isclose(np.sum(probs), 1.0)
    assert np.argmax(probs) == 2


def test_compute_ece():
    probs = np.array([
        [0.8, 0.2],
        [0.9, 0.1],
        [0.7, 0.3],
        [0.6, 0.4],
    ])
    labels = np.array([0, 0, 0, 0])
    ece_result = compute_ece(probs, labels, n_bins=5)
    assert "ece" in ece_result
    assert ece_result["ece"] >= 0.0
    assert ece_result["ece"] <= 1.0
