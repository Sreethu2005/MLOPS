"""
Unit tests for model training components, calibration, and evaluation metrics.
"""
import pytest
import numpy as np
from src.evaluation.metrics import (
    calculate_expected_calibration_error,
    evaluate_predictions,
    find_optimal_threshold,
    compute_calibration_curve
)
from src.training.train import SigmoidCalibrator

def test_evaluate_predictions_metrics():
    y_true = np.array([1, 1, 0, 0, 1, 0, 0, 1])
    y_prob = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.3, 0.4, 0.6])

    metrics = evaluate_predictions(y_true, y_prob, threshold=0.5)
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["brier_score"] < 0.1
    assert "ece" in metrics

def test_expected_calibration_error():
    # Perfectly calibrated predictions: confidence matches accuracy
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.5, 0.5, 0.5, 0.5])
    ece = calculate_expected_calibration_error(y_true, y_prob, n_bins=5)
    assert ece == pytest.approx(0.0, abs=1e-5)

def test_compute_calibration_curve():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    prob_true, prob_pred = compute_calibration_curve(y_true, y_prob, n_bins=2)
    assert len(prob_true) == 2
    assert prob_true[0] == 0.0
    assert prob_true[1] == 1.0

def test_sigmoid_calibrator():
    # Model with poorly calibrated unscaled probabilities
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.15, size=200)
    raw_probs = np.random.uniform(0.3, 0.8, size=200) # Severely overconfident

    calibrator = SigmoidCalibrator()
    calibrator.fit(raw_probs, y_true)
    cal_probs = calibrator.predict_proba(raw_probs)

    assert len(cal_probs) == 200
    assert (cal_probs >= 0.0).all() and (cal_probs <= 1.0).all()
    # Calibrated mean should be much closer to the true empirical base rate (~0.15)
    assert np.mean(cal_probs) < np.mean(raw_probs)

def test_find_optimal_threshold():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    y_prob = np.array([0.8, 0.6, 0.4, 0.35, 0.2, 0.15, 0.1, 0.05, 0.05, 0.01])

    # Screening test asking for recall >= 0.60
    t = find_optimal_threshold(y_true, y_prob, target_recall=0.60, min_precision=0.15)
    metrics = evaluate_predictions(y_true, y_prob, threshold=t)
    assert metrics["recall"] >= 0.60
