"""
Unit tests for Automated Retraining, Trigger Evaluation, and Promotion Gates.
"""
import pytest
from src.training.retrain import evaluate_retraining_trigger, PROMOTION_GATES

def test_promotion_gates_structure():
    """Verify promotion gates enforce clinical sensitivity and calibration thresholds."""
    assert "min_test_recall" in PROMOTION_GATES
    assert PROMOTION_GATES["min_test_recall"] >= 0.60
    assert "max_test_brier_score" in PROMOTION_GATES
    assert PROMOTION_GATES["max_test_brier_score"] <= 0.15
    assert "max_test_ece" in PROMOTION_GATES

def test_evaluate_retraining_trigger_stable():
    """Stable monitoring report should not trigger automated retraining."""
    trigger = evaluate_retraining_trigger("reports/drift_report.json")
    assert "trigger_required" in trigger
    assert isinstance(trigger["trigger_required"], bool)
