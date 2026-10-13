"""
Unit tests for Data Drift Detection, KS-Test, and Subgroup Fairness Auditing.
"""
import pytest
import numpy as np
import pandas as pd

from src.monitoring.drift_detector import (
    calculate_numerical_psi,
    calculate_categorical_psi,
    calculate_ks_statistic,
    detect_dataset_drift
)
from src.monitoring.fairness_audit import (
    compute_group_metrics,
    audit_demographic_fairness,
    map_age_cohort
)

def test_calculate_numerical_psi_identical():
    """Identical distributions should yield near-zero PSI and 'stable' status."""
    np.random.seed(42)
    ref = np.random.normal(loc=10.0, scale=2.0, size=1000)
    curr = np.random.normal(loc=10.0, scale=2.0, size=1000)

    res = calculate_numerical_psi(ref, curr)
    assert res["psi"] < 0.10
    assert res["status"] == "stable"

def test_calculate_numerical_psi_shifted():
    """Heavily shifted distribution should yield high PSI and 'critical' status."""
    np.random.seed(42)
    ref = np.random.normal(loc=10.0, scale=2.0, size=1000)
    curr = np.random.normal(loc=20.0, scale=2.0, size=1000) # +10 shift

    res = calculate_numerical_psi(ref, curr)
    assert res["psi"] >= 0.25
    assert res["status"] == "critical"

def test_calculate_categorical_psi():
    """Categorical PSI properly evaluates balanced and shifted categories."""
    ref = pd.Series(["A"] * 500 + ["B"] * 500)
    curr = pd.Series(["A"] * 500 + ["B"] * 500)
    res_stable = calculate_categorical_psi(ref, curr)
    assert res_stable["psi"] < 0.10
    assert res_stable["status"] == "stable"

    # Heavily skewed
    curr_skewed = pd.Series(["A"] * 950 + ["B"] * 50)
    res_skewed = calculate_categorical_psi(ref, curr_skewed)
    assert res_skewed["psi"] >= 0.25
    assert res_skewed["status"] == "critical"

def test_calculate_ks_statistic():
    """Kolmogorov-Smirnov test identifies distribution shift with p-value < 0.05."""
    np.random.seed(42)
    ref = np.random.exponential(scale=2.0, size=500)
    curr = np.random.exponential(scale=8.0, size=500)

    res = calculate_ks_statistic(ref, curr)
    assert res["ks_statistic"] > 0.20
    assert res["p_value"] < 0.05
    assert res["drift_detected"] is True

def test_detect_dataset_drift_report_structure():
    """detect_dataset_drift generates complete dictionary with expected summary fields."""
    np.random.seed(42)
    ref_df = pd.DataFrame({
        "num_medications": np.random.randint(5, 25, 200),
        "number_inpatient": np.random.randint(0, 4, 200),
        "age": np.random.choice(["[50-60)", "[60-70)", "[70-80)"], 200),
        "gender": np.random.choice(["Male", "Female"], 200)
    })
    curr_df = ref_df.sample(n=100, random_state=42).copy()

    report = detect_dataset_drift(ref_df, curr_df)

    assert "overall_status" in report
    assert report["overall_status"] in ["STABLE", "WARNING", "CRITICAL_DRIFT"]
    assert "feature_drift_details" in report
    assert "num_medications" in report["feature_drift_details"]

def test_map_age_cohort():
    """Verify age mapping partitions into defined clinical groups."""
    assert map_age_cohort("[20-30)") == "Under 50"
    assert map_age_cohort("[60-70)") == "50-69 Years"
    assert map_age_cohort("[80-90)") == "70+ Years (Geriatric)"

def test_audit_demographic_fairness():
    """audit_demographic_fairness calculates subgroup metrics and disparate impact."""
    df = pd.DataFrame({
        "gender": ["Male"] * 50 + ["Female"] * 50,
        "age": ["[40-50)"] * 30 + ["[60-70)"] * 30 + ["[70-80)"] * 40,
        "race": ["Caucasian"] * 80 + ["AfricanAmerican"] * 20
    })
    y_true = np.array([0, 1] * 50)
    y_pred = np.array([0, 1] * 50)
    y_prob = np.array([0.05, 0.25] * 50)

    report = audit_demographic_fairness(df, y_true, y_pred, y_prob)

    assert "overall_test_cohort" in report
    assert "subgroups" in report
    assert "gender" in report["subgroups"]
    assert "disparity_analysis" in report
    assert "clinical_equity_summary" in report
