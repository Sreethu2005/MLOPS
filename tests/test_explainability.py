"""
Unit tests for global feature importance and local TreeSHAP explainability.
"""
import pytest
import numpy as np
import joblib
import xgboost as xgb
from src.explainability.explainer import (
    compute_global_feature_importance,
    explain_instance,
    generate_shap_summary
)

@pytest.fixture
def dummy_model_and_features():
    np.random.seed(42)
    X = np.random.randn(50, 6)
    y = np.random.binomial(1, 0.3, size=50)
    dtrain = xgb.DMatrix(X, label=y)
    param = {"max_depth": 3, "learning_rate": 0.1, "objective": "binary:logistic"}
    model = xgb.train(param, dtrain, num_boost_round=10)
    feature_names = ["age", "time_in_hospital", "num_medications", "number_inpatient", "num_lab", "num_proc"]
    return model, feature_names, X

def test_global_feature_importance(dummy_model_and_features):
    model, feature_names, _ = dummy_model_and_features
    df = compute_global_feature_importance(model, feature_names)

    assert "feature" in df.columns
    assert "importance" in df.columns
    assert len(df) == len(feature_names)
    assert df["importance"].is_monotonic_decreasing

def test_explain_instance_format(dummy_model_and_features):
    model, feature_names, X = dummy_model_and_features
    single_patient = X[0:1]

    explanation = explain_instance(model, single_patient, feature_names, top_k=3)
    assert "base_value" in explanation
    assert "top_features" in explanation
    assert len(explanation["top_features"]) == 3

    for item in explanation["top_features"]:
        assert item["feature"] in feature_names
        assert isinstance(item["shap_value"], float)
        assert item["direction"] in ["increases_risk", "decreases_risk"]

def test_generate_shap_summary_plot(dummy_model_and_features, tmp_path):
    model, feature_names, X = dummy_model_and_features
    out_file = tmp_path / "test_shap.png"

    results = generate_shap_summary(model, X, feature_names, output_path=str(out_file), top_n=4)
    assert out_file.exists()
    assert out_file.stat().st_size > 0
    assert len(results["top_features"]) == 4
