"""
Unit tests for MLflow model registry and serving loading by alias.
"""
import pytest
import numpy as np
import pandas as pd
from src.serving.registry import load_model_by_alias
from src.features.pipeline import transform_data

def test_load_model_by_alias():
    loaded = load_model_by_alias(model_name="HospitalReadmissionClassifier", alias="production")
    
    assert loaded["model_name"] == "HospitalReadmissionClassifier"
    assert loaded["alias"] == "production"
    assert int(loaded["version"]) >= 1
    assert loaded["model"] is not None
    assert loaded["preprocessor"] is not None
    assert loaded["threshold"] == pytest.approx(0.10, abs=0.05)
    assert "disclaimer" in loaded["tags"]

def test_loaded_model_prediction_inference():
    loaded = load_model_by_alias(model_name="HospitalReadmissionClassifier", alias="production")
    model = loaded["model"]
    preprocessor = loaded["preprocessor"]
    calibrator = loaded["calibrator"]

    # Sample input DataFrame conforming to feature schema
    sample_df = pd.DataFrame([{
        "race": "Caucasian",
        "gender": "Female",
        "age": "[60-70)",
        "admission_type_id": 1,
        "admission_source_id": 7,
        "time_in_hospital": 3,
        "num_lab_procedures": 45,
        "num_procedures": 1,
        "num_medications": 14,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 1,
        "number_diagnoses": 7,
        "max_glu_serum": "None",
        "A1Cresult": "None",
        "metformin": "No",
        "repaglinide": "No",
        "glimepiride": "No",
        "glipizide": "Steady",
        "glyburide": "No",
        "pioglitazone": "No",
        "rosiglitazone": "No",
        "insulin": "Steady",
        "change": "Ch",
        "diabetesMed": "Yes",
        "diag_1_category": "Circulatory",
        "diag_2_category": "Diabetes",
        "diag_3_category": "Respiratory"
    }])

    import xgboost as xgb
    X_trans = preprocessor.transform(sample_df)
    dmat = xgb.DMatrix(X_trans)
    raw_prob = model.predict(dmat)[0]

    assert 0.0 <= raw_prob <= 1.0

    if calibrator is not None:
        cal_prob = calibrator.predict_proba(np.array([raw_prob]))[0]
        assert 0.0 <= cal_prob <= 1.0
