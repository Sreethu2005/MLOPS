"""
Unit tests for reproducible data pipeline, anti-leakage protections, and feature pipeline.
"""
import pytest
import numpy as np
import pandas as pd
from src.data.loader import categorize_icd9, patient_grouped_split
from src.data.validator import validate_data_quality
from src.features.pipeline import build_preprocessor, transform_data

@pytest.fixture
def sample_config():
    return {
        "project": {"random_seed": 42},
        "data": {
            "patient_id_col": "patient_nbr",
            "target_col": "readmitted_30d",
            "leakage_discharge_disposition_ids": [11, 13, 14, 19, 20, 21],
            "split": {"train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2}
        },
        "features": {
            "numerical_cols": ["time_in_hospital", "num_medications", "num_lab_procedures"],
            "categorical_cols": ["race", "gender", "age", "diag_1_category"]
        }
    }

@pytest.fixture
def mock_dataset():
    data = []
    # 20 unique patients, some with multiple encounters
    for p in range(1, 21):
        num_encounters = 2 if p % 3 == 0 else 1
        for enc in range(num_encounters):
            data.append({
                "encounter_id": p * 100 + enc,
                "patient_nbr": p,
                "race": "Caucasian" if p % 2 == 0 else "AfricanAmerican",
                "gender": "Female" if p % 3 == 0 else "Male",
                "age": "[50-60)",
                "diag_1_category": "Circulatory",
                "time_in_hospital": (p % 10) + 1,
                "num_medications": 10 + enc,
                "num_lab_procedures": 25,
                "discharge_disposition_id": 1,
                "readmitted_30d": 1 if p % 4 == 0 else 0
            })
    return pd.DataFrame(data)

def test_icd9_categorization():
    assert categorize_icd9("250.01") == "Diabetes"
    assert categorize_icd9("410.1") == "Circulatory"
    assert categorize_icd9("493.9") == "Respiratory"
    assert categorize_icd9("571.2") == "Digestive"
    assert categorize_icd9("599.0") == "Genitourinary"
    assert categorize_icd9("V45") == "Other"
    assert categorize_icd9(None) == "Missing"
    assert categorize_icd9("?") == "Missing"

def test_patient_grouped_split_zero_leakage(mock_dataset, sample_config):
    train_df, val_df, test_df = patient_grouped_split(mock_dataset, sample_config)

    train_patients = set(train_df["patient_nbr"])
    val_patients = set(val_df["patient_nbr"])
    test_patients = set(test_df["patient_nbr"])

    # Strict assertion: ZERO patient overlap
    assert len(train_patients.intersection(val_patients)) == 0
    assert len(train_patients.intersection(test_patients)) == 0
    assert len(val_patients.intersection(test_patients)) == 0
    assert len(train_df) + len(val_df) + len(test_df) == len(mock_dataset)

def test_feature_pipeline_fit_and_transform(mock_dataset, sample_config):
    preprocessor = build_preprocessor(sample_config)
    
    # Fit strictly on train
    preprocessor.fit(mock_dataset)
    X_trans, y = transform_data(mock_dataset, preprocessor, sample_config)

    assert X_trans.shape[0] == len(mock_dataset)
    assert not np.isnan(X_trans).any(), "Found unexpected NaNs in transformed output"
    assert y is not None and len(y) == len(mock_dataset)

def test_unseen_categories_graceful_handling(mock_dataset, sample_config):
    preprocessor = build_preprocessor(sample_config)
    preprocessor.fit(mock_dataset)

    # DataFrame with never-before-seen categorical values
    new_data = mock_dataset.iloc[:2].copy()
    new_data["race"] = "Extraterrestrial"
    new_data["diag_1_category"] = "UnknownCondition"

    X_new, _ = transform_data(new_data, preprocessor, sample_config)
    assert X_new.shape[0] == 2
    assert not np.isnan(X_new).any()
