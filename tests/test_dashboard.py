"""
Unit tests for Streamlit Dashboard helper functions and prediction bridge.
"""
import pytest
from dashboard.app import CLINICAL_PRESETS, predict_encounter

def test_clinical_presets_defined():
    """Verify standard clinical presets exist with required clinical features."""
    assert len(CLINICAL_PRESETS) >= 3
    high_risk_preset = CLINICAL_PRESETS.get("🚨 High Risk: Elderly with Heart Failure & Prior Admissions")
    assert high_risk_preset is not None
    assert high_risk_preset["number_inpatient"] >= 2
    assert "diag_1" in high_risk_preset

def test_predict_encounter_fallback_execution():
    """Verify predict_encounter executes cleanly via local fallback or live API."""
    sample_payload = {
        "race": "Caucasian",
        "gender": "Male",
        "age": "[70-80)",
        "admission_type_id": 1,
        "discharge_disposition_id": 1,
        "admission_source_id": 7,
        "time_in_hospital": 4,
        "num_lab_procedures": 45,
        "num_procedures": 2,
        "num_medications": 18,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 2,
        "diag_1": "414",
        "diag_2": "428",
        "diag_3": "250",
        "number_diagnoses": 8,
        "max_glu_serum": "None",
        "A1Cresult": "None",
        "change": "No",
        "diabetesMed": "Yes"
    }

    result = predict_encounter(sample_payload)

    assert "readmission_probability" in result
    assert 0.0 <= result["readmission_probability"] <= 1.0
    assert result["predicted_class"] in [0, 1]
    assert "top_contributing_features" in result
    assert len(result["top_contributing_features"]) > 0
    assert "disclaimer" in result
    assert "ACADEMIC" in result["disclaimer"]
