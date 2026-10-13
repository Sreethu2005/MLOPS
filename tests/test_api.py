"""
API Unit and Integration tests for FastAPI serving layer using TestClient.
"""
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from src.serving.app import app

client = TestClient(app)

@pytest.fixture
def sample_valid_encounter():
    return {
        "race": "Caucasian",
        "gender": "Female",
        "age": "[60-70)",
        "admission_type_id": 1,
        "admission_source_id": 7,
        "time_in_hospital": 3,
        "num_lab_procedures": 48,
        "num_procedures": 1,
        "num_medications": 14,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 1,
        "number_diagnoses": 7,
        "max_glu_serum": "None",
        "A1Cresult": ">8",
        "metformin": "Steady",
        "insulin": "Up",
        "change": "Ch",
        "diabetesMed": "Yes",
        "diag_1": "410.1",
        "diag_2": "250.02",
        "diag_3": "428"
    }

def test_health_endpoint():
    with TestClient(app) as tc:
        resp = tc.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "disclaimer" in data
        assert "ACADEMIC" in data["disclaimer"]

def test_model_info_endpoint():
    with TestClient(app) as tc:
        resp = tc.get("/model-info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_name"] == "HospitalReadmissionClassifier"
        assert data["alias"] == "production"
        assert "decision_threshold" in data
        assert "disclaimer" in data

def test_metrics_endpoint():
    with TestClient(app) as tc:
        resp = tc.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_requests" in data
        assert "avg_latency_ms" in data
        assert "high_risk_predictions" in data

def test_predict_valid_encounter(sample_valid_encounter):
    with TestClient(app) as tc:
        resp = tc.post("/predict", json=sample_valid_encounter)
        assert resp.status_code == 200
        data = resp.json()

        assert 0.0 <= data["readmission_probability"] <= 1.0
        assert data["predicted_class"] in [0, 1]
        assert data["predicted_label"] in ["High Risk (<30 Days Readmission)", "Low Risk"]
        assert data["decision_threshold"] > 0.0
        assert len(data["top_contributing_features"]) > 0
        assert "disclaimer" in data
        assert "ACADEMIC DECISION-SUPPORT PROTOTYPE ONLY" in data["disclaimer"]

        # Check SHAP explanation format
        feat = data["top_contributing_features"][0]
        assert "feature" in feat
        assert "shap_value" in feat
        assert feat["direction"] in ["increases_risk", "decreases_risk"]

        # Verify request was written to log file
        log_file = Path("logs/serving_requests.log")
        assert log_file.exists()
        assert log_file.stat().st_size > 0

def test_predict_invalid_schema_rejected(sample_valid_encounter):
    with TestClient(app) as tc:
        # 1. Invalid time_in_hospital (exceeds max of 14)
        invalid_payload = sample_valid_encounter.copy()
        invalid_payload["time_in_hospital"] = 25
        resp = tc.post("/predict", json=invalid_payload)
        assert resp.status_code == 422

        # 2. Invalid gender (must be Male or Female)
        invalid_payload = sample_valid_encounter.copy()
        invalid_payload["gender"] = "Unknown"
        resp = tc.post("/predict", json=invalid_payload)
        assert resp.status_code == 422

        # 3. Negative values
        invalid_payload = sample_valid_encounter.copy()
        invalid_payload["num_medications"] = -5
        resp = tc.post("/predict", json=invalid_payload)
        assert resp.status_code == 422
