"""
FastAPI Inference Service for Hospital Readmission Risk Screening.
Academic Decision-Support Prototype Only — Not for Clinical Diagnostic Use.
Provides /predict, /health, /model-info, and /metrics endpoints with logging and SHAP explanations.
"""
from contextlib import asynccontextmanager
from pathlib import Path
import hashlib
import json
import time
import uuid
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import xgboost as xgb
import mlflow

from src.data.loader import load_config
from src.features.pipeline import get_transformed_feature_names
from src.explainability.explainer import explain_instance
from src.serving.registry import load_model_by_alias
from src.serving.schemas import (
    PatientEncounterInput,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    MetricsResponse,
    FeatureExplanation,
    ACADEMIC_DISCLAIMER
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# State container
app_state = {
    "model_bundle": None,
    "feature_names": None,
    "config": None,
    "start_time": time.time(),
    "request_latencies": [],
    "total_requests": 0,
    "total_errors": 0,
    "high_risk_count": 0,
    "low_risk_count": 0,
    "predicted_risks": []
}

def log_request_event(log_data: dict):
    """Appends structured JSON request log for operational & drift monitoring."""
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "serving_requests.log"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_data) + "\n")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads configuration and production model bundle upon service startup."""
    logger.info("Initializing FastAPI Inference Service...")
    app_state["start_time"] = time.time()
    try:
        config = load_config("configs/config.yaml")
        app_state["config"] = config
        model_name = config["mlflow"].get("registered_model_name", "HospitalReadmissionClassifier")
        
        # Load champion model by @production alias
        bundle = load_model_by_alias(model_name=model_name, alias="production")
        app_state["model_bundle"] = bundle
        app_state["feature_names"] = get_transformed_feature_names(bundle["preprocessor"], config)
        # Configure MLflow Tracing and Tracking
        try:
            tracking_uri = config.get("mlflow", {}).get("tracking_uri", "http://localhost:5000")
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(config.get("mlflow", {}).get("experiment_name", "hospital_readmission_prediction"))
        except Exception as trace_err:
            logger.warning(f"MLflow tracing tracking URI initialization: {trace_err}")
            
        logger.info(f"Loaded production model '{model_name}' version {bundle['version']} successfully.")
    except Exception as e:
        logger.error(f"Failed to load model during startup: {e}")
        app_state["model_bundle"] = None
    yield
    logger.info("Shutting down inference service.")

app = FastAPI(
    title="Hospital Readmission Prediction API (Academic Prototype)",
    description=(
        "Decision-support prototype predicting 30-day readmission risk for diabetic inpatient encounters. "
        "ACADEMIC DECISION-SUPPORT PROTOTYPE ONLY: Not intended for clinical diagnostic, triage, or care decisions."
    ),
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def get_health():
    """Health check endpoint indicating service availability and model loading status."""
    bundle = app_state.get("model_bundle")
    loaded = bundle is not None
    uptime = time.time() - app_state["start_time"]

    return HealthResponse(
        status="healthy" if loaded else "degraded",
        model_loaded=loaded,
        model_name=bundle["model_name"] if loaded else "None",
        model_version=str(bundle["version"]) if loaded else "None",
        model_alias=bundle["alias"] if loaded else "None",
        uptime_seconds=round(uptime, 2)
    )

@app.get("/model-info", response_model=ModelInfoResponse, tags=["Metadata"])
def get_model_info():
    """Returns metadata, version, active threshold, and metrics of currently served model."""
    bundle = app_state.get("model_bundle")
    if not bundle:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not currently loaded.")

    return ModelInfoResponse(
        model_name=bundle["model_name"],
        version=str(bundle["version"]),
        alias=bundle["alias"],
        decision_threshold=bundle["threshold"],
        tags=bundle.get("tags", {}),
        metrics={
            "recall_test": bundle.get("tags", {}).get("recall_test"),
            "precision_test": bundle.get("tags", {}).get("precision_test"),
            "roc_auc_test": bundle.get("tags", {}).get("roc_auc_test"),
            "brier_score_test": bundle.get("tags", {}).get("brier_score_test")
        }
    )

def execute_scoring(bundle: dict, record_dict: dict, feature_names: list):
    """Executes model scoring, calibration, thresholding, and TreeSHAP calculation."""
    df_single = pd.DataFrame([record_dict])
    preprocessor = bundle["preprocessor"]
    X_trans = preprocessor.transform(df_single)
    model = bundle["model"]
    dmat = xgb.DMatrix(X_trans)
    raw_prob = float(model.predict(dmat)[0])
    calibrator = bundle["calibrator"]
    cal_prob = float(calibrator.predict_proba(np.array([raw_prob]))[0]) if calibrator is not None else raw_prob
    threshold = bundle["threshold"]
    predicted_class = 1 if cal_prob >= threshold else 0
    predicted_label = "High Risk (<30 Days Readmission)" if predicted_class == 1 else "Low Risk"
    explanation = explain_instance(model, X_trans, feature_names, top_k=5)
    return cal_prob, predicted_class, predicted_label, threshold, explanation

try:
    traced_scoring = mlflow.trace(name="hospital_readmission_screening")(execute_scoring)
except Exception:
    traced_scoring = execute_scoring

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_readmission(encounter: PatientEncounterInput):
    """
    Predicts 30-day readmission risk for a patient encounter and returns top local SHAP attributions.
    Rejects out-of-schema inputs.
    """
    start_time = time.time()
    req_id = str(uuid.uuid4())
    bundle = app_state.get("model_bundle")

    if not bundle:
        app_state["total_errors"] += 1
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference model is not currently loaded."
        )

    try:
        # 1. Resolve inputs and ICD-9 categories
        record_dict = encounter.resolve_categories()
        feature_names = app_state["feature_names"]

        # 2. Execute scoring with MLflow tracing & fallback
        try:
            cal_prob, predicted_class, predicted_label, threshold, explanation = traced_scoring(
                bundle, record_dict, feature_names
            )
        except Exception as trace_exc:
            logger.warning(f"Tracing fallback triggered: {trace_exc}")
            cal_prob, predicted_class, predicted_label, threshold, explanation = execute_scoring(
                bundle, record_dict, feature_names
            )
        top_explanations = [
            FeatureExplanation(
                feature=item["feature"],
                shap_value=item["shap_value"],
                direction=item["direction"],
                feature_value_scaled=item["feature_value_scaled"]
            )
            for item in explanation["top_features"]
        ]

        latency_ms = round((time.time() - start_time) * 1000, 2)

        # 7. Update operational telemetry
        app_state["total_requests"] += 1
        app_state["request_latencies"].append(latency_ms)
        app_state["predicted_risks"].append(cal_prob)
        if predicted_class == 1:
            app_state["high_risk_count"] += 1
        else:
            app_state["low_risk_count"] += 1

        # 8. Log request event for monitoring
        input_hash = hashlib.sha256(json.dumps(record_dict, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        log_event = {
            "timestamp": time.time(),
            "request_id": req_id,
            "input_hash": input_hash,
            "features": record_dict,
            "probability": round(cal_prob, 4),
            "predicted_class": predicted_class,
            "latency_ms": latency_ms,
            "status_code": 200
        }
        log_request_event(log_event)

        return PredictionResponse(
            readmission_probability=round(cal_prob, 4),
            predicted_class=predicted_class,
            predicted_label=predicted_label,
            decision_threshold=threshold,
            model_version=str(bundle["version"]),
            model_alias=bundle["alias"],
            top_contributing_features=top_explanations,
            latency_ms=latency_ms,
            request_id=req_id
        )

    except Exception as e:
        latency_ms = round((time.time() - start_time) * 1000, 2)
        app_state["total_errors"] += 1
        logger.error(f"Inference error for request {req_id}: {e}")
        log_request_event({
            "timestamp": time.time(),
            "request_id": req_id,
            "error": str(e),
            "latency_ms": latency_ms,
            "status_code": 500
        })
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
def get_monitoring_metrics():
    """Returns operational and telemetry metrics for system monitoring."""
    latencies = app_state["request_latencies"]
    risks = app_state["predicted_risks"]

    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0
    avg_risk = float(np.mean(risks)) if risks else 0.0

    return MetricsResponse(
        total_requests=app_state["total_requests"],
        total_errors=app_state["total_errors"],
        avg_latency_ms=round(avg_latency, 2),
        p95_latency_ms=round(p95_latency, 2),
        high_risk_predictions=app_state["high_risk_count"],
        low_risk_predictions=app_state["low_risk_count"],
        avg_predicted_risk=round(avg_risk, 4)
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.serving.app:app", host="0.0.0.0", port=8000, reload=True)
