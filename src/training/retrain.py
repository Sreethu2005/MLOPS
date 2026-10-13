"""
Automated Model Retraining, Champion-Challenger Evaluation & Promotion Pipeline.
Enforces quantitative quality gates before promoting new candidate models to production.

ACADEMIC DECISION-SUPPORT PROTOTYPE ONLY - Not for clinical diagnostic use.
"""
from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow
from mlflow.tracking import MlflowClient

from src.data.loader import load_config
from src.features.pipeline import (
    fit_and_save_preprocessor,
    transform_data,
    get_transformed_feature_names
)
from src.evaluation.metrics import (
    evaluate_predictions,
    find_optimal_threshold,
    SigmoidCalibrator
)
from src.serving.registry import register_best_model, load_model_by_alias
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Quantitative Promotion Gate Criteria
PROMOTION_GATES = {
    "min_test_recall": 0.60,        # Must catch >=60% of at-risk readmissions
    "min_test_roc_auc": 0.60,       # Discrimination capability threshold
    "max_test_brier_score": 0.12,   # Probability calibration reliability
    "max_test_ece": 0.05,           # Expected Calibration Error limit
    "max_roc_auc_drop": 0.03        # Maximum allowable drop vs current champion
}

def evaluate_retraining_trigger(drift_report_path: str = "reports/drift_report.json") -> dict:
    """
    Checks if active monitoring reports indicate a retraining trigger:
    - Critical feature drift (PSI >= 0.25)
    - Warning drift across > 2 features
    """
    drift_file = Path(drift_report_path)
    if not drift_file.exists():
        return {"trigger_required": False, "reason": "No drift report found; manual or scheduled trigger required."}

    with open(drift_file, "r", encoding="utf-8") as f:
        rep = json.load(f)

    crit = rep.get("features_critical", 0)
    warn = rep.get("features_warning", 0)

    if crit > 0:
        return {
            "trigger_required": True,
            "reason": f"Critical data drift detected ({crit} features with PSI >= 0.25). Immediate retraining required."
        }
    elif warn >= 3:
        return {
            "trigger_required": True,
            "reason": f"Multiple feature drift warnings detected ({warn} features with PSI >= 0.10)."
        }

    return {"trigger_required": False, "reason": "Distributions stable within calibrated thresholds."}

def run_retraining_pipeline(
    config_path: str = "configs/config.yaml",
    force: bool = False,
    output_log_path: str = "reports/retraining_log.json"
) -> dict:
    """
    Executes end-to-end retraining, validation, and promotion workflow:
    1. Evaluates retraining triggers (or checks force flag).
    2. Trains and calibrates candidate XGBoost model.
    3. Evaluates against Champion model from MLflow registry.
    4. Applies Champion-Challenger quality gates.
    5. Promotes or rolls back based on gate results.
    """
    config = load_config(config_path)
    trigger_eval = evaluate_retraining_trigger()

    if not trigger_eval["trigger_required"] and not force:
        logger.info(f"Retraining not triggered: {trigger_eval['reason']}")
        return {
            "status": "SKIPPED",
            "reason": trigger_eval["reason"],
            "promoted": False
        }

    logger.info(f"Initiating model retraining pipeline. Trigger reason: {trigger_eval['reason']}")

    # 1. Load data splits
    train_df = pd.read_parquet(config["data"]["train_data_path"])
    val_df = pd.read_parquet(config["data"]["val_data_path"])
    test_df = pd.read_parquet(config["data"]["test_data_path"])

    # 2. Refit candidate preprocessor
    candidate_preprocessor = fit_and_save_preprocessor(train_df, config, output_path="models/candidate_preprocessor.joblib")
    X_train, y_train = transform_data(train_df, candidate_preprocessor, config)
    X_val, y_val = transform_data(val_df, candidate_preprocessor, config)
    X_test, y_test = transform_data(test_df, candidate_preprocessor, config)

    # 3. Train Candidate Model (XGBoost)
    xgb_params = config["training"]["models"]["xgboost"]
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    candidate_model = xgb.train(
        params={
            "max_depth": xgb_params.get("max_depth", 4),
            "learning_rate": xgb_params.get("learning_rate", 0.05),
            "subsample": xgb_params.get("subsample", 0.8),
            "colsample_bytree": xgb_params.get("colsample_bytree", 0.8),
            "scale_pos_weight": xgb_params.get("scale_pos_weight", 7.9),
            "eval_metric": "logloss",
            "random_state": 42
        },
        dtrain=dtrain,
        num_boost_round=xgb_params.get("n_estimators", 150),
        evals=[(dtrain, "train"), (dval, "val")],
        verbose_eval=False
    )

    # 4. Calibrate Candidate Model
    val_raw_probs = candidate_model.predict(dval)
    calibrator = SigmoidCalibrator()
    calibrator.fit(val_raw_probs, y_val)
    val_cal_probs = calibrator.predict_proba(val_raw_probs)

    # Optimal threshold for screening sensitivity >= 60%
    chosen_threshold = find_optimal_threshold(y_val, val_cal_probs, target_recall=0.60, min_precision=0.15)

    # 5. Evaluate on untouched Test set
    test_raw_probs = candidate_model.predict(dtest)
    test_cal_probs = calibrator.predict_proba(test_raw_probs)
    candidate_metrics = evaluate_predictions(y_test, test_cal_probs, threshold=chosen_threshold)

    logger.info(
        f"Candidate Model Test Metrics: Recall={candidate_metrics['recall']:.4f}, "
        f"Precision={candidate_metrics['precision']:.4f}, ROC-AUC={candidate_metrics['roc_auc']:.4f}, "
        f"Brier={candidate_metrics['brier_score']:.4f}, ECE={candidate_metrics['ece']:.4f}"
    )

    # 6. Retrieve Active Champion Metrics
    champion_roc_auc = 0.6273 # Default baseline from phase 2
    try:
        champ_bundle = load_model_by_alias(alias="production")
        if champ_bundle.get("tags"):
            champion_roc_auc = float(champ_bundle["tags"].get("roc_auc_test", 0.6273))
    except Exception:
        pass

    # 7. Evaluate Champion-Challenger Quality Gates
    gate_checks = {
        "recall_ge_60": bool(candidate_metrics["recall"] >= PROMOTION_GATES["min_test_recall"]),
        "roc_auc_ge_60": bool(candidate_metrics["roc_auc"] >= PROMOTION_GATES["min_test_roc_auc"]),
        "roc_auc_drop_acceptable": bool((champion_roc_auc - candidate_metrics["roc_auc"]) <= PROMOTION_GATES["max_roc_auc_drop"]),
        "brier_score_acceptable": bool(candidate_metrics["brier_score"] <= PROMOTION_GATES["max_test_brier_score"]),
        "ece_acceptable": bool(candidate_metrics["ece"] <= PROMOTION_GATES["max_test_ece"])
    }

    all_gates_passed = all(gate_checks.values())

    # 8. Promotion or Rollback Gate Decision
    if all_gates_passed:
        decision = "PROMOTED"
        logger.info("ALL QUALITY GATES PASSED! Promoting candidate model to Production Champion.")

        # Persist candidate as active best_model and preprocessor
        joblib.dump(candidate_preprocessor, "models/preprocessor.joblib")
        candidate_artifact = {
            "model_name": "XGBoost",
            "model": candidate_model,
            "calibrator": calibrator,
            "threshold": chosen_threshold,
            "metrics": candidate_metrics,
            "promoted_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        joblib.dump(candidate_artifact, "models/best_model.joblib")

        # Register in MLflow Model Registry
        try:
            reg_info = register_best_model(config_path)
            model_version = reg_info.get("version", "N/A")
        except Exception as e:
            logger.warning(f"MLflow remote registration note: {e}")
            model_version = "v_retrained_local"

        action_summary = f"Candidate passed all 5 gates. Promoted as production champion (version {model_version})."
    else:
        decision = "REJECTED_ROLLBACK"
        failed_gates = [k for k, v in gate_checks.items() if not v]
        action_summary = f"Candidate failed quality gates ({', '.join(failed_gates)}). Promotion blocked; champion retained."
        logger.warning(f"PROMOTION BLOCKED: {action_summary}")

    retraining_log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "decision": decision,
        "action_summary": action_summary,
        "trigger_evaluation": trigger_eval,
        "gate_criteria": PROMOTION_GATES,
        "gate_checks": gate_checks,
        "candidate_metrics": candidate_metrics,
        "champion_baseline_roc_auc": champion_roc_auc
    }

    # Save retraining audit log
    log_file = Path(output_log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if log_file.exists():
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append(retraining_log_entry)

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return retraining_log_entry

if __name__ == "__main__":
    result = run_retraining_pipeline(force=True)
    print(f"Retraining Pipeline Result: {result['decision']} - {result['action_summary']}")
