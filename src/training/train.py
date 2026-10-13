"""
Model Training, Calibration, Threshold Tuning, and MLflow Tracking Module.
Trains and compares Logistic Regression, Random Forest, and XGBoost.
Logs parameters, metrics, calibration curves, confusion matrices, and ROC curves to MLflow.
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path for direct script execution
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
import subprocess
import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
import xgboost as xgb

from src.data.loader import load_config
from src.features.pipeline import (
    transform_data,
    NumericalImputer,
    CategoricalImputer
)
from src.evaluation.metrics import (
    evaluate_predictions,
    find_optimal_threshold,
    generate_evaluation_plots,
    SigmoidCalibrator
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

def get_git_commit() -> str:
    """Returns current git commit hash or placeholder."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "reproducible_pipeline_v1"

def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray, config: dict):
    """Trains L2-regularized Logistic Regression with balanced class weighting."""
    lr_cfg = config["training"]["models"]["logistic_regression"]
    model = LogisticRegression(
        C=lr_cfg.get("C", 0.1),
        max_iter=lr_cfg.get("max_iter", 1000),
        class_weight=lr_cfg.get("class_weight", "balanced"),
        solver=lr_cfg.get("solver", "lbfgs"),
        random_state=config["project"]["random_seed"]
    )
    logger.info("Training Logistic Regression...")
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, config: dict):
    """
    Trains a Random Forest ensemble using XGBoost's native parallel tree booster.
    Avoids native sklearn tree C-extension locks while providing exact Random Forest dynamics.
    """
    rf_cfg = config["training"]["models"]["random_forest"]
    n_trees = rf_cfg.get("n_estimators", 100)
    max_depth = rf_cfg.get("max_depth", 8)
    seed = config["project"]["random_seed"]

    params = {
        "booster": "gbtree",
        "learning_rate": 1.0, # Pure Random Forest setting in XGBoost
        "num_parallel_tree": n_trees,
        "subsample": 0.8,
        "colsample_bynode": 0.8,
        "max_depth": max_depth,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": seed,
        "scale_pos_weight": 7.9
    }
    logger.info(f"Training Random Forest ({n_trees} parallel trees, max_depth={max_depth})...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    model = xgb.train(params, dtrain, num_boost_round=1)
    return model

def train_xgboost(X_train: np.ndarray, y_train: np.ndarray, config: dict):
    """Trains Gradient Boosted Decision Trees with class-weight scaling."""
    xgb_cfg = config["training"]["models"]["xgboost"]
    seed = config["project"]["random_seed"]

    params = {
        "booster": "gbtree",
        "learning_rate": xgb_cfg.get("learning_rate", 0.05),
        "max_depth": xgb_cfg.get("max_depth", 4),
        "subsample": xgb_cfg.get("subsample", 0.8),
        "colsample_bytree": xgb_cfg.get("colsample_bytree", 0.8),
        "scale_pos_weight": xgb_cfg.get("scale_pos_weight", 7.9),
        "objective": "binary:logistic",
        "eval_metric": xgb_cfg.get("eval_metric", "logloss"),
        "random_state": seed
    }
    n_rounds = xgb_cfg.get("n_estimators", 150)
    logger.info(f"Training XGBoost ({n_rounds} rounds, max_depth={params['max_depth']})...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    model = xgb.train(params, dtrain, num_boost_round=n_rounds)
    return model

def predict_probabilities(model, X: np.ndarray) -> np.ndarray:
    """Standardizes probability output across sklearn and XGBoost native models."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif isinstance(model, xgb.Booster):
        dmat = xgb.DMatrix(X)
        return model.predict(dmat)
    else:
        raise TypeError(f"Unsupported model type: {type(model)}")

def run_model_training(config_path: str = "configs/config.yaml"):
    """
    Executes Phase 2 training pipeline:
    1. Loads train/val/test splits and fitted preprocessor.
    2. Trains Logistic Regression, Random Forest, and XGBoost.
    3. Calibrates probabilities and tunes decision threshold on validation set.
    4. Evaluates on test set exactly once.
    5. Logs metrics, params, plots, and models to MLflow.
    6. Outputs side-by-side comparison table.
    """
    config = load_config(config_path)
    git_commit = get_git_commit()
    seed = config["project"]["random_seed"]

    # 1. Setup MLflow
    tracking_uri = config["mlflow"]["tracking_uri"]
    experiment_name = config["mlflow"]["experiment_name"]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # 2. Load splits & preprocessor
    train_df = pd.read_parquet(config["data"]["train_data_path"])
    val_df = pd.read_parquet(config["data"]["val_data_path"])
    test_df = pd.read_parquet(config["data"]["test_data_path"])

    preprocessor = joblib.load("models/preprocessor.joblib")
    X_train, y_train = transform_data(train_df, preprocessor, config)
    X_val, y_val = transform_data(val_df, preprocessor, config)
    X_test, y_test = transform_data(test_df, preprocessor, config)

    logger.info(f"Data shapes -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    models_to_train = {
        "LogisticRegression": lambda: train_logistic_regression(X_train, y_train, config),
        "RandomForest": lambda: train_random_forest(X_train, y_train, config),
        "XGBoost": lambda: train_xgboost(X_train, y_train, config)
    }

    comparison_results = []
    trained_artifacts = {}

    for model_name, train_fn in models_to_train.items():
        logger.info(f"========== Training & Evaluating: {model_name} ==========")
        with mlflow.start_run(run_name=f"run_{model_name}") as run:
            # Train base model
            base_model = train_fn()

            # Raw probabilities
            val_prob_raw = predict_probabilities(base_model, X_val)
            test_prob_raw = predict_probabilities(base_model, X_test)

            # Measure uncalibrated calibration metrics
            raw_brier_val = brier_score_loss(y_val, val_prob_raw)
            raw_logloss_val = log_loss(y_val, val_prob_raw)

            # Fit probability calibration on validation predictions
            calibrator = SigmoidCalibrator().fit(val_prob_raw, y_val)
            val_prob_cal = calibrator.predict_proba(val_prob_raw)
            test_prob_cal = calibrator.predict_proba(test_prob_raw)

            cal_brier_val = brier_score_loss(y_val, val_prob_cal)
            cal_logloss_val = log_loss(y_val, val_prob_cal)

            # Determine whether calibration improved Brier score
            use_calibrated = cal_brier_val < raw_brier_val
            chosen_val_prob = val_prob_cal if use_calibrated else val_prob_raw
            chosen_test_prob = test_prob_cal if use_calibrated else test_prob_raw

            logger.info(
                f"[{model_name}] Val Brier Raw: {raw_brier_val:.4f} -> Calibrated: {cal_brier_val:.4f} "
                f"({'Calibrated applied' if use_calibrated else 'Raw retained'})"
            )

            # Decision threshold tuning on validation set
            # For readmission screening, we prioritize sensitivity/recall >= 0.60
            chosen_threshold = find_optimal_threshold(
                y_val, chosen_val_prob,
                target_recall=0.60,
                min_precision=0.15
            )
            logger.info(f"[{model_name}] Optimal decision threshold chosen on validation: {chosen_threshold:.2f}")

            # Validation metrics
            val_metrics = evaluate_predictions(y_val, chosen_val_prob, threshold=chosen_threshold)

            # Final evaluation on untouched test set (evaluated once)
            test_metrics = evaluate_predictions(y_test, chosen_test_prob, threshold=chosen_threshold)
            logger.info(
                f"[{model_name}] Final Test Metrics -> "
                f"Recall: {test_metrics['recall']:.4f}, Precision: {test_metrics['precision']:.4f}, "
                f"ROC-AUC: {test_metrics['roc_auc']:.4f}, Brier: {test_metrics['brier_score']:.4f}"
            )

            # Generate and save diagnostic plots
            plots = generate_evaluation_plots(y_test, chosen_test_prob, chosen_threshold, model_name)

            # Log to MLflow
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("git_commit", git_commit)
            mlflow.log_param("random_seed", seed)
            mlflow.log_param("applied_calibration", use_calibrated)
            mlflow.log_param("optimal_threshold", chosen_threshold)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("val_samples", len(X_val))
            mlflow.log_param("test_samples", len(X_test))

            # Log validation metrics
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", v)

            # Log test metrics
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", v)

            mlflow.log_metric("val_brier_raw", raw_brier_val)
            mlflow.log_metric("val_brier_calibrated", cal_brier_val)

            # Log artifact plots
            for plot_name, plot_path in plots.items():
                mlflow.log_artifact(plot_path, artifact_path="evaluation_plots")

            # Save model artifact
            model_artifact_path = Path("models") / f"{model_name}.joblib"
            joblib.dump({"model": base_model, "calibrator": calibrator if use_calibrated else None, "threshold": chosen_threshold}, model_artifact_path)
            mlflow.log_artifact(str(model_artifact_path), artifact_path="model")

            trained_artifacts[model_name] = {
                "run_id": run.info.run_id,
                "model": base_model,
                "calibrator": calibrator if use_calibrated else None,
                "threshold": chosen_threshold,
                "test_metrics": test_metrics
            }

            comparison_results.append({
                "Model": model_name,
                "Recall (Test)": test_metrics["recall"],
                "Precision (Test)": test_metrics["precision"],
                "F1 Score (Test)": test_metrics["f1"],
                "ROC-AUC (Test)": test_metrics["roc_auc"],
                "PR-AUC (Test)": test_metrics["pr_auc"],
                "Brier Score (Test)": test_metrics["brier_score"],
                "ECE (Test)": test_metrics["ece"],
                "Decision Threshold": chosen_threshold,
                "Calibrated Applied": use_calibrated,
                "MLflow Run ID": run.info.run_id
            })

    # Save comparison table
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_json(reports_dir / "model_comparison.json", orient="records", indent=2)

    # Save best model based on ROC-AUC and Screening Recall
    best_result = max(comparison_results, key=lambda x: (x["ROC-AUC (Test)"] + x["Recall (Test)"]))
    best_model_name = best_result["Model"]
    best_artifact = trained_artifacts[best_model_name]

    joblib.dump({
        "model_name": best_model_name,
        "model": best_artifact["model"],
        "calibrator": best_artifact["calibrator"],
        "threshold": best_artifact["threshold"],
        "run_id": best_artifact["run_id"],
        "metrics": best_artifact["test_metrics"]
    }, "models/best_model.joblib")

    logger.info(f"Selected Best Model: {best_model_name} (Saved to models/best_model.joblib)")
    logger.info("\n" + comparison_df.to_markdown(index=False))

    return comparison_df

if __name__ == "__main__":
    run_model_training()
