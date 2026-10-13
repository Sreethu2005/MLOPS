"""
End-to-End MLOps Pipeline Orchestrator.
Cross-platform entry point to execute data prep, training, explainability,
registry, monitoring, and retraining stages.

ACADEMIC DECISION-SUPPORT PROTOTYPE ONLY - Not for clinical diagnostic use.
"""
import argparse
import sys
import time
from pathlib import Path

from src.utils.logging_utils import get_logger

logger = get_logger("PipelineOrchestrator")

def run_data_step():
    logger.info(">>> STEP 1: Running Data Pipeline & Validation...")
    from src.data.loader import run_data_pipeline
    from src.data.validator import run_validation
    train_df, val_df, test_df = run_data_pipeline()
    val_res = run_validation()
    logger.info(f"Data pipeline complete. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}. DQ Status: {val_res['overall_status']}")

def run_features_step():
    logger.info(">>> STEP 2: Running Feature Engineering Pipeline...")
    from src.features.pipeline import run_feature_pipeline
    preprocessor = run_feature_pipeline()
    logger.info("Feature engineering complete. Preprocessor fitted on training split.")

def run_train_step():
    logger.info(">>> STEP 3: Running Model Training & Calibration...")
    from src.training.train import run_model_training
    train_res = run_model_training()
    logger.info("Training complete. Winner model saved to models/best_model.joblib.")

def run_explain_step():
    logger.info(">>> STEP 4: Running TreeSHAP Explainability Pipeline...")
    from src.explainability.explainer import run_explainability_pipeline
    shap_results, sample_explanation = run_explainability_pipeline()
    logger.info(f"Explainability complete. Global plot: {shap_results['plot_path']}")

def run_registry_step():
    logger.info(">>> STEP 5: Registering Model in MLflow Model Registry...")
    from src.serving.registry import register_best_model
    reg_res = register_best_model()
    logger.info(f"Registry complete. Registered {reg_res['model_name']} version {reg_res['version']} with @production alias.")

def run_monitor_step():
    logger.info(">>> STEP 6: Running Data Drift & Fairness Auditing...")
    from src.monitoring.drift_detector import run_drift_analysis
    from src.monitoring.fairness_audit import run_fairness_audit
    from src.monitoring.simulator import run_simulation
    d_res = run_drift_analysis()
    f_res = run_fairness_audit()
    s_res = run_simulation()
    logger.info(f"Monitoring complete. Drift: {d_res['overall_status']} | Test Recall: {f_res['overall_test_cohort']['recall_sensitivity']:.4f}")

def run_retrain_step():
    logger.info(">>> STEP 7: Running Champion-Challenger Retraining Pipeline...")
    from src.training.retrain import run_retraining_pipeline
    ret_res = run_retraining_pipeline(force=True)
    logger.info(f"Retraining complete. Decision: {ret_res['decision']}")

def run_test_step():
    logger.info(">>> STEP 8: Executing Full Pytest Suite...")
    import pytest
    exit_code = pytest.main(["-v", "tests/"])
    if exit_code != 0:
        logger.error(f"Pytest exited with status code {exit_code}")
        sys.exit(exit_code)
    logger.info("Full test suite passed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Hospital Readmission MLOps Pipeline Orchestrator")
    parser.add_argument(
        "--step",
        choices=["all", "data", "features", "train", "explain", "registry", "monitor", "retrain", "test"],
        default="all",
        help="Specify which pipeline step to execute (default: all)"
    )
    args = parser.parse_args()

    start_time = time.time()
    logger.info(f"Starting MLOps pipeline run (step: {args.step})...")

    step_map = {
        "data": [run_data_step],
        "features": [run_features_step],
        "train": [run_train_step],
        "explain": [run_explain_step],
        "registry": [run_registry_step],
        "monitor": [run_monitor_step],
        "retrain": [run_retrain_step],
        "test": [run_test_step],
        "all": [
            run_data_step,
            run_features_step,
            run_train_step,
            run_explain_step,
            run_registry_step,
            run_monitor_step,
            run_test_step
        ]
    }

    steps_to_run = step_map[args.step]
    for step_fn in steps_to_run:
        step_fn()

    elapsed = time.time() - start_time
    logger.info(f"Pipeline execution finished successfully in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
