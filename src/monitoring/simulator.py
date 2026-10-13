"""
Drift & Anomaly Simulation Script.
Simulates normal and shifted patient batches to verify that PSI / KS monitors
accurately distinguish stable conditions from critical drift scenarios.

ACADEMIC DECISION-SUPPORT PROTOTYPE ONLY.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.data.loader import load_config
from src.monitoring.drift_detector import detect_dataset_drift
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

def generate_normal_batch(test_df: pd.DataFrame, n_samples: int = 500, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a normal in-distribution batch by random sampling from test set.
    Expected PSI: < 0.10 (STABLE).
    """
    sample_df = test_df.sample(n=min(n_samples, len(test_df)), random_state=random_state).copy()
    return sample_df

def generate_drifted_batch(test_df: pd.DataFrame, n_samples: int = 500, random_state: int = 42) -> pd.DataFrame:
    """
    Simulates acute clinical distribution drift:
    - Significant surge in prior inpatient admissions (+2 to +4 admissions)
    - Length of stay shift (+3 to +5 days)
    - Polypharmacy increase (+8 to +15 medications)
    - Severe shift towards older, higher-comorbidity cohort
    Expected PSI: >= 0.25 (CRITICAL_DRIFT).
    """
    np.random.seed(random_state)
    sample_df = test_df.sample(n=min(n_samples, len(test_df)), random_state=random_state).copy()

    # Shift clinical variables
    sample_df["time_in_hospital"] = np.clip(sample_df["time_in_hospital"] + np.random.randint(3, 6, size=len(sample_df)), 1, 14)
    sample_df["number_inpatient"] = sample_df["number_inpatient"] + np.random.randint(2, 5, size=len(sample_df))
    sample_df["num_medications"] = np.clip(sample_df["num_medications"] + np.random.randint(8, 16, size=len(sample_df)), 1, 100)
    sample_df["number_diagnoses"] = np.clip(sample_df["number_diagnoses"] + np.random.randint(2, 5, size=len(sample_df)), 1, 16)

    # Shift demographics towards older cohorts
    older_ages = ["[70-80)", "[80-90)", "[90-100)"]
    sample_df["age"] = np.random.choice(older_ages, size=len(sample_df), p=[0.5, 0.4, 0.1])

    return sample_df

def run_simulation(config_path: str = "configs/config.yaml") -> dict:
    """
    Executes dual simulation:
    1. Normal batch -> verifies stability.
    2. Drifted batch -> verifies PSI alert trigger and retraining flag.
    """
    config = load_config(config_path)
    train_path = config["data"]["train_data_path"]
    test_path = config["data"]["test_data_path"]

    ref_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    logger.info("Executing Scenario 1: Normal In-Distribution Traffic...")
    normal_batch = generate_normal_batch(test_df, n_samples=500)
    normal_report = detect_dataset_drift(ref_df, normal_batch, config=config)

    logger.info(f"Normal Batch Overall Status: {normal_report['overall_status']} (Critical features: {normal_report['features_critical']})")

    logger.info("Executing Scenario 2: Simulated High-Acuity Distribution Drift...")
    drifted_batch = generate_drifted_batch(test_df, n_samples=500)
    drifted_report = detect_dataset_drift(ref_df, drifted_batch, config=config)

    logger.info(f"Drifted Batch Overall Status: {drifted_report['overall_status']} (Critical features: {drifted_report['features_critical']})")

    simulation_summary = {
        "scenario_normal": {
            "sample_size": len(normal_batch),
            "status": normal_report["overall_status"],
            "features_critical": normal_report["features_critical"],
            "features_warning": normal_report["features_warning"],
            "action": normal_report["action_recommendation"]
        },
        "scenario_drifted": {
            "sample_size": len(drifted_batch),
            "status": drifted_report["overall_status"],
            "features_critical": drifted_report["features_critical"],
            "features_warning": drifted_report["features_warning"],
            "action": drifted_report["action_recommendation"],
            "drifted_feature_psi": {
                k: v["psi"] for k, v in drifted_report["feature_drift_details"].items()
                if v.get("status") in ["warning", "critical"]
            }
        }
    }

    out_file = Path("reports/drift_simulation_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(simulation_summary, f, indent=2)

    logger.info(f"Saved drift simulation report to: {out_file}")
    return simulation_summary

if __name__ == "__main__":
    results = run_simulation()
    print("Drift simulation completed.")
    print("Normal status:", results["scenario_normal"]["status"])
    print("Drifted status:", results["scenario_drifted"]["status"])
