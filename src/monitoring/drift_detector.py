"""
Data Drift Detection Engine: Population Stability Index (PSI) & Kolmogorov-Smirnov Tests.
Monitors feature distribution shifts between baseline training data and production serving traffic.

ACADEMIC DECISION-SUPPORT PROTOTYPE ONLY - Not for clinical diagnostic use.
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path for direct script execution
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.data.loader import load_config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

def calculate_numerical_psi(reference: np.ndarray, current: np.ndarray, num_bins: int = 10, epsilon: float = 1e-4) -> dict:
    """
    Calculates the Population Stability Index (PSI) for continuous numerical features.
    
    Binnings are defined from the reference distribution quantiles.
    PSI < 0.10: Stable (No shift)
    0.10 <= PSI < 0.25: Moderate shift (Warning / Monitor closely)
    PSI >= 0.25: Significant shift (Critical drift / Trigger retraining)
    """
    ref_clean = reference[~np.isnan(reference)]
    curr_clean = current[~np.isnan(current)]

    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return {"psi": 0.0, "status": "insufficient_data", "details": []}

    # Bin edges based on reference percentiles
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref_clean, percentiles)
    bin_edges = np.unique(bin_edges)

    # Handle edge case where distribution has very low variance
    if len(bin_edges) < 2:
        bin_edges = np.array([ref_clean.min() - 1e-3, ref_clean.max() + 1e-3])

    # Ensure outermost boundaries cover current data
    bin_edges[0] = min(bin_edges[0], curr_clean.min() - 1e-3)
    bin_edges[-1] = max(bin_edges[-1], curr_clean.max() + 1e-3)

    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    curr_counts, _ = np.histogram(curr_clean, bins=bin_edges)

    ref_props = ref_counts / len(ref_clean)
    curr_props = curr_counts / len(curr_clean)

    # Add epsilon to prevent log(0) and division by zero
    ref_props = np.where(ref_props == 0, epsilon, ref_props)
    curr_props = np.where(curr_props == 0, epsilon, curr_props)

    # Re-normalize
    ref_props = ref_props / np.sum(ref_props)
    curr_props = curr_props / np.sum(curr_props)

    psi_contributions = (curr_props - ref_props) * np.log(curr_props / ref_props)
    total_psi = float(np.sum(psi_contributions))

    if total_psi < 0.10:
        status = "stable"
    elif total_psi < 0.25:
        status = "warning"
    else:
        status = "critical"

    return {
        "psi": round(total_psi, 4),
        "status": status,
        "num_bins": len(ref_counts),
        "bin_edges": [round(float(e), 3) for e in bin_edges.tolist()]
    }

def calculate_categorical_psi(reference: pd.Series, current: pd.Series, epsilon: float = 1e-4) -> dict:
    """
    Calculates Population Stability Index (PSI) for discrete categorical features.
    """
    ref_s = reference.dropna().astype(str)
    curr_s = current.dropna().astype(str)

    if len(ref_s) == 0 or len(curr_s) == 0:
        return {"psi": 0.0, "status": "insufficient_data", "categories": {}}

    all_categories = sorted(list(set(ref_s.unique()).union(set(curr_s.unique()))))

    ref_counts = ref_s.value_counts()
    curr_counts = curr_s.value_counts()

    ref_total = len(ref_s)
    curr_total = len(curr_s)

    psi_total = 0.0
    category_breakdown = {}

    for cat in all_categories:
        ref_p = ref_counts.get(cat, 0) / ref_total
        curr_p = curr_counts.get(cat, 0) / curr_total

        ref_p_adj = max(ref_p, epsilon)
        curr_p_adj = max(curr_p, epsilon)

        contrib = (curr_p_adj - ref_p_adj) * np.log(curr_p_adj / ref_p_adj)
        psi_total += contrib

        category_breakdown[cat] = {
            "ref_prop": round(float(ref_p), 4),
            "curr_prop": round(float(curr_p), 4),
            "psi_contrib": round(float(contrib), 4)
        }

    total_psi = float(psi_total)
    if total_psi < 0.10:
        status = "stable"
    elif total_psi < 0.25:
        status = "warning"
    else:
        status = "critical"

    return {
        "psi": round(total_psi, 4),
        "status": status,
        "category_count": len(all_categories),
        "category_breakdown": category_breakdown
    }

def calculate_ks_statistic(reference: np.ndarray, current: np.ndarray, p_val_threshold: float = 0.05) -> dict:
    """
    Performs Kolmogorov-Smirnov 2-sample test to test if two continuous distributions differ.
    """
    ref_clean = reference[~np.isnan(reference)]
    curr_clean = current[~np.isnan(current)]

    if len(ref_clean) < 5 or len(curr_clean) < 5:
        return {"ks_statistic": 0.0, "p_value": 1.0, "drift_detected": False}

    res = ks_2samp(ref_clean, curr_clean)
    drift = bool(res.pvalue < p_val_threshold)

    return {
        "ks_statistic": round(float(res.statistic), 4),
        "p_value": round(float(res.pvalue), 6),
        "drift_detected": drift
    }

def load_serving_logs_as_dataframe(log_path: str = "logs/serving_requests.log") -> pd.DataFrame:
    """
    Parses JSON Lines from the serving log into a structured DataFrame.
    """
    file_p = Path(log_path)
    if not file_p.exists():
        logger.warning(f"Serving log file {log_path} not found.")
        return pd.DataFrame()

    records = []
    with open(file_p, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                try:
                    entry = json.loads(line_str)
                    raw_feat = entry.get("request", {})
                    # Add prediction meta
                    raw_feat["predicted_class"] = entry.get("predicted_class")
                    raw_feat["readmission_probability"] = entry.get("readmission_probability")
                    records.append(raw_feat)
                except Exception:
                    pass

    return pd.DataFrame(records)

def detect_dataset_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, config: dict = None) -> dict:
    """
    Comprehensive drift detection comparing baseline reference to current evaluation batch.
    Evaluates both continuous and categorical features via PSI and KS-tests.
    """
    if config is None:
        config = load_config("configs/config.yaml")

    mon_cfg = config.get("monitoring", {})
    tracked_features = mon_cfg.get("drift_tracked_features", [
        "num_medications", "number_inpatient", "time_in_hospital",
        "num_lab_procedures", "number_diagnoses", "age", "gender", "race"
    ])
    p_threshold = mon_cfg.get("drift_thresholds", {}).get("ks_p_value_threshold", 0.05)

    drift_results = {}
    warning_count = 0
    critical_count = 0

    for feat in tracked_features:
        if feat not in reference_df.columns or feat not in current_df.columns:
            logger.debug(f"Feature '{feat}' not present in both datasets. Skipping.")
            continue

        ref_series = reference_df[feat]
        curr_series = current_df[feat]

        if pd.api.types.is_numeric_dtype(ref_series) and ref_series.nunique() > 10:
            # Continuous numerical feature
            ref_arr = ref_series.to_numpy(dtype=float)
            curr_arr = curr_series.to_numpy(dtype=float)

            psi_info = calculate_numerical_psi(ref_arr, curr_arr)
            ks_info = calculate_ks_statistic(ref_arr, curr_arr, p_val_threshold=p_threshold)

            feat_res = {
                "feature_type": "numerical",
                "psi": psi_info["psi"],
                "status": psi_info["status"],
                "ks_statistic": ks_info["ks_statistic"],
                "ks_p_value": ks_info["p_value"],
                "distribution_shift_detected": ks_info["drift_detected"]
            }
        else:
            # Discrete or categorical feature
            psi_info = calculate_categorical_psi(ref_series, curr_series)
            feat_res = {
                "feature_type": "categorical",
                "psi": psi_info["psi"],
                "status": psi_info["status"],
                "category_count": psi_info.get("category_count", 0)
            }

        if feat_res["status"] == "critical":
            critical_count += 1
        elif feat_res["status"] == "warning":
            warning_count += 1

        drift_results[feat] = feat_res

    # Overall system health assessment
    if critical_count > 0:
        overall_status = "CRITICAL_DRIFT"
        recommendation = "CRITICAL DATA DRIFT DETECTED. Immediate feature investigation and pipeline retraining required."
    elif warning_count > 0:
        overall_status = "WARNING"
        recommendation = "MODERATE DRIFT OBSERVED. Increase monitoring frequency; schedule review before retraining."
    else:
        overall_status = "STABLE"
        recommendation = "Distributions are statistically stable within calibrated thresholds."

    return {
        "timestamp": pd.Timestamp.now().isoformat(),
        "reference_sample_size": len(reference_df),
        "current_sample_size": len(current_df),
        "total_features_evaluated": len(drift_results),
        "features_warning": warning_count,
        "features_critical": critical_count,
        "overall_status": overall_status,
        "action_recommendation": recommendation,
        "feature_drift_details": drift_results
    }

def run_drift_analysis(config_path: str = "configs/config.yaml", current_df: pd.DataFrame = None, output_path: str = "reports/drift_report.json") -> dict:
    """
    Main entry point for running data drift audit against the reference training split.
    """
    config = load_config(config_path)
    ref_path = config.get("monitoring", {}).get("reference_data_path", "data/processed/train.parquet")

    logger.info(f"Loading reference training baseline from: {ref_path}")
    if not Path(ref_path).exists():
        raise FileNotFoundError(f"Reference training dataset not found at {ref_path}. Run Phase 1 first.")

    reference_df = pd.read_parquet(ref_path)

    if current_df is None:
        log_path = config.get("monitoring", {}).get("serving_log_path", "logs/serving_requests.log")
        logger.info(f"Attempting to load current batch from serving logs: {log_path}")
        current_df = load_serving_logs_as_dataframe(log_path)

        if len(current_df) < 10:
            logger.info("Serving logs have fewer than 10 entries; evaluating held-out test split as current baseline.")
            current_df = pd.read_parquet(config["data"]["test_data_path"])

    report = detect_dataset_drift(reference_df, current_df, config=config)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Saved drift detection report to {out_file} (Overall status: {report['overall_status']})")
    return report

if __name__ == "__main__":
    report = run_drift_analysis()
    print(f"Drift Analysis Status: {report['overall_status']} | Evaluated: {report['total_features_evaluated']} features")
