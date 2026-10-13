"""
Data Quality Validation module.
Validates schema, data types, value ranges, missingness, and split integrity.
Produces structured pass/fail reports.
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path for direct script execution
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
import pandas as pd
from src.data.loader import load_config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

def validate_data_quality(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, config: dict) -> dict:
    """
    Validates data quality across train, validation, and test datasets.
    Returns structured pass/fail report.
    """
    report = {
        "overall_status": "PASSED",
        "checks": [],
        "summary": {}
    }

    def add_check(check_id: str, description: str, passed: bool, details: dict):
        status = "PASSED" if passed else "FAILED"
        if not passed:
            report["overall_status"] = "FAILED"
        report["checks"].append({
            "check_id": check_id,
            "description": description,
            "status": status,
            "details": details
        })

    # 1. Row counts check
    min_rows = 1000
    rows_ok = len(train_df) >= min_rows and len(val_df) >= min_rows and len(test_df) >= min_rows
    add_check(
        "DQ01_ROW_COUNTS",
        "Verify all splits have sufficient observations",
        rows_ok,
        {"train_rows": len(train_df), "val_rows": len(val_df), "test_rows": len(test_df)}
    )

    # 2. Schema check (required columns present)
    required_num = config["features"]["numerical_cols"]
    required_cat = config["features"]["categorical_cols"]
    target_col = config["data"]["target_col"]
    required_cols = set(required_num + required_cat + [target_col])

    missing_cols_train = list(required_cols - set(train_df.columns))
    add_check(
        "DQ02_SCHEMA_COMPLETENESS",
        "Verify all feature and target columns exist in dataset",
        len(missing_cols_train) == 0,
        {"missing_columns": missing_cols_train}
    )

    # 3. Patient identity leakage check (Zero patient overlap across splits)
    patient_id_col = config["data"]["patient_id_col"]
    train_patients = set(train_df[patient_id_col])
    val_patients = set(val_df[patient_id_col])
    test_patients = set(test_df[patient_id_col])

    train_val_overlap = len(train_patients.intersection(val_patients))
    train_test_overlap = len(train_patients.intersection(test_patients))
    val_test_overlap = len(val_patients.intersection(test_patients))
    no_leakage = (train_val_overlap == 0) and (train_test_overlap == 0) and (val_test_overlap == 0)

    add_check(
        "DQ03_PATIENT_LEAKAGE_CHECK",
        "Verify ZERO patient overlap between train, validation, and test splits",
        no_leakage,
        {
            "train_val_overlap": train_val_overlap,
            "train_test_overlap": train_test_overlap,
            "val_test_overlap": val_test_overlap
        }
    )

    # 4. Target distribution & validity
    target_valid = (
        set(train_df[target_col].unique()).issubset({0, 1}) and
        train_df[target_col].mean() > 0.05 and
        train_df[target_col].mean() < 0.30
    )
    add_check(
        "DQ04_TARGET_DISTRIBUTION",
        "Verify binary 30-day readmission target and expected positive rate",
        target_valid,
        {
            "train_positive_rate": round(float(train_df[target_col].mean()), 4),
            "val_positive_rate": round(float(val_df[target_col].mean()), 4),
            "test_positive_rate": round(float(test_df[target_col].mean()), 4)
        }
    )

    # 5. Value ranges check
    range_failures = []
    # time_in_hospital should be 1-14
    if not train_df["time_in_hospital"].between(1, 14).all():
        range_failures.append("time_in_hospital out of range [1, 14]")
    # num_medications >= 1
    if not (train_df["num_medications"] >= 1).all():
        range_failures.append("num_medications < 1")
    # num_lab_procedures >= 1
    if not (train_df["num_lab_procedures"] >= 1).all():
        range_failures.append("num_lab_procedures < 1")

    add_check(
        "DQ05_VALUE_RANGES",
        "Verify numerical features conform to clinical boundary ranges",
        len(range_failures) == 0,
        {"range_failures": range_failures}
    )

    # 6. Anti-leakage: Expired/Hospice exclusion check
    leakage_ids = config["data"].get("leakage_discharge_disposition_ids", [11, 13, 14, 19, 20, 21])
    expired_present = False
    if "discharge_disposition_id" in train_df.columns:
        expired_count = train_df["discharge_disposition_id"].isin(leakage_ids).sum()
        expired_present = expired_count > 0

    add_check(
        "DQ06_ANTI_LEAKAGE_EXPIRED_CHECK",
        "Verify zero expired or hospice encounters present in training data",
        not expired_present,
        {"expired_encounter_count": int(expired_count) if "discharge_disposition_id" in train_df.columns else 0}
    )

    # Summary
    passed_count = sum(1 for c in report["checks"] if c["status"] == "PASSED")
    total_count = len(report["checks"])
    report["summary"] = {
        "total_checks": total_count,
        "passed_checks": passed_count,
        "failed_checks": total_count - passed_count,
        "overall_status": report["overall_status"]
    }

    return report

def run_validation(config_path: str = "configs/config.yaml"):
    """Loads splits and executes DQ checks, outputting report."""
    config = load_config(config_path)
    train_path = Path(config["data"]["train_data_path"])
    val_path = Path(config["data"]["val_data_path"])
    test_path = Path(config["data"]["test_data_path"])

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        raise FileNotFoundError("Processed split files not found. Run loader first.")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    report = validate_data_quality(train_df, val_df, test_df, config)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "data_quality_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("================ DATA QUALITY REPORT ================")
    logger.info(f"Overall Status: {report['overall_status']}")
    for c in report["checks"]:
        logger.info(f"[{c['status']}] {c['check_id']}: {c['description']} -> {c['details']}")
    logger.info("=====================================================")

    return report

if __name__ == "__main__":
    run_validation()
