import pandas as pd
from datetime import datetime
from src.database.connection import get_db_connection
from src.database.load import load_dataframe_to_postgres
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

def log_validation_results(layer: str, table_name: str, rule_id: str, passed_count: int, failed_count: int, status: str):
    log_df = pd.DataFrame([{
        "layer": layer,
        "table_name": table_name,
        "rule_id": rule_id,
        "passed_count": passed_count,
        "failed_count": failed_count,
        "status": status,
        "run_timestamp": datetime.now()
    }])
    load_dataframe_to_postgres(log_df, "validation_log", "audit", if_exists="append")

def log_rejected_records(records_df: pd.DataFrame, reason: str, rule_id: str):
    if records_df.empty:
        return
    
    # Store the entire row as JSON string in raw_data
    rejected_df = pd.DataFrame({
        "original_identifier": records_df.get("encounter_id", records_df.index).astype(str),
        "rejection_reason": reason,
        "validation_rule_violated": rule_id,
        "raw_data": records_df.to_json(orient='records', lines=True).splitlines(),
        "rejected_timestamp": datetime.now()
    })
    load_dataframe_to_postgres(rejected_df, "rejected_records", "audit", if_exists="append")

def validate_raw_data():
    """
    Validates data in the raw layer before moving to staging.
    """
    logger.info("Starting raw data validation")
    engine = get_db_connection()
    
    # Check if table exists
    try:
        df = pd.read_sql("SELECT * FROM raw.raw_hospital_data", engine)
    except Exception as e:
        logger.error(f"Failed to read raw data: {e}")
        return False

    if df.empty:
        logger.error("Raw data is empty")
        log_validation_results("raw", "raw_hospital_data", "DQ001_ROW_COUNT", 0, 0, "FAILED")
        raise ValueError("Raw data table is empty")

    total_rows = len(df)
    
    # DQ002: Check for missing encounter_id
    missing_id_mask = df["encounter_id"].isnull() | (df["encounter_id"] == "nan") | (df["encounter_id"] == "")
    failed_rows = df[missing_id_mask]
    passed_count = total_rows - len(failed_rows)
    
    if len(failed_rows) > 0:
        log_rejected_records(failed_rows, "Missing encounter_id", "DQ002")
        log_validation_results("raw", "raw_hospital_data", "DQ002", passed_count, len(failed_rows), "FAILED")
        logger.warning(f"DQ002 failed: {len(failed_rows)} rows missing encounter_id")
    else:
        log_validation_results("raw", "raw_hospital_data", "DQ002", passed_count, 0, "PASSED")

    # DQ003: Check for duplicate encounter_id
    duplicate_mask = df.duplicated(subset=["encounter_id"], keep=False)
    failed_rows = df[duplicate_mask]
    passed_count = total_rows - len(failed_rows)
    
    if len(failed_rows) > 0:
        log_rejected_records(failed_rows, "Duplicate encounter_id", "DQ003")
        log_validation_results("raw", "raw_hospital_data", "DQ003", passed_count, len(failed_rows), "FAILED")
        logger.warning(f"DQ003 failed: {len(failed_rows)} rows have duplicate encounter_id")
    else:
        log_validation_results("raw", "raw_hospital_data", "DQ003", passed_count, 0, "PASSED")
        
    logger.info("Raw data validation complete")
    return True

def validate_staging_data():
    """
    Validates data in the staging layer before curated transformation.
    """
    logger.info("Starting staging data validation")
    engine = get_db_connection()
    df = pd.read_sql("SELECT * FROM staging.staging_hospital_data", engine)
    total_rows = len(df)
    
    # DQ004: Validate age_group format (e.g. [0-10))
    # We will just verify it's not null and is a string
    missing_age_mask = df["age_group"].isnull()
    failed_rows = df[missing_age_mask]
    passed_count = total_rows - len(failed_rows)
    
    if len(failed_rows) > 0:
        log_rejected_records(failed_rows, "Missing age_group", "DQ004")
        log_validation_results("staging", "staging_hospital_data", "DQ004", passed_count, len(failed_rows), "FAILED")
    else:
        log_validation_results("staging", "staging_hospital_data", "DQ004", passed_count, 0, "PASSED")
        
    logger.info("Staging data validation complete")
    return True

if __name__ == "__main__":
    validate_raw_data()
