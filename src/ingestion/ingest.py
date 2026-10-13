import os
import zipfile
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from src.utils.logging_utils import get_logger
from src.database.load import load_dataframe_to_postgres

load_dotenv()
logger = get_logger(__name__)

def download_and_extract_data(url: str, extract_to: str) -> str:
    """
    Downloads and extracts the dataset ZIP file.
    Returns the path to the extracted CSV file.
    """
    zip_path = os.path.join(extract_to, "dataset.zip")
    
    # Check if a dataset_diabetes/diabetic_data.csv already exists manually
    potential_csv = os.path.join(extract_to, "dataset_diabetes", "diabetic_data.csv")
    if os.path.exists(potential_csv):
        logger.info(f"Dataset already exists at {potential_csv}. Skipping download.")
        return potential_csv
        
    try:
        logger.info(f"Downloading dataset from {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        logger.info(f"Extracting {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            
        csv_path = os.path.join(extract_to, "dataset_diabetes", "diabetic_data.csv")
        return csv_path
    except Exception as e:
        logger.error(f"Error downloading/extracting data: {e}")
        # Check fallback
        fallback_csv = os.path.join(extract_to, "diabetic_data.csv")
        if os.path.exists(fallback_csv):
             logger.info(f"Found fallback CSV at {fallback_csv}. Using it.")
             return fallback_csv
        raise e

def log_ingestion_metadata(source_name: str, file_name: str, status: str, row_count: int, col_count: int, error_msg: str = None):
    """
    Logs ingestion metadata to the audit schema.
    """
    metadata = pd.DataFrame([{
        "source_name": source_name,
        "file_name": file_name,
        "extraction_timestamp": datetime.now(),
        "status": status,
        "row_count": row_count,
        "column_count": col_count,
        "error_message": error_msg
    }])
    
    load_dataframe_to_postgres(metadata, "ingestion_log", "audit", if_exists="append")

def ingest_raw_data():
    """
    Main ingestion function.
    Downloads data, reads it into pandas, and loads it into the raw database layer.
    """
    logger.info("Starting ingestion process")
    url = os.getenv("DATA_SOURCE_URL", "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip")
    
    # Path mappings for Docker vs Local
    # We use absolute path based on the script location to handle both
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_dir = os.path.join(base_dir, "data", "raw", "input")
    
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir, exist_ok=True)
        
    try:
        csv_path = download_and_extract_data(url, raw_dir)
        logger.info(f"Reading CSV from {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # All cols as string for raw layer
        df = df.astype(str)
        
        row_count = len(df)
        col_count = len(df.columns)
        
        logger.info(f"Loaded {row_count} rows and {col_count} columns")
        
        # Load to PostgreSQL RAW schema
        # Drop table if exists and recreate for idempotency of full ingestion
        load_dataframe_to_postgres(df, "raw_hospital_data", "raw", if_exists="replace")
        
        log_ingestion_metadata(url, os.path.basename(csv_path), "SUCCESS", row_count, col_count)
        logger.info("Ingestion complete")
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        log_ingestion_metadata(url, "unknown", "FAILED", 0, 0, str(e))
        raise e

if __name__ == "__main__":
    ingest_raw_data()
