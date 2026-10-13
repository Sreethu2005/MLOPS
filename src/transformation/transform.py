import pandas as pd
from src.database.connection import get_db_connection
from src.database.load import load_dataframe_to_postgres
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

def transform_to_staging():
    """
    Reads from raw, cleans null representations ('?'), handles duplicates,
    casts types, and loads to staging.
    """
    logger.info("Starting transformation to staging layer")
    engine = get_db_connection()
    df = pd.read_sql("SELECT * FROM raw.raw_hospital_data", engine)
    
    # Drop ingestion metadata columns
    if '_ingestion_timestamp' in df.columns:
        df = df.drop(columns=['_ingestion_timestamp'])
        
    # Replace '?' with None
    df = df.replace('?', None)
    
    # Remove exact duplicates
    initial_count = len(df)
    df = df.drop_duplicates()
    logger.info(f"Removed {initial_count - len(df)} exact duplicate rows")
    
    # Remove duplicate encounter_ids (keep first)
    df = df.drop_duplicates(subset=['encounter_id'], keep='first')
    
    # Rename columns for clarity where needed
    df = df.rename(columns={'age': 'age_group', 'time_in_hospital': 'length_of_stay', 'readmitted': 'readmission_label'})
    
    # Cast to numeric types where applicable
    numeric_cols = ['encounter_id', 'patient_nbr', 'admission_type_id', 'discharge_disposition_id', 
                    'admission_source_id', 'length_of_stay', 'num_lab_procedures', 'num_procedures', 
                    'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Drop rows where encounter_id or patient_nbr is null
    df = df.dropna(subset=['encounter_id', 'patient_nbr'])
    
    # Load to staging
    # We only select columns that exist in staging
    staging_cols = ['encounter_id', 'patient_nbr', 'race', 'gender', 'age_group', 
                    'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 
                    'length_of_stay', 'num_lab_procedures', 'num_procedures', 'num_medications', 
                    'number_outpatient', 'number_emergency', 'number_inpatient', 
                    'diag_1', 'diag_2', 'diag_3', 'number_diagnoses', 'readmission_label']
    
    df_staging = df[staging_cols]
    
    load_dataframe_to_postgres(df_staging, "staging_hospital_data", "staging", if_exists="replace")
    logger.info(f"Loaded {len(df_staging)} rows to staging")

def clean_and_transform_curated():
    """
    Reads from staging, generates surrogate keys, standardizes values, 
    and loads to curated tables.
    """
    logger.info("Starting transformation to curated layer")
    engine = get_db_connection()
    df = pd.read_sql("SELECT * FROM staging.staging_hospital_data", engine)
    
    # 1. Standardize values
    # Gender: handle 'Unknown/Invalid' -> map to None or keep as is
    df['gender'] = df['gender'].replace('Unknown/Invalid', None)
    
    # Race: standard clean
    # Readmission:
    # <30, >30, NO
    # We want to create is_readmitted boolean
    df['is_readmitted'] = df['readmission_label'].apply(lambda x: True if x in ['<30', '>30'] else False)
    
    # 2. Patients Dimension
    # Get unique patients. Sort by encounter_id to get the latest demographics if multiple
    patients = df.sort_values('encounter_id').drop_duplicates('patient_nbr', keep='last')
    patients_dim = patients[['patient_nbr', 'race', 'gender', 'age_group']].copy()
    patients_dim['patient_surrogate_key'] = range(1, len(patients_dim) + 1)
    
    # 3. Admissions Fact
    # Merge patient surrogate key back to main df
    df = df.merge(patients_dim[['patient_nbr', 'patient_surrogate_key']], on='patient_nbr', how='left')
    
    admissions_fact = df[['encounter_id', 'patient_surrogate_key', 'admission_type_id', 
                          'discharge_disposition_id', 'admission_source_id', 'length_of_stay', 'readmission_label']].copy()
    admissions_fact['encounter_surrogate_key'] = range(1, len(admissions_fact) + 1)
    
    # Update df with encounter surrogate key for further processing
    df = df.merge(admissions_fact[['encounter_id', 'encounter_surrogate_key']], on='encounter_id', how='left')
    
    # 4. Patient Summary Fact
    # prior_admission_count can be approximated by number_inpatient
    patient_summary = df[['encounter_surrogate_key', 'patient_surrogate_key', 
                          'number_inpatient', 'num_lab_procedures', 'num_procedures', 
                          'num_medications', 'number_diagnoses']].copy()
    patient_summary = patient_summary.rename(columns={'number_inpatient': 'prior_admission_count'})
    
    # 5. Readmissions Fact
    readmissions = df[['encounter_surrogate_key', 'patient_surrogate_key', 'readmission_label', 'is_readmitted']].copy()
    readmissions = readmissions.rename(columns={'readmission_label': 'readmission_status'})
    
    # 6. Diagnoses Dimension (Melted)
    diag_1 = df[['encounter_surrogate_key', 'diag_1']].rename(columns={'diag_1': 'diagnosis_code'})
    diag_1['diagnosis_sequence'] = 1
    diag_2 = df[['encounter_surrogate_key', 'diag_2']].rename(columns={'diag_2': 'diagnosis_code'})
    diag_2['diagnosis_sequence'] = 2
    diag_3 = df[['encounter_surrogate_key', 'diag_3']].rename(columns={'diag_3': 'diagnosis_code'})
    diag_3['diagnosis_sequence'] = 3
    
    diagnoses_dim = pd.concat([diag_1, diag_2, diag_3]).dropna(subset=['diagnosis_code'])
    
    # Load to curated tables in correct order
    # Truncate tables first (cascade handled by replace/truncate)
    # Since we have foreign keys, replacing via pandas is tricky. We'll delete data and append.
    
    with engine.begin() as conn:
        from sqlalchemy import text
        conn.execute(text("TRUNCATE TABLE curated.readmissions, curated.diagnoses, curated.patient_summary, curated.admissions, curated.patients CASCADE"))
        
    load_dataframe_to_postgres(patients_dim, "patients", "curated", if_exists="append")
    load_dataframe_to_postgres(admissions_fact, "admissions", "curated", if_exists="append")
    load_dataframe_to_postgres(patient_summary, "patient_summary", "curated", if_exists="append")
    load_dataframe_to_postgres(readmissions, "readmissions", "curated", if_exists="append")
    load_dataframe_to_postgres(diagnoses_dim, "diagnoses", "curated", if_exists="append")
    
    logger.info("Curated transformation complete")

if __name__ == "__main__":
    transform_to_staging()
    clean_and_transform_curated()
