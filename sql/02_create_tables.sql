-- RAW TABLES
CREATE TABLE IF NOT EXISTS raw.raw_hospital_data (
    encounter_id VARCHAR(50),
    patient_nbr VARCHAR(50),
    race VARCHAR(100),
    gender VARCHAR(50),
    age VARCHAR(50),
    weight VARCHAR(50),
    admission_type_id VARCHAR(50),
    discharge_disposition_id VARCHAR(50),
    admission_source_id VARCHAR(50),
    time_in_hospital VARCHAR(50),
    payer_code VARCHAR(50),
    medical_specialty VARCHAR(100),
    num_lab_procedures VARCHAR(50),
    num_procedures VARCHAR(50),
    num_medications VARCHAR(50),
    number_outpatient VARCHAR(50),
    number_emergency VARCHAR(50),
    number_inpatient VARCHAR(50),
    diag_1 VARCHAR(100),
    diag_2 VARCHAR(100),
    diag_3 VARCHAR(100),
    number_diagnoses VARCHAR(50),
    max_glu_serum VARCHAR(50),
    A1Cresult VARCHAR(50),
    metformin VARCHAR(50),
    repaglinide VARCHAR(50),
    nateglinide VARCHAR(50),
    chlorpropamide VARCHAR(50),
    glimepiride VARCHAR(50),
    acetohexamide VARCHAR(50),
    glipizide VARCHAR(50),
    glyburide VARCHAR(50),
    tolbutamide VARCHAR(50),
    pioglitazone VARCHAR(50),
    rosiglitazone VARCHAR(50),
    acarbose VARCHAR(50),
    miglitol VARCHAR(50),
    troglitazone VARCHAR(50),
    tolazamide VARCHAR(50),
    examide VARCHAR(50),
    citoglipton VARCHAR(50),
    insulin VARCHAR(50),
    "glyburide-metformin" VARCHAR(50),
    "glipizide-metformin" VARCHAR(50),
    "glimepiride-pioglitazone" VARCHAR(50),
    "metformin-rosiglitazone" VARCHAR(50),
    "metformin-pioglitazone" VARCHAR(50),
    change VARCHAR(50),
    diabetesMed VARCHAR(50),
    readmitted VARCHAR(50),
    _ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- STAGING TABLES
CREATE TABLE IF NOT EXISTS staging.staging_hospital_data (
    encounter_id BIGINT PRIMARY KEY,
    patient_nbr BIGINT,
    race VARCHAR(100),
    gender VARCHAR(50),
    age_group VARCHAR(50),
    admission_type_id INTEGER,
    discharge_disposition_id INTEGER,
    admission_source_id INTEGER,
    length_of_stay INTEGER,
    num_lab_procedures INTEGER,
    num_procedures INTEGER,
    num_medications INTEGER,
    number_outpatient INTEGER,
    number_emergency INTEGER,
    number_inpatient INTEGER,
    diag_1 VARCHAR(100),
    diag_2 VARCHAR(100),
    diag_3 VARCHAR(100),
    number_diagnoses INTEGER,
    readmission_label VARCHAR(50)
);

-- CURATED TABLES
CREATE TABLE IF NOT EXISTS curated.patients (
    patient_surrogate_key SERIAL PRIMARY KEY,
    patient_nbr BIGINT UNIQUE,
    race VARCHAR(100),
    gender VARCHAR(50),
    age_group VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS curated.admissions (
    encounter_surrogate_key SERIAL PRIMARY KEY,
    encounter_id BIGINT UNIQUE,
    patient_surrogate_key INTEGER REFERENCES curated.patients(patient_surrogate_key),
    admission_type_id INTEGER,
    discharge_disposition_id INTEGER,
    admission_source_id INTEGER,
    length_of_stay INTEGER,
    readmission_label VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS curated.patient_summary (
    encounter_surrogate_key INTEGER PRIMARY KEY REFERENCES curated.admissions(encounter_surrogate_key),
    patient_surrogate_key INTEGER REFERENCES curated.patients(patient_surrogate_key),
    prior_admission_count INTEGER,
    num_lab_procedures INTEGER,
    num_procedures INTEGER,
    num_medications INTEGER,
    number_diagnoses INTEGER
);

CREATE TABLE IF NOT EXISTS curated.diagnoses (
    diagnosis_id SERIAL PRIMARY KEY,
    encounter_surrogate_key INTEGER REFERENCES curated.admissions(encounter_surrogate_key),
    diagnosis_code VARCHAR(100),
    diagnosis_sequence INTEGER -- 1 for diag_1, 2 for diag_2, etc.
);

CREATE TABLE IF NOT EXISTS curated.readmissions (
    encounter_surrogate_key INTEGER PRIMARY KEY REFERENCES curated.admissions(encounter_surrogate_key),
    patient_surrogate_key INTEGER REFERENCES curated.patients(patient_surrogate_key),
    readmission_status VARCHAR(50),
    is_readmitted BOOLEAN
);

-- AUDIT TABLES
CREATE TABLE IF NOT EXISTS audit.ingestion_log (
    ingestion_id SERIAL PRIMARY KEY,
    source_name VARCHAR(255),
    file_name VARCHAR(255),
    extraction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50),
    row_count INTEGER,
    column_count INTEGER,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS audit.validation_log (
    validation_id SERIAL PRIMARY KEY,
    run_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    layer VARCHAR(50),
    table_name VARCHAR(100),
    rule_id VARCHAR(50),
    passed_count INTEGER,
    failed_count INTEGER,
    status VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS audit.rejected_records (
    rejection_id SERIAL PRIMARY KEY,
    original_identifier VARCHAR(255),
    rejection_reason TEXT,
    rejected_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    validation_rule_violated VARCHAR(100),
    raw_data JSONB
);
