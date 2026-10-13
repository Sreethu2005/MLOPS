-- Indexes for Staging
CREATE INDEX IF NOT EXISTS idx_staging_patient_nbr ON staging.staging_hospital_data(patient_nbr);
CREATE INDEX IF NOT EXISTS idx_staging_encounter_id ON staging.staging_hospital_data(encounter_id);

-- Indexes for Curated
CREATE INDEX IF NOT EXISTS idx_patients_patient_nbr ON curated.patients(patient_nbr);
CREATE INDEX IF NOT EXISTS idx_patients_age_group ON curated.patients(age_group);

CREATE INDEX IF NOT EXISTS idx_admissions_patient_sk ON curated.admissions(patient_surrogate_key);
CREATE INDEX IF NOT EXISTS idx_admissions_encounter_id ON curated.admissions(encounter_id);

CREATE INDEX IF NOT EXISTS idx_diagnoses_encounter_sk ON curated.diagnoses(encounter_surrogate_key);
CREATE INDEX IF NOT EXISTS idx_diagnoses_code ON curated.diagnoses(diagnosis_code);

CREATE INDEX IF NOT EXISTS idx_readmissions_patient_sk ON curated.readmissions(patient_surrogate_key);
