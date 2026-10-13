-- 1. Readmission Rate by Age Group
CREATE OR REPLACE VIEW analytics.readmission_by_age_group AS
SELECT 
    p.age_group,
    COUNT(a.encounter_surrogate_key) AS total_admissions,
    SUM(CASE WHEN r.is_readmitted THEN 1 ELSE 0 END) AS readmitted_count,
    ROUND(SUM(CASE WHEN r.is_readmitted THEN 1 ELSE 0 END) * 100.0 / COUNT(a.encounter_surrogate_key), 2) AS readmission_rate
FROM curated.admissions a
JOIN curated.patients p ON a.patient_surrogate_key = p.patient_surrogate_key
JOIN curated.readmissions r ON a.encounter_surrogate_key = r.encounter_surrogate_key
GROUP BY p.age_group;

-- 2. Readmission by Diagnosis (Primary Diagnosis - sequence 1)
CREATE OR REPLACE VIEW analytics.readmission_by_diagnosis AS
SELECT 
    d.diagnosis_code,
    COUNT(a.encounter_surrogate_key) AS total_admissions,
    SUM(CASE WHEN r.is_readmitted THEN 1 ELSE 0 END) AS readmitted_count,
    ROUND(SUM(CASE WHEN r.is_readmitted THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(a.encounter_surrogate_key), 0), 2) AS readmission_rate
FROM curated.admissions a
JOIN curated.diagnoses d ON a.encounter_surrogate_key = d.encounter_surrogate_key AND d.diagnosis_sequence = 1
JOIN curated.readmissions r ON a.encounter_surrogate_key = r.encounter_surrogate_key
GROUP BY d.diagnosis_code
HAVING COUNT(a.encounter_surrogate_key) > 50; -- Only significant diagnoses

-- 3. Length of Stay Analysis
CREATE OR REPLACE VIEW analytics.length_of_stay_analysis AS
SELECT 
    a.length_of_stay,
    COUNT(a.encounter_surrogate_key) AS admission_count,
    SUM(CASE WHEN r.is_readmitted THEN 1 ELSE 0 END) AS readmitted_count,
    ROUND(SUM(CASE WHEN r.is_readmitted THEN 1 ELSE 0 END) * 100.0 / COUNT(a.encounter_surrogate_key), 2) AS readmission_rate
FROM curated.admissions a
JOIN curated.readmissions r ON a.encounter_surrogate_key = r.encounter_surrogate_key
GROUP BY a.length_of_stay;

-- 4. High-Risk Cohort (Descriptive: Prior Admissions > 2 & High Meds)
CREATE OR REPLACE VIEW analytics.high_risk_cohort AS
SELECT 
    p.patient_surrogate_key,
    p.patient_nbr,
    p.age_group,
    COUNT(a.encounter_surrogate_key) AS total_encounters,
    MAX(ps.prior_admission_count) AS max_prior_admissions,
    AVG(ps.num_medications) AS avg_medications,
    ROUND(SUM(CASE WHEN r.is_readmitted THEN 1 ELSE 0 END) * 100.0 / COUNT(a.encounter_surrogate_key), 2) AS historical_readmission_rate
FROM curated.patients p
JOIN curated.admissions a ON p.patient_surrogate_key = a.patient_surrogate_key
JOIN curated.patient_summary ps ON a.encounter_surrogate_key = ps.encounter_surrogate_key
JOIN curated.readmissions r ON a.encounter_surrogate_key = r.encounter_surrogate_key
GROUP BY p.patient_surrogate_key, p.patient_nbr, p.age_group
HAVING MAX(ps.prior_admission_count) >= 2 AND AVG(ps.num_medications) > 15;

-- 5. Overall KPI View
CREATE OR REPLACE VIEW analytics.overall_kpis AS
SELECT
    COUNT(DISTINCT p.patient_surrogate_key) AS unique_patients,
    COUNT(a.encounter_surrogate_key) AS total_admissions,
    ROUND(AVG(a.length_of_stay), 2) AS avg_length_of_stay,
    ROUND(AVG(ps.number_diagnoses), 2) AS avg_diagnoses_per_encounter,
    ROUND(SUM(CASE WHEN r.is_readmitted THEN 1 ELSE 0 END) * 100.0 / COUNT(a.encounter_surrogate_key), 2) AS overall_readmission_rate
FROM curated.admissions a
JOIN curated.patients p ON a.patient_surrogate_key = p.patient_surrogate_key
JOIN curated.patient_summary ps ON a.encounter_surrogate_key = ps.encounter_surrogate_key
JOIN curated.readmissions r ON a.encounter_surrogate_key = r.encounter_surrogate_key;
