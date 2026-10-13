# Data Dictionary

## Schema: Curated

### Table: patients
| Column Name | Data Type | Description | Nullable | Source Column | Transformation |
|---|---|---|---|---|---|
| patient_surrogate_key | INT | Unique surrogate key for patient | No | N/A | Generated (SERIAL) |
| patient_nbr | BIGINT | Original patient identifier | Yes | patient_nbr | None |
| race | VARCHAR | Race of the patient | Yes | race | Cleaned '?' |
| gender | VARCHAR | Gender of the patient | Yes | gender | Cleaned 'Unknown/Invalid' |
| age_group | VARCHAR | Age group | Yes | age | Renamed |

### Table: admissions
| Column Name | Data Type | Description | Nullable | Source Column | Transformation |
|---|---|---|---|---|---|
| encounter_surrogate_key | INT | Unique surrogate key for encounter | No | N/A | Generated (SERIAL) |
| encounter_id | BIGINT | Original encounter identifier | Yes | encounter_id | None |
| patient_surrogate_key | INT | FK to patients | Yes | N/A | Mapped |
| admission_type_id | INT | ID for admission type | Yes | admission_type_id | Cast to INT |
| length_of_stay | INT | Days in hospital | Yes | time_in_hospital | Cast to INT |

### Table: patient_summary
| Column Name | Data Type | Description | Nullable | Source Column | Transformation |
|---|---|---|---|---|---|
| prior_admission_count | INT | Number of prior inpatient visits | Yes | number_inpatient | Renamed |
| num_medications | INT | Count of medications | Yes | num_medications | Cast to INT |

### Table: readmissions
| Column Name | Data Type | Description | Nullable | Source Column | Transformation |
|---|---|---|---|---|---|
| readmission_status | VARCHAR | Original status (<30, >30, NO) | Yes | readmitted | Renamed |
| is_readmitted | BOOLEAN | True if <30 or >30 | Yes | readmitted | Calculated |

## Schema: Audit

### Table: rejected_records
| Column Name | Data Type | Description | Nullable | Source Column | Transformation |
|---|---|---|---|---|---|
| original_identifier | VARCHAR | Encounter ID or index | Yes | N/A | String conversion |
| rejection_reason | TEXT | Reason for failure | Yes | N/A | Generated |
| raw_data | JSONB | Original row data | Yes | N/A | JSON conversion |
