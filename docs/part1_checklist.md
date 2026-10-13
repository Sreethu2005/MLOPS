# Part 1 Completion Checklist

| Requirement | Implemented? | File/Location | How to Verify |
|---|---|---|---|
| Dataset acquired/documented | Yes | `docs/dataset_source.md` | Read the doc |
| Reproducible ingestion | Yes | `src/ingestion/ingest.py` | Run the ingest script |
| Raw copy | Yes | `data/raw/input/` | Check folder |
| Extraction metadata | Yes | `audit.ingestion_log` | Query the table |
| Error logging | Yes | `audit.validation_log` | Query the table |
| Repeatable ingestion | Yes | `src/ingestion/ingest.py` | Idempotent table replace |
| Airflow DAG | Yes | `dags/hospital_readmission_etl.py`| Open Airflow UI |
| Raw layer | Yes | `sql/01_create_schemas.sql` | Query DB |
| Staging layer | Yes | `sql/01_create_schemas.sql` | Query DB |
| Cleaned/curated layer | Yes | `sql/01_create_schemas.sql` | Query DB |
| Analytical layer | Yes | `sql/04_analytics.sql` | Query DB |
| Missing-value handling | Yes | `src/transformation/transform.py` | Check for NULLs instead of '?' |
| Standardization | Yes | `src/transformation/transform.py` | Check boolean is_readmitted |
| Duplicate removal | Yes | `src/transformation/transform.py` | Check code |
| Invalid encounter handling | Yes | `src/validation/validate.py` | Check rejected_records |
| Surrogate keys | Yes | `src/transformation/transform.py` | Check curated tables |
| Length of stay | Yes | `src/transformation/transform.py` | Renamed from time_in_hospital |
| Prior admissions | Yes | `src/transformation/transform.py` | Used number_inpatient |
| Diagnosis count | Yes | `src/transformation/transform.py` | Used number_diagnoses |
| Medication count | Yes | `src/transformation/transform.py` | Used num_medications |
| Admission table | Yes | `sql/02_create_tables.sql` | Query DB |
| Diagnosis table | Yes | `sql/02_create_tables.sql` | Query DB |
| Patient summary | Yes | `sql/02_create_tables.sql` | Query DB |
| Readmission table | Yes | `sql/02_create_tables.sql` | Query DB |
| Data-quality checks | Yes | `src/validation/validate.py` | Run validation script |
| Rejected records | Yes | `audit.rejected_records` | Query DB |
| PostgreSQL | Yes | `docker-compose.yml` | Run `docker ps` |
| Analytical data mart | Yes | `sql/04_analytics.sql` | View `analytics` schema |
| Five or more dashboard views | Yes | `dashboard/app.py` | Open Streamlit |
| Architecture diagram | Yes | `docs/architecture.md` | Read the doc |
| Data dictionary | Yes | `docs/data_dictionary.md` | Read the doc |
| Validation rules | Yes | `docs/validation_rules.md` | Read the doc |
| End-to-end execution | Yes | `README.md` | Follow instructions |
| Report | Yes | `reports/part1_report.md` | Read the report |
| Screenshots checklist | Yes | `screenshots/README.md` | View list |
| README | Yes | `README.md` | Read the doc |
