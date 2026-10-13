# Project Report: Hospital Readmission Analytics (Part 1)

## 1. Title Page
**Project**: Hospital Readmission Analytics and Prediction (Part 1)
**Focus**: Data Engineering, Pipeline Orchestration, and Descriptive Analytics

## 2. Abstract
This report details the execution of Part 1 for the Hospital Readmission Analytics project. The goal was to build a robust data engineering pipeline using Python, PostgreSQL, and Apache Airflow to acquire, clean, validate, and analyze a decade of clinical care data regarding diabetic patients. The outcome is a curated star-schema data mart visualized through a Streamlit dashboard.

## 3. Introduction
Hospital readmissions are a critical metric for evaluating the quality of care and managing healthcare costs. This project leverages historical data to build an automated ETL pipeline that transforms raw clinical data into actionable insights, setting the stage for future predictive modeling.

## 4. Problem Statement
Raw clinical data is often messy, unstructured, and filled with missing values or invalid entries. The challenge is to automatically ingest this data, clean it reliably, handle missing representations securely, and structure it so that analysts can easily query descriptive metrics without encountering data anomalies.

## 5. Objectives
1. Acquire the UCI Diabetes 130-US Hospitals dataset.
2. Develop a reproducible ingestion and validation pipeline.
3. Design and populate a PostgreSQL data warehouse with raw, staging, curated, and analytics layers.
4. Orchestrate the workflow using Apache Airflow.
5. Provide an interactive dashboard for descriptive analytics.

## 6. Dataset Description
The selected dataset is the **UCI Diabetes 130-US Hospitals (1999–2008)** dataset. It contains over 100,000 instances and 50 attributes, representing encounters of diabetic patients. Features include demographics (age, gender, race), admission/discharge details, length of stay, and 24 specific medication metrics.

## 7. Data Architecture
The architecture comprises:
- **Local Landing**: Raw data storage.
- **Processing Engine**: Pandas for extraction and transformation.
- **Orchestrator**: Apache Airflow managing the DAG execution.
- **Storage**: PostgreSQL with schemas for `raw`, `staging`, `curated` (star-schema), `audit`, and `analytics`.
- **Presentation**: Streamlit Dashboard.

## 8. Data Ingestion
Ingestion is handled by a Python script that downloads the dataset zip file directly from UCI. It extracts the CSV and loads it into the `raw` schema with all columns cast as text to prevent data loss. A manual fallback is built in case of network restrictions.

## 9. ETL Pipeline
The pipeline is orchestrated by Airflow via the `hospital_readmission_etl` DAG:
1. `setup_schemas` & `setup_tables`
2. `ingest_raw_data`
3. `validate_raw_data`
4. `transform_to_staging`
5. `validate_staging_data`
6. `clean_and_transform_curated`
7. `create_analytics_views`

## 10. Data Cleaning
Data cleaning involves:
- Dropping exact duplicates.
- Deduplicating on `encounter_id`.
- Mapping the string `'?'` to standard SQL `NULL`.
- Cleaning the `gender` column (handling `Unknown/Invalid`).
- Type casting numerical strings to integers.

## 11. Data Validation
Data quality is enforced through specific rules:
- **DQ001**: Row count checks.
- **DQ002**: Null `encounter_id` rejection.
- **DQ003**: Duplicate `encounter_id` rejection.
- **DQ004**: Null `age_group` rejection.
Violations are logged in `audit.rejected_records` with the original JSON payload.

## 12. PostgreSQL Database Design
The curated schema is normalized into a star-schema:
- **Dimension Tables**: `patients`, `diagnoses`
- **Fact Tables**: `admissions`, `patient_summary`, `readmissions`
Surrogate keys (`patient_surrogate_key`, `encounter_surrogate_key`) are generated to decouple from the source identifiers and provide anonymity.

## 13. Analytical Data Mart
Views are created in the `analytics` schema to support the dashboard efficiently. Examples include `readmission_by_age_group`, `length_of_stay_analysis`, and `high_risk_cohort`.

## 14. Dashboard Design
The Streamlit dashboard contains:
- KPI summary metrics.
- Readmission distributions by demographics and length of stay.
- A descriptive view of a high-risk cohort (patients with >= 2 prior admissions and > 15 average medications).

## 15. Results and Observations
The pipeline successfully ingested the ~100k records. Preliminary observations show that certain diagnoses and longer lengths of stay correlate loosely with higher historical readmission rates. The high-risk cohort view successfully filters down the patient population to the most historically active patients.

## 16. Privacy and Ethical Considerations
All patient data is de-identified. We explicitly state that the dashboard provides descriptive analytics and should not be used for direct medical diagnosis or treatment decisions. Surrogate keys ensure that even if the raw identifiers were compromised, the curated data remains disconnected.

## 17. Limitations
- The dataset is relatively old (1999-2008).
- Dates are not available; length of stay is provided in discrete days.
- Many fields originally contained '?' which reduces the effective sample size for some specific analytical queries.

## 18. Part 1 Demonstration
The end-to-end execution can be demonstrated by spinning up the Docker Compose environment, triggering the Airflow DAG, querying the resulting PostgreSQL tables, and viewing the Streamlit dashboard.

## 19. Conclusion
Part 1 successfully established the foundational data engineering infrastructure. The pipeline is robust, idempotent, and heavily audited. The data is now clean and structured, ready for analytical querying.

## 20. Future Work – Part 2 MLOps
In Part 2, the curated data will be used to train predictive models (e.g., Random Forest, XGBoost) to classify readmission risk. This will involve MLflow for tracking, FastAPI for serving, and drift monitoring.

## 21. References
- UCI Machine Learning Repository: Diabetes 130-US hospitals for years 1999-2008 Data Set.
