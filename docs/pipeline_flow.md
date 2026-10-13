# Pipeline Flow Document

This document explains the end-to-end data pipeline.

1. **Data Acquisition**: The Python script `src/ingestion/ingest.py` downloads the UCI ZIP file, extracts it, and locates `diabetic_data.csv`. A manual fallback is supported.
2. **Raw Ingestion**: The raw CSV is loaded entirely into a Pandas dataframe and written to the `raw.raw_hospital_data` PostgreSQL table. All columns are treated as text to prevent data loss.
3. **Metadata Logging**: Metadata (source, row count, columns, status) is logged to `audit.ingestion_log`.
4. **Raw Validation**: `src/validation/validate.py` checks for row counts, missing `encounter_id`, and duplicate `encounter_id`. Failures are logged to `audit.rejected_records`.
5. **Staging**: `src/transformation/transform.py` creates a staging dataframe. It maps `?` to NULL, drops duplicates, casts numerical fields to their correct types, and loads into `staging.staging_hospital_data`.
6. **Cleaning & Transformation**: The curated transformation normalizes the data into a star-schema inspired model. Categorical variables like gender are cleaned. A boolean `is_readmitted` is generated.
7. **Surrogate Key Generation**: Sequential surrogate keys (`patient_surrogate_key`, `encounter_surrogate_key`) are generated for the `patients` and `admissions` tables to detach from raw identifiers.
8. **PostgreSQL Loading**: The cleaned, dimensionally modeled data is loaded into the `curated` schema.
9. **Data-Quality Checks**: A secondary validation on the staging table ensures that critical columns like `age_group` are not null.
10. **Analytical Data Mart**: SQL views in the `analytics` schema aggregate the curated data for dashboard consumption (e.g., readmission by age group, high risk cohorts).
11. **Streamlit Dashboard**: A Python application that connects to PostgreSQL and visualizes the analytical views using Plotly.
