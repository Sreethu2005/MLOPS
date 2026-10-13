# Screenshots Checklist

Please capture the following screenshots to submit as evidence of your project execution. Place them in this folder.

1. **Dataset Source**
   - **What it proves**: You accessed the correct UCI dataset.
   - **How to capture**: Take a screenshot of the UCI dataset web page or the downloaded `dataset.zip` in your local folder.

2. **Project Folder Structure**
   - **What it proves**: The project follows the required structure.
   - **How to capture**: Open VSCode or your file explorer and take a screenshot of the folder tree.

3. **Airflow DAG Graph**
   - **What it proves**: The pipeline has the correct dependencies.
   - **How to capture**: Open Airflow UI (`http://localhost:8080`), click the `hospital_readmission_etl` DAG, and take a screenshot of the "Graph" view.

4. **Successful DAG Execution**
   - **What it proves**: The pipeline runs end-to-end without failing.
   - **How to capture**: Take a screenshot of the "Tree" or "Grid" view in Airflow showing all tasks marked green (Success).

5. **Raw Data**
   - **What it proves**: The data was ingested successfully.
   - **How to capture**: Run `SELECT * FROM raw.raw_hospital_data LIMIT 10;` in DBeaver/pgAdmin and screenshot the result.

6. **Data Validation Output**
   - **What it proves**: Validation rules executed.
   - **How to capture**: Run `SELECT * FROM audit.validation_log;` and screenshot the result.

7. **Rejected Records / Error Log**
   - **What it proves**: Failed records are caught and isolated.
   - **How to capture**: Run `SELECT * FROM audit.rejected_records LIMIT 5;` and screenshot the result.

8. **PostgreSQL Curated Tables**
   - **What it proves**: Star-schema transformation succeeded.
   - **How to capture**: Run a query like `SELECT * FROM curated.patient_summary LIMIT 10;` showing surrogate keys.

9. **Analytical Views**
   - **What it proves**: Data mart was built correctly.
   - **How to capture**: Run `SELECT * FROM analytics.readmission_by_age_group;` and screenshot.

10. **Streamlit Dashboard Overview**
    - **What it proves**: The dashboard works and reads the database.
    - **How to capture**: Open `http://localhost:8501` and screenshot the top half (KPIs and main charts).

11. **Terminal showing tests passing**
    - **What it proves**: Unit tests run and pass.
    - **How to capture**: Run `pytest tests/` in the terminal and screenshot the green "passed" output.
