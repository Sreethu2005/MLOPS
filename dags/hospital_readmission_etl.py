import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.utils.dates import days_ago

from src.ingestion.ingest import ingest_raw_data
from src.validation.validate import validate_raw_data, validate_staging_data
from src.transformation.transform import transform_to_staging, clean_and_transform_curated

default_args = {
    'owner': 'data_engineer',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    'hospital_readmission_etl',
    default_args=default_args,
    description='ETL pipeline for Hospital Readmission Analytics',
    schedule_interval='@daily',
    start_date=days_ago(1),
    catchup=False,
    tags=['hospital', 'readmission', 'part1'],
    template_searchpath=[os.path.abspath(os.path.dirname(os.path.dirname(__file__)))],
) as dag:

    # 1. Setup Database Schemas and Tables
    setup_schemas = PostgresOperator(
        task_id='setup_schemas',
        postgres_conn_id='postgres_default', # Will use AIRFLOW_CONN_POSTGRES_DEFAULT if set, or we can just use python script
        sql='sql/01_create_schemas.sql',
    )
    
    setup_tables = PostgresOperator(
        task_id='setup_tables',
        postgres_conn_id='postgres_default',
        sql='sql/02_create_tables.sql',
    )
    
    setup_indexes = PostgresOperator(
        task_id='setup_indexes',
        postgres_conn_id='postgres_default',
        sql='sql/03_create_indexes.sql',
    )

    # 2. Ingestion
    ingest_raw = PythonOperator(
        task_id='ingest_raw_data',
        python_callable=ingest_raw_data,
    )

    # 3. Validate Raw
    validate_raw = PythonOperator(
        task_id='validate_raw_data',
        python_callable=validate_raw_data,
    )

    # 4. Transform to Staging
    transform_staging = PythonOperator(
        task_id='transform_to_staging',
        python_callable=transform_to_staging,
    )

    # 5. Validate Staging
    validate_staging = PythonOperator(
        task_id='validate_staging_data',
        python_callable=validate_staging_data,
    )

    # 6. Clean and Transform to Curated
    transform_curated = PythonOperator(
        task_id='clean_and_transform_curated',
        python_callable=clean_and_transform_curated,
    )

    # 7. Create Analytics Views
    create_analytics_views = PostgresOperator(
        task_id='create_analytics_views',
        postgres_conn_id='postgres_default',
        sql='sql/04_analytics.sql',
    )

    # Define Dependencies
    setup_schemas >> setup_tables >> setup_indexes >> ingest_raw
    ingest_raw >> validate_raw >> transform_staging >> validate_staging >> transform_curated >> create_analytics_views
