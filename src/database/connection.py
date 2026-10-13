import os
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from dotenv import load_dotenv

load_dotenv()

def get_db_connection() -> Engine:
    """
    Creates and returns a SQLAlchemy engine for PostgreSQL based on env vars.
    """
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "hospital_db")
    user = os.getenv("POSTGRES_USER", "airflow")
    password = os.getenv("POSTGRES_PASSWORD", "airflow")

    # The Docker compose connection string
    conn_str = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    engine = create_engine(conn_str)
    return engine
