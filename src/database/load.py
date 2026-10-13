import pandas as pd
from sqlalchemy.engine import Engine
from sqlalchemy import text
from src.database.connection import get_db_connection
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

def load_dataframe_to_postgres(df: pd.DataFrame, table_name: str, schema: str, if_exists: str = "append", index: bool = False):
    """
    Loads a Pandas DataFrame into a PostgreSQL table.
    """
    engine = get_db_connection()
    try:
        logger.info(f"Loading {len(df)} rows into {schema}.{table_name}")
        df.to_sql(table_name, engine, schema=schema, if_exists=if_exists, index=index)
        logger.info(f"Successfully loaded {table_name}")
    except Exception as e:
        logger.error(f"Failed to load {table_name}: {e}")
        raise e

def execute_sql_file(file_path: str):
    """
    Executes a SQL script file.
    """
    engine = get_db_connection()
    try:
        with open(file_path, 'r') as f:
            sql = f.read()
            
        with engine.begin() as conn:
            # Simple split by semicolon might be naive if there are semicolons inside strings
            # But sufficient for simple setup scripts
            statements = [s.strip() for s in sql.split(';') if s.strip()]
            for statement in statements:
                conn.execute(text(statement))
        logger.info(f"Successfully executed SQL script {file_path}")
    except Exception as e:
        logger.error(f"Failed to execute SQL script {file_path}: {e}")
        raise e
