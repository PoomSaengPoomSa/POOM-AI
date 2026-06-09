import os
import pymysql
import pymysql.cursors
from contextlib import contextmanager
from dotenv import load_dotenv, find_dotenv

# 로컬 .env 또는 상위 폴더 탐색을 통한 통합 .env 로드
load_dotenv(find_dotenv())

# Strict Environment Variable Validation (No hardcoded fallback defaults in code)
DB_HOST = os.getenv("DB_HOST")
DB_PORT_STR = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

missing_vars = []
if not DB_HOST: missing_vars.append("DB_HOST")
if not DB_PORT_STR: missing_vars.append("DB_PORT")
if not DB_USER: missing_vars.append("DB_USER")
if not DB_PASSWORD: missing_vars.append("DB_PASSWORD")
if not DB_NAME: missing_vars.append("DB_NAME")

if missing_vars:
    raise ValueError(
        f"Database configuration error: The following required environment variables "
        f"are missing from the .env file: {', '.join(missing_vars)}. "
        f"Please verify your local .env file at {find_dotenv()}"
    )

DB_PORT = int(DB_PORT_STR)

@contextmanager
def get_db_connection():
    """
    Context manager that yields a database connection.
    Automatically handles commits and rollbacks.
    """
    connection = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor
    )
    try:
        yield connection
        connection.commit()
    except Exception as e:
        connection.rollback()
        raise e
    finally:
        connection.close()

@contextmanager
def get_db_cursor():
    """
    Context manager that yields a cursor from a database connection.
    """
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            yield cursor
