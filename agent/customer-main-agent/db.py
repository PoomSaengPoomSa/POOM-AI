import os
import pymysql
import pymysql.cursors
from contextlib import contextmanager
from dotenv import load_dotenv

# 루트 경로의 .env 로드 설정
# customer-main-agent 폴더 기준 상위 3단계 위에 위치한 프로젝트 루트의 .env 파일
current_dir = os.path.dirname(os.path.abspath(__file__))
root_env_path = os.path.abspath(os.path.join(current_dir, "..", "..", "..", ".env"))

if os.path.exists(root_env_path):
    load_dotenv(dotenv_path=root_env_path)
else:
    # Fallback to standard root or local search
    load_dotenv()

# MySQL 환경변수 추출 및 검증
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
        f"Database configuration error: Required environment variables are missing: {', '.join(missing_vars)}. "
        f"Verified path: {root_env_path}"
    )

DB_PORT = int(DB_PORT_STR)

# LangSmith 추적 활성화 환경변수 매핑 처리
if os.getenv("LANGSMITH_TRACING") == "true":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
if os.getenv("LANGSMITH_ENDPOINT"):
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
if os.getenv("LANGSMITH_PROJECT"):
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT").strip('"\'')

@contextmanager
def get_db_connection():
    """
    Context manager that yields a pymysql database connection.
    Automatically commits on success, rollbacks on error.
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
    Context manager that yields a dictionary cursor.
    """
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            yield cursor
