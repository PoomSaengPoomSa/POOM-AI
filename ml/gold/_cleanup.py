from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import pymysql, os

conn = pymysql.connect(
    host=os.getenv('DB_HOST'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    database=os.getenv('DB_NAME'),
    port=int(os.getenv('DB_PORT')),
    charset='utf8mb4'
)
with conn.cursor() as cur:
    cur.execute("DELETE FROM trend_llm_report WHERE report_id = 'test_76baa204'")
    print('Deleted rows:', cur.rowcount)
conn.commit()
conn.close()
print('Done')
