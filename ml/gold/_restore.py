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
    cur.execute("""
        INSERT INTO trend_llm_report (report_id, type, model_name, language, content, status, data_source, created_at)
        VALUES ('test_76baa204', 'gold', 'gpt-4o', 'ko', '안녕하세요! 뭘 도와드릴까요?', 'done', 'FRED, ECOS', NULL)
    """)
    print('Restored rows:', cur.rowcount)
conn.commit()
conn.close()
print('복구 완료')
