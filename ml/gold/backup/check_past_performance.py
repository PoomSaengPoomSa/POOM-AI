import pymysql, os
from dotenv import load_dotenv, find_dotenv

def main():
    load_dotenv(find_dotenv())
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    DB_NAME = os.getenv('DB_NAME')
    
    connection = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=int(DB_PORT),
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM gold_performance ORDER BY evaluated_at DESC LIMIT 10")
            rows = cursor.fetchall()
            print("=== Past Gold Performance History ===")
            for r in rows:
                print(f"RunID: {r['run_id']} | Acc: {r['accuracy']:.4f} | F1: {r['f1_score']:.4f} | EvalAt: {r['evaluated_at']}")
    finally:
        connection.close()

if __name__ == '__main__':
    main()
