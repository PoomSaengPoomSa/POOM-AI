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
            cursor.execute("DESCRIBE ml_baserate_raw")
            cols = cursor.fetchall()
            print("=== ml_baserate_raw columns ===")
            for c in cols:
                print(f"  {c['Field']} ({c['Type']})")
                
            cursor.execute("SELECT loaded_date, us_fed_rate, kr_base_rate FROM ml_baserate_raw ORDER BY loaded_date DESC LIMIT 5")
            rows = cursor.fetchall()
            print("\n=== ml_baserate_raw sample ===")
            for r in rows:
                print(r)
    finally:
        connection.close()

if __name__ == '__main__':
    main()
