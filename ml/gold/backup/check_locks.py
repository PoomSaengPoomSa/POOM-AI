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
            # 1. SHOW PROCESSLIST
            cursor.execute("SHOW PROCESSLIST")
            rows = cursor.fetchall()
            print("=== MySQL Processlist ===")
            for r in rows:
                print(f"Id: {r['Id']} | User: {r['User']} | Host: {r['Host']} | db: {r['db']} | Command: {r['Command']} | Time: {r['Time']} | State: {r['State']} | Info: {r['Info']}")
                
            # 2. SHOW ENGINE INNODB STATUS
            cursor.execute("SHOW ENGINE INNODB STATUS")
            status = cursor.fetchone()
            print("\n=== InnoDB Status (Snippet) ===")
            # Status는 큰 텍스트이므로 락 섹션을 포함하는 부분 출력
            status_text = list(status.values())[2]
            if "TRANSACTIONS" in status_text:
                tx_part = status_text.split("TRANSACTIONS")[1][:1000]
                print(tx_part)
            else:
                print(status_text[:500])
                
    finally:
        connection.close()

if __name__ == '__main__':
    main()
