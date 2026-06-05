import pymysql, os

def main():
    DB_USER = os.getenv('DB_USER', 'root')
    DB_PASSWORD = os.getenv('DB_PASSWORD', 'root') # 기본 fallback
    DB_HOST = os.getenv('DB_HOST', '127.0.0.1')
    DB_PORT = os.getenv('DB_PORT', '3306')
    DB_NAME = os.getenv('DB_NAME', 'poom')
    
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
    
    DB_USER = os.getenv('DB_USER', DB_USER)
    DB_PASSWORD = os.getenv('DB_PASSWORD', DB_PASSWORD)
    DB_HOST = os.getenv('DB_HOST', DB_HOST)
    DB_PORT = os.getenv('DB_PORT', DB_PORT)
    DB_NAME = os.getenv('DB_NAME', DB_NAME)
    
    print(f"Connecting to MySQL: {DB_HOST}:{DB_PORT} as {DB_USER}")
    
    try:
        connection = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=int(DB_PORT),
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=3 # 3초 타임아웃
        )
        
        try:
            with connection.cursor() as cursor:
                # Sleep 상태이고 Time이 60초 이상인 세션 또는 락 대기 프로세스 조회
                cursor.execute("SHOW PROCESSLIST")
                processes = cursor.fetchall()
                
                my_thread_id = connection.thread_id()
                print(f"My Thread ID: {my_thread_id}")
                
                kill_count = 0
                for p in processes:
                    pid = p['Id']
                    command = p['Command']
                    time_sec = p['Time']
                    state = p['State']
                    info = p['Info']
                    
                    # 내 세션은 패스
                    if pid == my_thread_id:
                        continue
                        
                    # 오래 떠있거나 락에 물린 외부 세션 강제 종료
                    # 특히 uvicorn이나 python에서 물고 있는 세션들 정리
                    if command == 'Sleep' and time_sec > 10:
                        print(f"Killing sleepy Process {pid} (Time: {time_sec}s)...")
                        try:
                            cursor.execute(f"KILL {pid}")
                            kill_count += 1
                        except Exception as ex:
                            print(f"Failed to kill {pid}: {ex}")
                    elif info and 'ml_' in info:
                        print(f"Killing query Process {pid} (Info: {info})...")
                        try:
                            cursor.execute(f"KILL {pid}")
                            kill_count += 1
                        except Exception as ex:
                            print(f"Failed to kill {pid}: {ex}")
                            
                print(f"Killed {kill_count} hanging process(es).")
        finally:
            connection.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
