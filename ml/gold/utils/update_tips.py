import os
import pymysql
import pandas as pd
from dotenv import load_dotenv, find_dotenv

def main():
    load_dotenv(find_dotenv())
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    DB_NAME = os.getenv('DB_NAME')
    
    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
        print("[ERROR] Missing DB credentials in .env")
        return
        
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
            # 1. ml_gold_raw 테이블에 컬럼 추가 (존재 유무 체크 후)
            cursor.execute("DESCRIBE ml_gold_raw")
            gold_cols = [c['Field'] for c in cursor.fetchall()]
            
            if 'us_fed_rate' not in gold_cols:
                print("[DB] Adding 'us_fed_rate' column to ml_gold_raw...")
                cursor.execute("ALTER TABLE ml_gold_raw ADD COLUMN us_fed_rate DECIMAL(15, 6) DEFAULT NULL")
            if 'kr_base_rate' not in gold_cols:
                print("[DB] Adding 'kr_base_rate' column to ml_gold_raw...")
                cursor.execute("ALTER TABLE ml_gold_raw ADD COLUMN kr_base_rate DECIMAL(15, 6) DEFAULT NULL")
                
            # 2. ml_baserate_raw 에서 금리 시계열 로드
            print("[DB] Loading interest rates from ml_baserate_raw...")
            cursor.execute("SELECT loaded_date, us_fed_rate, kr_base_rate FROM ml_baserate_raw ORDER BY loaded_date ASC")
            base_rows = cursor.fetchall()
            
        df_base = pd.DataFrame(base_rows)
        # loaded_date를 YYYY-MM 문자열로 변환
        df_base['ym'] = pd.to_datetime(df_base['loaded_date']).dt.strftime('%Y-%m')
        
        # 금리 결측치 보간 (ffill & bfill)
        df_base['us_fed_rate'] = pd.to_numeric(df_base['us_fed_rate'], errors='coerce')
        df_base['kr_base_rate'] = pd.to_numeric(df_base['kr_base_rate'], errors='coerce')
        df_base['us_fed_rate'] = df_base['us_fed_rate'].ffill().bfill()
        df_base['kr_base_rate'] = df_base['kr_base_rate'].ffill().bfill()
        
        # 딕셔너리로 매핑 데이터 변환
        rate_map = {}
        for _, row in df_base.iterrows():
            rate_map[row['ym']] = {
                'us_fed_rate': float(row['us_fed_rate']) if pd.notnull(row['us_fed_rate']) else None,
                'kr_base_rate': float(row['kr_base_rate']) if pd.notnull(row['kr_base_rate']) else None
            }
            
        # 3. ml_gold_raw 의 모든 loaded_date와 primary key를 쿼리해서 일괄 업데이트
        with connection.cursor() as cursor:
            # gr_id와 loaded_date를 쿼리함
            # loaded_date의 타입은 datetime이거나 string일 수 있으므로 날짜 객체 또는 문자열에 맞춰 처리
            cursor.execute("SELECT gr_id, loaded_date FROM ml_gold_raw")
            gold_rows = cursor.fetchall()
            
            print(f"[DB] Mapping interest rates to {len(gold_rows)} daily gold records...")
            
            update_data = []
            for grow in gold_rows:
                gr_id = grow['gr_id']
                ldate = grow['loaded_date']
                ym = pd.to_datetime(ldate).strftime('%Y-%m')
                
                # 금리 가져오기 (만약 맵에 없으면 forward fill 방어)
                rates = rate_map.get(ym, None)
                if rates is None:
                    # 가장 가까운 연월을 찾아 매핑
                    nearest_ym = sorted(list(rate_map.keys()))[-1]
                    rates = rate_map[nearest_ym]
                    
                update_data.append((rates['us_fed_rate'], rates['kr_base_rate'], gr_id))
                
            # Bulk UPDATE 수행
            update_sql = "UPDATE ml_gold_raw SET us_fed_rate = %s, kr_base_rate = %s WHERE gr_id = %s"
            cursor.executemany(update_sql, update_data)
            connection.commit()
            print("[DB] Successfully migrated us_fed_rate and kr_base_rate to ml_gold_raw table!")
            
    finally:
        connection.close()

if __name__ == '__main__':
    main()
