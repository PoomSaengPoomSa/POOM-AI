import os
import pymysql
import pandas as pd
from dotenv import load_dotenv

def collect_all():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # back 폴더의 .env를 직접 경유하여 동기화
    back_env_path = os.path.join(os.path.dirname(base_dir), 'back', '.env')
    load_dotenv(dotenv_path=back_env_path)

    # Load DB credentials
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    DB_NAME = os.getenv('DB_NAME')

    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
        print("[Error] Missing DB configuration in .env file.")
        return

    DB_PORT = int(DB_PORT)

    print("=" * 55)
    print("Database (MySQL) Real Estate Data Collection")
    print("=" * 55)
    print(f"  DB Connection Info: {DB_HOST}:{DB_PORT}/{DB_NAME} (User: {DB_USER})")

    try:
        connection = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        print("  DB Connection successful!")
        
        with connection.cursor() as cursor:
            # Query all records from ml_realestate_raw sorted by loaded_date
            sql = "SELECT * FROM ml_realestate_raw ORDER BY loaded_date ASC"
            print("  Querying ml_realestate_raw table...")
            cursor.execute(sql)
            rows = cursor.fetchall()
            
        connection.close()
    except Exception as e:
        print(f"  [Error] DB Connection & Query failed: {e}")
        return

    if not rows:
        print("  [Error] No data found. Terminating.")
        return

    print(f"  Successfully collected {len(rows)} rows.")

    # Convert to DataFrame
    df = pd.DataFrame(rows)

    # Convert loaded_date (datetime) -> date_ym (string with YYYYMM format)
    df['date_ym'] = pd.to_datetime(df['loaded_date']).dt.strftime('%Y%m')
    
    # Drop unwanted DB sequence/date columns
    if 'rr_id' in df.columns:
        df = df.drop(columns=['rr_id'])
    if 'loaded_date' in df.columns:
        df = df.drop(columns=['loaded_date'])

    # Reorder columns to place date_ym first
    cols = ['date_ym'] + [col for col in df.columns if col != 'date_ym']
    df = df[cols]

    # Convert values to numeric
    numeric_cols = [col for col in df.columns if col != 'date_ym']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Setup directories and save
    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    save_path = os.path.join(data_dir, 'raw_data.csv')

    # Save to CSV
    df.to_csv(save_path, index=False, encoding='utf-8-sig')

    # -----------------------------------------
    # Save Metadata
    # -----------------------------------------
    metadata = [
        ('date_ym',          '기준연월',          'DB/ml_realestate_raw', '데이터 기준 연월 (YYYYMM 형식)'),
        ('house_price_idx',  '매매가격지수',       'DB/ml_realestate_raw', '전국 아파트 매매가격지수 (예측 타겟의 원천)'),
        ('kr_cpi',           '한국 소비자물가지수', 'DB/ml_realestate_raw', '소비자물가지수 총지수 (2020=100)'),
        ('kr_unemployment',  '한국 실업률',       'DB/ml_realestate_raw', '한국 월별 실업률 (%)'),
        ('kr_base_rate',     '한국 기준금리',     'DB/ml_realestate_raw', '한국은행 기준금리 (%)'),
        ('kr_mortgage_rate', '주택담보대출 금리',  'DB/ml_realestate_raw', '예금은행 신규취급액 주택담보대출금리 (%)'),
        ('kospi200',         'KOSPI200 지수',    'DB/ml_realestate_raw', '코스피 200 종가 지수'),
        ('apt_trade_count',  '아파트 거래량',      'DB/ml_realestate_raw', '전국 아파트 매매 거래 건수'),
        ('kr_m2',            '한국 M2 통화량',    'DB/ml_realestate_raw', '한국 광의통화 M2 (원)'),
        ('buyer_dominance',  '매수우위지수',      'DB/ml_realestate_raw', 'KB 매수우위지수 (100기준, 높을수록 매수세 강함)'),
    ]

    meta_df = pd.DataFrame(metadata, columns=['컬럼영문명', '컬럼한글명', '출처', '설명'])
    meta_df = meta_df[meta_df['컬럼영문명'].isin(df.columns)]
    meta_path = os.path.join(data_dir, 'metadata.csv')
    meta_df.to_csv(meta_path, index=False, encoding='utf-8-sig')
    print(f"Saving metadata to: {meta_path}")

    print("\n" + "=" * 55)
    print("Data collection completed successfully!")
    print("=" * 55)
    print(f"   Save Path  : {save_path}")
    print(f"   Data Size  : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"   Period     : {df['date_ym'].min()} ~ {df['date_ym'].max()}")
    print(f"   Columns    : {list(df.columns)}")
    print(f"\n   Missing values status:")
    na_found = False
    for c in df.columns:
        n = df[c].isna().sum()
        if n > 0:
            print(f"     {c}: {n} rows")
            na_found = True
    if not na_found:
        print("     No missing values found.")

if __name__ == '__main__':
    collect_all()
