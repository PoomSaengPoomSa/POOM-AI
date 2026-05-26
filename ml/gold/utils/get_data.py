import os
import pymysql
import pandas as pd
from dotenv import load_dotenv

def collect_all():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.abspath(os.path.join(base_dir, '../../.env'))
    load_dotenv(dotenv_path=env_path)

    # Load DB credentials
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    DB_NAME = os.getenv('DB_NAME')

    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
        print("[Error] Missing DB configuration in .env file (DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME).")
        return

    DB_PORT = int(DB_PORT)

    print("=" * 55)
    print("Database (MySQL) Gold Data Collection")
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
            # Query all records from ml_gold_raw sorted by loaded_date
            sql = "SELECT * FROM ml_gold_raw ORDER BY loaded_date ASC"
            print("  Querying ml_gold_raw table...")
            cursor.execute(sql)
            rows = cursor.fetchall()
            
        connection.close()
    except Exception as e:
        print(f"  [Warning] DB Connection & Query failed: {e}")
        save_path = os.path.join(base_dir, 'data', 'raw_data.csv')
        if os.path.exists(save_path):
            print(f"  [OK] Found cached raw_data.csv. Proceeding with cache.")
            return
        else:
            print(f"  [Error] No cached raw_data.csv found. Halting.")
            raise e

    if not rows:
        print("  [Error] No data found. Terminating.")
        return

    print(f"  Successfully collected {len(rows)} rows.")

    # Convert to DataFrame
    df = pd.DataFrame(rows)

    # Format loaded_date (datetime -> string YYYY-MM-DD)
    df['loaded_date'] = pd.to_datetime(df['loaded_date']).dt.strftime('%Y-%m-%d')
    
    # Drop unwanted DB sequence/ID columns
    if 'gr_id' in df.columns:
        df = df.drop(columns=['gr_id'])

    # Reorder columns to place loaded_date first
    cols = ['loaded_date'] + [col for col in df.columns if col != 'loaded_date']
    df = df[cols]

    # Convert values to numeric (except loaded_date)
    numeric_cols = [col for col in df.columns if col != 'loaded_date']
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
        ('loaded_date',      '데이터 기준 일자',  'DB/ml_gold_raw', '데이터 수집 및 기준 일자 (YYYY-MM-DD 형식)'),
        ('gold',             '국제 금 선물 가격', 'DB/ml_gold_raw', '온스당 국제 금 선물 가격 ($)'),
        ('kr_usd_exchange',  '원/달러 환율',      'DB/ml_gold_raw', '원/미달러 하루 종가 환율 (원)'),
        ('wti_oil',          'WTI 서부텍사스유 가격', 'DB/ml_gold_raw', 'WTI 원유 현물 종가 가격 ($)'),
        ('dxy_proxy',        '달러 인덱스',       'DB/ml_gold_raw', '미국 달러화 가치를 보여주는 달러 인덱스 (DXY) 프록시'),
        ('vix',              'VIX 변동성 지수',   'DB/ml_gold_raw', 'CBOE 변동성 지수 (S&P 500 옵션 기반 시장 불안 심리 지표)'),
        ('kospi200',         '코스피 200 지수',   'DB/ml_gold_raw', '한국 주식시장 대표 KOSPI 200 종가 지수'),
        ('sp500',            'S&P 500 지수',      'DB/ml_gold_raw', '미국 주식시장 대표 S&P 500 종가 지수'),
        ('kr_cpi',           '한국 소비자물가지수', 'DB/ml_gold_raw', '한국 소비자물가지수 총지수 (2020=100)'),
    ]

    meta_df = pd.DataFrame(metadata, columns=['컬럼영문명', '컬럼한글명', '출처', '설명'])
    meta_df = meta_df[meta_df['컬럼영문명'].isin(df.columns)]
    meta_path = os.path.join(data_dir, 'metadata.csv')
    meta_df.to_csv(meta_path, index=False, encoding='utf-8-sig')
    print(f"Saving metadata to: {meta_path}")

    print("\n" + "=" * 55)
    print("Gold data collection completed successfully!")
    print("=" * 55)
    print(f"   Save Path  : {save_path}")
    print(f"   Data Size  : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"   Period     : {df['loaded_date'].min()} ~ {df['loaded_date'].max()}")
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
