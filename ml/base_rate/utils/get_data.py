import os
import pymysql
import pandas as pd
from dotenv import load_dotenv, find_dotenv


def collect_all():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(base_dir, '.env')
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv(find_dotenv())

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
    print("Database (MySQL) Data Collection")
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
            # Query all records from ml_baserate_raw sorted by loaded_date
            sql = "SELECT * FROM ml_baserate_raw ORDER BY loaded_date ASC"
            print("  Querying ml_baserate_raw table...")
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

    # Rename column 'kr_gdp' to 'kr_gdp_index' for downstream compatibility
    column_mapping = {
        'kr_gdp': 'kr_gdp_index'
    }
    df = df.rename(columns=column_mapping)
    
    # Drop unwanted DB sequence/date columns
    if 'br_id' in df.columns:
        df = df.drop(columns=['br_id'])
    if 'loaded_date' in df.columns:
        df = df.drop(columns=['loaded_date'])

    # Reorder columns to place date_ym first
    cols = ['date_ym'] + [col for col in df.columns if col != 'date_ym']
    df = df[cols]

    # Convert values to numeric (handling decimals/floats safely)
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
        ('date_ym',          '기준연월',          'DB/ml_baserate_raw', '데이터 기준 연월 (YYYYMM 형식)'),
        ('kr_base_rate',     '한국 기준금리',     'DB/ml_baserate_raw', '한국은행 기준금리 (%, 722Y001). 예측 타겟의 원천'),
        ('kr_cpi',           '한국 소비자물가지수', 'DB/ml_baserate_raw', '소비자물가지수 총지수 (2020=100, 901Y010). 인플레이션 지표'),
        ('kr_unemployment',  '한국 실업률',       'DB/ml_baserate_raw', '한국 월별 실업률 (%, LRHUTTTTKRM156S). 노동시장 지표'),
        ('kr_usd_exchange',  '원/달러 환율',      'DB/ml_baserate_raw', '원/미달러 월평균 환율 (원, DEXKOUS). 외환시장 지표'),
        ('kr_m2',            '한국 M2 통화량',    'DB/ml_baserate_raw', '한국 광의통화 M2 (원, MYAGM2KRM189N). 유동성 지표'),
        ('kr_gdp_index',     '한국 GDP 지수',     'DB/ml_baserate_raw', '한국 실질GDP 지수 (분기→월 보간, NAEXKP01KRQ661S). 경기 지표'),
        ('us_fed_rate',      '미국 연방기금금리',  'DB/ml_baserate_raw', '미국 연방기금 목표금리 (%, FEDFUNDS). 글로벌 금리 기준'),
        ('vix',              'VIX 변동성 지수',   'DB/ml_baserate_raw', 'CBOE 변동성 지수 월평균 (VIXCLS). 시장 불안 심리 지표'),
        ('wti_oil',          'WTI 유가',          'DB/ml_baserate_raw', 'WTI 원유 현물가격 월평균 ($, DCOILWTICO). 원자재·에너지 지표'),
    ]

    meta_df = pd.DataFrame(metadata, columns=['컬럼영문명', '컬럼한글명', '출처', '설명'])
    # Filter metadata for actually existing columns
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
