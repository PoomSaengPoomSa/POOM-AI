import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

def load_and_split_data():
    # 1. 경로 및 환경변수 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(current_dir))
    data_dir = os.path.join(base_dir, 'data', 'ml')
    load_dotenv(dotenv_path=os.path.join(base_dir, '.env'))

    # 파일 읽기
    path_m = os.path.join(data_dir, 'rawdata_m.csv')
    path_d = os.path.join(data_dir, 'rawdata_d.csv')

    if not os.path.exists(path_m) or not os.path.exists(path_d):
        print("❌ 통합 데이터 파일(rawdata_m.csv 또는 rawdata_d.csv)이 존재하지 않습니다.")
        return

    df_m = pd.read_csv(path_m)
    df_d = pd.read_csv(path_d)

    print("📊 데이터 로드 완료. 목적별 분리 시작...")

    # 2. [금값 예측 데이터] - 일별 (Gold)
    # plan.md: 오늘 금값, 전날 대비 변화율, CPI, 환율, 유가, 달러인덱스, VIX, KOSPI200, S&P500
    # *CPI는 월별 데이터이므로 일별 병합 시 결측치가 많을 수 있음 (필요시 보간 처리 권장)
    gold_cols = [
        'date', 'gold', 'gold_pct_change', 'kr_usd_exchange', 
        'wti_oil', 'dxy_proxy', 'vix', 'kospi200', 'sp500','kr_cpi'
    ]
    gold_data = df_d[[c for c in gold_cols if c in df_d.columns]].copy()
    gold_data.to_csv(os.path.join(data_dir, 'gold_data.csv'), index=False, encoding='utf-8-sig')

    # 3. [매매가격지수 예측 데이터] - 월별 (Real Estate)
    # plan.md: 이번달 매매지수, CPI, 실업률, 기준금리, 주담대금리, KOSPI200, 변화율, 거래건수, M2 등
    re_cols = [
        'date', 'house_price_idx', 'house_price_idx_pct', 'kr_cpi', 
        'kr_unemployment', 'kr_base_rate', 'kr_mortgage_rate', 
        'kospi200', 'apt_trade_count', 'kr_m2', 'buyer_dominance'
    ]
    realestate_data = df_m[[c for c in re_cols if c in df_m.columns]].copy()
    realestate_data.to_csv(os.path.join(data_dir, 'realestate_data.csv'), index=False, encoding='utf-8-sig')

    # 4. [기준금리 예측 데이터] - 월별 (Base Rate)
    # plan.md: 이번달 금리, 변화율, CPI, 실업률, 환율, GDP, M2, 미 연준금리, VIX, 유가
    br_cols = [
        'date', 'kr_base_rate', 'kr_cpi', 'kr_unemployment', 
        'kr_usd_exchange', 'kr_gdp', 'kr_m2', 'us_fed_rate', 'vix', 'wti_oil'
    ]
    baserate_data = df_m[[c for c in br_cols if c in df_m.columns]].copy()
    baserate_data.to_csv(os.path.join(data_dir, 'baserate_data.csv'), index=False, encoding='utf-8-sig')

    print(f"✅ CSV 파일 저장 완료: data/ml/ 내 gold, realestate, baserate_data.csv")

    # 5. MySQL 데이터베이스 적재
    upload_to_mysql(
        {'ml_gold': gold_data, 'ml_realestate': realestate_data, 'ml_baserate': baserate_data}
    )

def upload_to_mysql(data_dict):
    # .env에서 DB 정보 로드
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT', '3306')
    db_name = os.getenv('DB_NAME')

    if not all([db_user, db_password, db_host, db_name]):
        print("⚠ DB 연결 정보가 부족하여 데이터베이스 적재를 건너뜁니다.")
        return

    try:
        # SQLAlchemy 엔진 생성 (pymysql 드라이버 사용)
        engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?charset=utf8mb4")

        for table_name, df in data_dict.items():
            # 데이터프레임을 MySQL 테이블로 저장
            df.to_sql(name=table_name, con=engine, if_exists='replace', index=False)
            print(f"🚀 DB 적재 완료: {table_name} ({len(df)} rows)")

        print("\n🎉 모든 데이터 분석 준비 및 DB 적재가 완료되었습니다!")

    except Exception as e:
        print(f"❌ DB 적재 중 오류 발생: {e}")

if __name__ == '__main__':
    load_and_split_data()