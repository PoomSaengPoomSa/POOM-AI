import os
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# ═══════════════════════════════════════════════════
#  공통 환경 및 날짜 설정 (최근 60일 치만 수집하여 계산 보정)
# ═══════════════════════════════════════════════════
today = datetime.today()
start_date_obj = today - timedelta(days=60)

# 동적 시작일/종료일
START_YM = start_date_obj.strftime('%Y%m')
START_Q = f"{start_date_obj.year}Q{(start_date_obj.month - 1) // 3 + 1}"
START_DASH = start_date_obj.strftime('%Y-%m-%d')

END_YM = today.strftime('%Y%m')                           
END_Q = f"{today.year}Q{(today.month - 1) // 3 + 1}"      
END_DASH = today.strftime('%Y-%m-%d')                     
END_YF = (today + timedelta(days=1)).strftime('%Y-%m-%d') 

# ═══════════════════════════════════════════════════
#  1. 데이터 수집 함수 (기존 로직 동일)
# ═══════════════════════════════════════════════════
def fetch_ecos(api_key, stat_code, item_code, period='M', start=START_YM, end=END_YM, col_name='value'):
    url = f"https://ecos.bok.or.kr/api/StatisticSearch/{api_key}/json/kr/1/10000/{stat_code}/{period}/{start}/{end}/{item_code}"
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200: return pd.DataFrame()
    body = resp.json()
    if 'StatisticSearch' not in body: return pd.DataFrame()

    rows = body['StatisticSearch']['row']
    df = pd.DataFrame(rows)[['TIME', 'DATA_VALUE']].copy()
    if period == 'M':
        df['date'] = df['TIME'].str[:4] + '-' + df['TIME'].str[4:6]
    elif period == 'D':
        df['date'] = df['TIME'].str[:4] + '-' + df['TIME'].str[4:6] + '-' + df['TIME'].str[6:8]
    elif period == 'Q':
        quarter_to_month = {'1': '01', '2': '04', '3': '07', '4': '10'}
        df['date'] = df['TIME'].str[:4] + '-' + df['TIME'].str[5].map(quarter_to_month)
    df[col_name] = pd.to_numeric(df['DATA_VALUE'], errors='coerce')
    return df[['date', col_name]]

def fetch_fred(api_key, series_id, col_name='value', start=START_DASH, end=END_DASH, frequency='m'):
    params = {'series_id': series_id, 'api_key': api_key, 'file_type': 'json', 'observation_start': start, 'observation_end': end, 'frequency': frequency, 'aggregation_method': 'avg' if frequency == 'm' else 'eop'}
    resp = requests.get("https://api.stlouisfed.org/fred/series/observations", params=params, timeout=30)
    if resp.status_code != 200: return pd.DataFrame()
    body = resp.json()
    
    # 🌟 [수정 지점 1] observations 키가 없거나, 리스트가 비어있을(len==0) 경우 빈 DF 반환
    if 'observations' not in body or not body['observations']: 
        return pd.DataFrame()

    df = pd.DataFrame(body['observations'])
    
    # 🌟 [수정 지점 2] DataFrame은 만들어졌으나 date나 value 컬럼이 없는 경우 방어
    if 'date' not in df.columns or 'value' not in df.columns:
        return pd.DataFrame()

    df = df[['date', 'value']].copy()
    if frequency == 'm': df['date'] = df['date'].str[:7]
    df[col_name] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=[col_name])
    return df[['date', col_name]]

def get_period_list(start, end):
    periods, sy, sm = [], int(start[:4]), int(start[4:])
    ey, em = int(end[:4]), int(end[4:])
    y, m = sy, sm
    while (y, m) <= (ey, em):
        periods.append(f"{y}{m:02d}")
        m += 1
        if m > 12: m, y = 1, y + 1
    return periods

def fetch_reb(api_key, stat_code, item_code, col_name, region_code=None, start=START_YM, end=END_YM):
    periods = get_period_list(start, end)
    rows_all = []
    for period in periods:
        params = {'KEY': api_key, 'Type': 'json', 'pIndex': '1', 'pSize': '1000', 'STATBL_ID': stat_code, 'DTACYCLE_CD': 'MM', 'WRTTIME_IDTFR_ID': period, 'ITEM_ID': item_code}
        if region_code: params['REGION_CD'] = region_code
        try:
            resp = requests.get('https://www.reb.or.kr/r-one/openapi/SttsApiTblData.do', params=params, timeout=30)
            body = resp.json()
            if 'SttsApiTblData' in body:
                rows = body['SttsApiTblData'][1].get('row', [])
                if rows: rows_all.extend(rows)
        except Exception: pass
        time.sleep(0.1)

    if not rows_all: return pd.DataFrame()
    df = pd.DataFrame(rows_all)
    if 'CLS_ID' in df.columns: df = df[df['CLS_ID'] == 500001].copy()
    if 'ITM_ID' in df.columns: df = df[df['ITM_ID'] == 100001].copy()
    df = df[['WRTTIME_IDTFR_ID', 'DTA_VAL']].copy()
    df['date'] = df['WRTTIME_IDTFR_ID'].str[:4] + '-' + df['WRTTIME_IDTFR_ID'].str[4:6]
    df[col_name] = pd.to_numeric(df['DTA_VAL'], errors='coerce')
    return df[['date', col_name]]

def download_yf_all(tickers: dict, start: str, end: str) -> pd.DataFrame:
    frames = []
    for col_name, ticker in tickers.items():
        try:
            df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
            if df.empty: continue
            close = df["Close"].iloc[:, 0] if isinstance(df.columns, pd.MultiIndex) else df["Close"]
            close.name = col_name
            frames.append(close)
        except Exception: pass
    if not frames: return pd.DataFrame()
    result = pd.concat(frames, axis=1)
    result.index.name = "date"
    return result

def make_yf_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    d = df.copy()
    d.index = d.index.tz_localize(None) if d.index.tz is not None else pd.to_datetime(d.index)
    d = d.reset_index()
    d["date"] = d["date"].dt.strftime("%Y-%m-%d")
    cols = ["date", "gold", "sp500", "dxy", "kospi200"]
    return d[[c for c in cols if c in d.columns]]

def make_yf_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty: return daily
    m = daily.copy()
    m["date"] = pd.to_datetime(m["date"])
    m = m.set_index("date")
    monthly = m.select_dtypes(include="number").resample("ME").mean().reset_index()
    monthly["date"] = monthly["date"].dt.strftime("%Y-%m")
    cols = ["date", "gold", "sp500", "dxy", "kospi200"]
    return monthly[[c for c in cols if c in monthly.columns]]


# ═══════════════════════════════════════════════════
#  2. DB 증분 적재(Append) 함수
# ═══════════════════════════════════════════════════
def get_max_date_from_db(engine, table_name):
    """DB에 저장된 가장 최근 날짜를 가져옵니다. 테이블이 없으면 None 반환"""
    try:
        with engine.connect() as conn:
            query = text(f"SELECT MAX(date) FROM {table_name}")
            result = conn.execute(query).scalar()
            return result
    except Exception:
        return None

def upload_new_records(df, table_name, engine):
    """DB의 마지막 날짜 이후의 새로운 데이터만 필터링하여 Append"""
    if df.empty: return

    max_date = get_max_date_from_db(engine, table_name)
    
    if max_date:
        # DB에 데이터가 존재하면 최신 날짜 이후의 데이터만 슬라이싱
        new_data = df[df['date'] > max_date].copy()
    else:
        # 테이블이 처음 생성되는 경우 전체 적재
        new_data = df.copy()

    if new_data.empty:
        print(f" ╰─ ⏸️ [{table_name}] 이미 최신 상태입니다. (업데이트 없음)")
    else:
        new_data.to_sql(name=table_name, con=engine, if_exists='append', index=False)
        print(f" ╰─ 🚀 [{table_name}] 새로운 데이터 {len(new_data)}건 적재 완료! (이후 날짜: {new_data['date'].iloc[0]} ~)")

# ═══════════════════════════════════════════════════
#  3. 메인 실행 파이프라인
# ═══════════════════════════════════════════════════
def run_daily_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    load_dotenv(dotenv_path=os.path.join(base_dir, '.env'))

    # API Keys & DB Config
    ecos_key = os.getenv('ECOS_API_KEY')
    fred_key = os.getenv('FRED_API_KEY')
    reb_key = os.getenv('REB_API_KEY')
    db_user, db_password = os.getenv('DB_USER'), os.getenv('DB_PASSWORD')
    db_host, db_port, db_name = os.getenv('DB_HOST'), os.getenv('DB_PORT', '3306'), os.getenv('DB_NAME')

    if not all([ecos_key, fred_key, reb_key, db_user, db_password, db_host, db_name]):
        print("❌ .env 설정(API KEY 또는 DB 정보)이 누락되었습니다.")
        return

    engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?charset=utf8mb4")

    print(f"\n🔄 [1/3] 최근 60일 원시 데이터 수집 중... ({START_DASH} ~ {END_DASH})")
    
    ecos_m_dfs, fred_m_dfs, fred_d_dfs, reb_dfs = [], [], [], []
    
    # ECOS
    ecos_inds = [('722Y001', '0101000', 'kr_base_rate', 'M'), ('901Y009', '0', 'kr_cpi', 'M'), ('121Y006', 'BECBLA0302', 'kr_mortgage_rate', 'M'), ('161Y005', 'BBHS00', 'kr_m2', 'M'), ('200Y102', '10111', 'kr_gdp', 'Q')]
    for stat, item, col, prd in ecos_inds:
        df = fetch_ecos(ecos_key, stat, item, prd, START_YM if prd=='M' else START_Q, END_YM if prd=='M' else END_Q, col)
        if not df.empty: ecos_m_dfs.append(df)
        time.sleep(0.2)
        
    # FRED
    for sid, col in [('DEXKOUS', 'kr_usd_exchange'), ('VIXCLS', 'vix'), ('DCOILWTICO', 'wti_oil'), ('FEDFUNDS', 'us_fed_rate'), ('LRHUTTTTKRM156S', 'kr_unemployment')]:
        df = fetch_fred(fred_key, sid, col, START_DASH, END_DASH, 'm')
        if not df.empty: fred_m_dfs.append(df)
        time.sleep(0.2)
    for sid, col in [('DEXKOUS', 'kr_usd_exchange'), ('VIXCLS', 'vix'), ('DTWEXBGS', 'dxy_proxy'), ('DCOILWTICO', 'wti_oil')]:
        df = fetch_fred(fred_key, sid, col, START_DASH, END_DASH, 'd')
        if not df.empty: fred_d_dfs.append(df)
        time.sleep(0.2)

    # R-ONE
    for stat, item, col, rcode in [('A_2024_00045', '100001', 'house_price_idx', None), ('A_2024_00554', '100001', 'apt_trade_count', '500001'), ('A_2024_00076', '100001', 'buyer_dominance', None)]:
        df = fetch_reb(reb_key, stat, item, col, rcode, START_YM, END_YM)
        if not df.empty: reb_dfs.append(df)

    # YFinance
    yf_raw = download_yf_all({"gold": "GC=F", "sp500": "^GSPC", "dxy": "DX-Y.NYB", "kospi200": "^KS200"}, START_DASH, END_YF)
    yf_daily = make_yf_daily(yf_raw)
    yf_monthly = make_yf_monthly(yf_daily)

    print("🧩 [2/3] 메모리 상 데이터 병합 및 결측치 보정 중...")
    
    # --- 월별 병합 (master_m) ---
    all_m_dfs = ecos_m_dfs + fred_m_dfs + reb_dfs + ([yf_monthly] if not yf_monthly.empty else [])
    if all_m_dfs:
        master_m = all_m_dfs[0]
        for df in all_m_dfs[1:]:
            master_m = pd.merge(master_m, df, on='date', how='outer')
        master_m['date'] = pd.to_datetime(master_m['date']).dt.strftime('%Y-%m')
        master_m = master_m.sort_values('date').reset_index(drop=True)
        # GDP 결측치 보정 (ffill)
        if 'kr_gdp' in master_m.columns: master_m['kr_gdp'] = master_m['kr_gdp'].ffill(limit=2)
    else:
        master_m = pd.DataFrame()

    # --- 일별 병합 (master_d) ---
    all_d_dfs = fred_d_dfs + ([yf_daily] if not yf_daily.empty else [])
    if all_d_dfs:
        master_d = all_d_dfs[0]
        for df in all_d_dfs[1:]:
            master_d = pd.merge(master_d, df, on='date', how='outer')
        master_d['date'] = pd.to_datetime(master_d['date'])
        
        # 월별 지표(CPI 등)를 일별로 Broadcasting
        if not master_m.empty and 'kr_cpi' in master_m.columns:
            master_d['year_month'] = master_d['date'].dt.strftime('%Y-%m')
            cpi_subset = master_m[['date', 'kr_cpi']].copy()
            cpi_subset.rename(columns={'date': 'year_month'}, inplace=True)
            master_d = pd.merge(master_d, cpi_subset, on='year_month', how='left')
            master_d.drop(columns=['year_month'], inplace=True)
            
        master_d['date'] = master_d['date'].dt.strftime('%Y-%m-%d')
        master_d = master_d.sort_values('date').reset_index(drop=True)
    else:
        master_d = pd.DataFrame()

    print("💾 [3/3] DB 증분 적재 (새로운 행만 추가) 중...")
    
    # Gold (Daily)
    gold_cols = ['date', 'gold', 'kr_usd_exchange', 'wti_oil', 'dxy_proxy', 'vix', 'kospi200', 'sp500', 'kr_cpi']
    gold_data = master_d[[c for c in gold_cols if c in master_d.columns]].copy() if not master_d.empty else pd.DataFrame()
    upload_new_records(gold_data, 'ml_gold', engine)

    # Real Estate (Monthly)
    re_cols = ['date', 'house_price_idx', 'kr_cpi', 'kr_unemployment', 'kr_base_rate', 'kr_mortgage_rate', 'kospi200', 'apt_trade_count', 'kr_m2', 'buyer_dominance']
    re_data = master_m[[c for c in re_cols if c in master_m.columns]].copy() if not master_m.empty else pd.DataFrame()
    upload_new_records(re_data, 'ml_realestate', engine)

    # Base Rate (Monthly)
    br_cols = ['date', 'kr_base_rate', 'kr_cpi', 'kr_unemployment', 'kr_usd_exchange', 'kr_gdp', 'kr_m2', 'us_fed_rate', 'vix', 'wti_oil']
    br_data = master_m[[c for c in br_cols if c in master_m.columns]].copy() if not master_m.empty else pd.DataFrame()
    upload_new_records(br_data, 'ml_baserate', engine)

    print("\n🎉 일일 데이터 자동 수집 및 적재 파이프라인 완료!")

if __name__ == '__main__':
    run_daily_pipeline()
