import os
import time
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv

# ═══════════════════════════════════════════════════
#  공통 환경 및 날짜 설정
# ═══════════════════════════════════════════════════
today = datetime.today()

# 각 API 포맷에 맞춘 동적 종료일(END DATE)
END_YM = today.strftime('%Y%m')                           # YYYYMM (ECOS, R-ONE 월별)
END_Q = f"{today.year}Q{(today.month - 1) // 3 + 1}"      # YYYYQn (ECOS 분기별)
END_DASH = today.strftime('%Y-%m-%d')                     # YYYY-MM-DD (FRED 일/월별)
# yfinance는 end 속성값의 전날까지 수집하므로 하루 더해줌
END_YF = (today + timedelta(days=1)).strftime('%Y-%m-%d') 

START_YM = '201401'
START_Q = '2014Q1'
START_DASH = '2014-01-01'

# ═══════════════════════════════════════════════════
#  1. 한국은행 ECOS 및 미연준 FRED 데이터 수집
# ═══════════════════════════════════════════════════
def fetch_ecos(api_key, stat_code, item_code, period='M', start='201401', end=END_YM, col_name='value'):
    url = (
        f"https://ecos.bok.or.kr/api/StatisticSearch"
        f"/{api_key}/json/kr/1/10000"
        f"/{stat_code}/{period}/{start}/{end}/{item_code}"
    )
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        print(f"     ⚠ ECOS HTTP {resp.status_code}")
        return pd.DataFrame()

    body = resp.json()
    if 'StatisticSearch' not in body:
        error_msg = body.get('RESULT', {}).get('MESSAGE', '알 수 없는 오류')
        print(f"     ⚠ ECOS API 수집 실패 [{stat_code}-{item_code}]: {error_msg}")
        return pd.DataFrame()

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
    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': start,
        'observation_end': end,
        'frequency': frequency,
        'aggregation_method': 'avg' if frequency == 'm' else 'eop',
    }
    resp = requests.get("https://api.stlouisfed.org/fred/series/observations", params=params, timeout=30)
    if resp.status_code != 200:
        print(f"     ⚠ FRED HTTP {resp.status_code}")
        return pd.DataFrame()

    body = resp.json()
    if 'observations' not in body:
        return pd.DataFrame()

    df = pd.DataFrame(body['observations'])[['date', 'value']].copy()
    if frequency == 'm':
        df['date'] = df['date'].str[:7]
        
    df[col_name] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=[col_name])
    return df[['date', col_name]]

# ═══════════════════════════════════════════════════
#  2. R-ONE 부동산 데이터 수집
# ═══════════════════════════════════════════════════
def get_period_list(start, end):
    periods = []
    sy, sm = int(start[:4]), int(start[4:])
    ey, em = int(end[:4]),   int(end[4:])
    y, m = sy, sm
    while (y, m) <= (ey, em):
        periods.append(f"{y}{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return periods

def fetch_reb(api_key, stat_code, item_code, col_name, region_code=None, start=START_YM, end=END_YM):
    print(f"  📥 {col_name} 수집 중...")
    periods = get_period_list(start, end)
    rows_all = []

    for period in periods:
        params = {
            'KEY':              api_key,
            'Type':             'json',
            'pIndex':           '1',
            'pSize':            '1000',
            'STATBL_ID':        stat_code,
            'DTACYCLE_CD':      'MM',
            'WRTTIME_IDTFR_ID': period,
            'ITEM_ID':          item_code,
        }
        if region_code:
            params['REGION_CD'] = region_code

        try:
            resp = requests.get('https://www.reb.or.kr/r-one/openapi/SttsApiTblData.do', params=params, timeout=30)
            body = resp.json()
            if 'SttsApiTblData' in body:
                rows = body['SttsApiTblData'][1].get('row', [])
                if rows:
                    rows_all.extend(rows)
        except Exception as e:
            print(f"     ❌ {period} 오류: {e}")
        time.sleep(0.1)

    if not rows_all:
        print(f"     ⚠️  데이터 없음")
        return pd.DataFrame()

    df = pd.DataFrame(rows_all)
    if 'CLS_ID' in df.columns:
        df = df[df['CLS_ID'] == 500001].copy()
    if 'ITM_ID' in df.columns:
        df = df[df['ITM_ID'] == 100001].copy()

    df = df[['WRTTIME_IDTFR_ID', 'DTA_VAL']].copy()
    df['date'] = df['WRTTIME_IDTFR_ID'].str[:4] + '-' + df['WRTTIME_IDTFR_ID'].str[4:6]
    df[col_name] = pd.to_numeric(df['DTA_VAL'], errors='coerce')

    print(f"     ✅ {len(df)}건 수집 완료")
    return df[['date', col_name]]

# ═══════════════════════════════════════════════════
#  3. yfinance (주가, 금, 달러) 데이터 수집
# ═══════════════════════════════════════════════════
def download_yf_all(tickers: dict, start: str, end: str) -> pd.DataFrame:
    frames = []
    for col_name, ticker in tickers.items():
        print(f"  📥 {ticker:12s} → {col_name}")
        try:
            df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
            if df.empty:
                print(f"     ⚠️  데이터 없음: {ticker}")
                continue

            if isinstance(df.columns, pd.MultiIndex):
                close = df["Close"].iloc[:, 0]
            else:
                close = df["Close"]

            close.name = col_name
            frames.append(close)
        except Exception as e:
            print(f"     ❌ 오류 ({ticker}): {e}")

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, axis=1)
    result.index.name = "date"
    return result

def make_yf_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    d = df.copy()
    if d.index.tz is not None:
        d.index = d.index.tz_localize(None)
    else:
        d.index = pd.to_datetime(d.index)

    d["gold_pct_change"] = d["gold"].pct_change() * 100
    d = d.reset_index()
    d["date"] = d["date"].dt.strftime("%Y-%m-%d")
    cols = ["date", "gold", "gold_pct_change", "sp500", "dxy", "kospi200"]
    return d[[c for c in cols if c in d.columns]]

def make_yf_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty: return daily
    m = daily.copy()
    m["date"] = pd.to_datetime(m["date"])
    m = m.set_index("date")

    numeric_cols = m.select_dtypes(include="number").columns.tolist()
    monthly = m[numeric_cols].resample("ME").mean()
    monthly = monthly.drop(columns=["gold_pct_change"], errors="ignore")
    if "gold" in monthly.columns:
        monthly["gold_pct_change_monthly"] = monthly["gold"].pct_change() * 100

    monthly = monthly.reset_index()
    monthly["date"] = monthly["date"].dt.strftime("%Y-%m")
    cols = ["date", "gold", "gold_pct_change_monthly", "sp500", "dxy", "kospi200"]
    return monthly[[c for c in cols if c in monthly.columns]]

# ═══════════════════════════════════════════════════
#  데이터 병합 및 저장 헬퍼 함수
# ═══════════════════════════════════════════════════
def merge_and_save(dfs, file_path, desc):
    if not dfs:
        print(f"⚠ {desc} 수집된 데이터가 없어 파일을 생성하지 않습니다.")
        return
        
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on='date', how='outer')
        
    merged = merged.sort_values('date').reset_index(drop=True)
    merged.to_csv(file_path, index=False, encoding='utf-8-sig')
    print(f"✅ {desc} 저장 완료: {file_path} ({len(merged)}건)")

# ═══════════════════════════════════════════════════
#  메인 실행
# ═══════════════════════════════════════════════════
def collect_all():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    load_dotenv(dotenv_path=os.path.join(base_dir, '.env'))

    ecos_key = os.getenv('ECOS_API_KEY')
    fred_key = os.getenv('FRED_API_KEY')
    reb_key = os.getenv('REB_API_KEY')

    if not all([ecos_key, fred_key, reb_key]):
        print("❌ .env 파일에서 하나 이상의 API_KEY(ECOS, FRED, REB)를 찾을 수 없습니다.")
        return

    save_dir = os.path.join(base_dir, 'data', 'ml')
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print(f"  통합 원시 데이터 수집 파이프라인 (종료일 기준: {END_DASH})")
    print("=" * 60)

    # ─────────────────────────────────────────
    # 1. ECOS 데이터
    # ─────────────────────────────────────────
    print("\n[1/4] 🇰🇷 한국은행(ECOS) 데이터 수집 시작...")
    ecos_m_dfs, ecos_d_dfs = [], []
    ecos_indicators = [
        ('722Y001', '0101000', 'kr_base_rate', '한국 기준금리', 'M'),
        ('901Y009', '0', 'kr_cpi', '한국 CPI(소비자물가지수)', 'M'),
        ('121Y006', 'BECBLA0302', 'kr_mortgage_rate', '주택담보대출금리', 'M'),
        ('161Y005', 'BBHS00', 'kr_m2', '한국 M2 통화량', 'M'),
        ('200Y102', '10111', 'kr_gdp', '한국 GDP', 'Q')
    ]
    for stat, item, col, desc, prd in ecos_indicators:
        print(f"  📊 {desc} ({'월별' if prd == 'M' else '분기별'})...")
        req_start = START_YM if prd == 'M' else START_Q
        req_end = END_YM if prd == 'M' else END_Q
        df = fetch_ecos(ecos_key, stat, item, period=prd, start=req_start, end=req_end, col_name=col)
        if not df.empty: ecos_m_dfs.append(df)
        time.sleep(0.3)

    merge_and_save(ecos_m_dfs, os.path.join(save_dir, 'ecos_m.csv'), "ECOS 월별 데이터")

    # ─────────────────────────────────────────
    # 2. FRED 데이터
    # ─────────────────────────────────────────
    print("\n[2/4] 🌐 FRED 데이터 수집 시작...")
    fred_m_dfs, fred_d_dfs = [], []
    fred_m_indicators = [
        ('DEXKOUS', 'kr_usd_exchange', '원/달러 환율'),
        ('VIXCLS', 'vix', 'VIX 변동성 지수'),
        ('DCOILWTICO', 'wti_oil', 'WTI 유가'),
        ('FEDFUNDS', 'us_fed_rate', '미국 연방기금금리'),
        ('LRHUTTTTKRM156S', 'kr_unemployment', '한국 실업률(OECD)'),
    ]
    fred_d_indicators = [
        ('DEXKOUS', 'kr_usd_exchange', '원/달러 환율'),
        ('VIXCLS', 'vix', 'VIX 변동성 지수'),
        ('DTWEXBGS', 'dxy_proxy', '달러 인덱스(FRED Broad)'),
        ('DCOILWTICO', 'wti_oil', 'WTI 유가')
    ]
    for series_id, col, desc in fred_m_indicators:
        print(f"  📊 {desc} (월별)...")
        df = fetch_fred(fred_key, series_id, col_name=col, start=START_DASH, end=END_DASH, frequency='m')
        if not df.empty: fred_m_dfs.append(df)
        time.sleep(0.3)

    for series_id, col, desc in fred_d_indicators:
        print(f"  📊 {desc} (일별)...")
        df = fetch_fred(fred_key, series_id, col_name=col, start=START_DASH, end=END_DASH, frequency='d')
        if not df.empty: fred_d_dfs.append(df)
        time.sleep(0.3)

    merge_and_save(fred_m_dfs, os.path.join(save_dir, 'fred_m.csv'), "FRED 월별 데이터")
    merge_and_save(fred_d_dfs, os.path.join(save_dir, 'fred_d.csv'), "FRED 일별 데이터")

    # ─────────────────────────────────────────
    # 3. R-ONE 부동산 데이터
    # ─────────────────────────────────────────
    print("\n[3/4] 🏢 R-ONE 부동산 데이터 수집 시작...")
    reb_indicators = [
        ('A_2024_00045', '100001', 'house_price_idx', None),
        ('A_2024_00554', '100001', 'apt_trade_count', '500001'),
        ('A_2024_00076', '100001', 'buyer_dominance', None),
    ]
    reb_dfs = []
    for stat_code, item_code, col_name, region_code in reb_indicators:
        df = fetch_reb(reb_key, stat_code, item_code, col_name, region_code, start=START_YM, end=END_YM)
        if not df.empty: reb_dfs.append(df)

    if reb_dfs:
        reb_merged = reb_dfs[0]
        for df in reb_dfs[1:]:
            reb_merged = pd.merge(reb_merged, df, on='date', how='outer')
        reb_merged = reb_merged.sort_values('date').reset_index(drop=True)
        reb_merged['house_price_idx_pct'] = reb_merged['house_price_idx'].pct_change() * 100
        
        reb_cols = ['date', 'house_price_idx', 'house_price_idx_pct', 'apt_trade_count', 'buyer_dominance']
        reb_merged = reb_merged[[c for c in reb_cols if c in reb_merged.columns]]
        
        reb_path = os.path.join(save_dir, 'realestate_m.csv')
        reb_merged.to_csv(reb_path, index=False, encoding='utf-8-sig')
        print(f"✅ R-ONE 부동산 월별 데이터 저장 완료: {reb_path} ({len(reb_merged)}건)")

    # ─────────────────────────────────────────
    # 4. yfinance 데이터
    # ─────────────────────────────────────────
    print("\n[4/4] 📈 yfinance (금융/증시) 데이터 수집 시작...")
    tickers = {"gold": "GC=F", "sp500": "^GSPC", "dxy": "DX-Y.NYB", "kospi200": "^KS200"}
    yf_raw = download_yf_all(tickers, START_DASH, END_YF)
    
    if not yf_raw.empty:
        yf_daily = make_yf_daily(yf_raw)
        yf_daily_path = os.path.join(save_dir, "yfinance_d.csv")
        yf_daily.to_csv(yf_daily_path, index=False, encoding="utf-8-sig")
        print(f"✅ yfinance 일별 데이터 저장 완료: {yf_daily_path} ({len(yf_daily)}건)")

        yf_monthly = make_yf_monthly(yf_daily)
        yf_monthly_path = os.path.join(save_dir, "yfinance_m.csv")
        yf_monthly.to_csv(yf_monthly_path, index=False, encoding="utf-8-sig")
        print(f"✅ yfinance 월별 데이터 저장 완료: {yf_monthly_path} ({len(yf_monthly)}건)")

    print("\n🎉 모든 데이터의 수집 및 통합 병합 처리가 완료되었습니다!")

if __name__ == '__main__':
    collect_all()