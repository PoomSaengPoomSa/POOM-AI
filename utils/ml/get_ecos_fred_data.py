import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

# ═══════════════════════════════════════════════════
#  한국은행 ECOS API 호출
# ═══════════════════════════════════════════════════
def fetch_ecos(api_key, stat_code, item_code, period='M', 
               start='201401', end='202512', col_name='value'):
    """
    ECOS API 데이터 수집 (M: 월별, D: 일별, Q: 분기별)
    """
    url = (
        f"https://ecos.bok.or.kr/api/StatisticSearch"
        f"/{api_key}/json/kr/1/10000"
        f"/{stat_code}/{period}/{start}/{end}/{item_code}"
    )
    resp = requests.get(url, timeout=30)

    if resp.status_code != 200:
        print(f"     ⚠ HTTP {resp.status_code}")
        return pd.DataFrame()

    body = resp.json()

    if 'StatisticSearch' not in body:
        error_msg = body.get('RESULT', {}).get('MESSAGE', '알 수 없는 오류')
        print(f"     ⚠ API 수집 실패 [{stat_code}-{item_code}]: {error_msg}")
        return pd.DataFrame()

    rows = body['StatisticSearch']['row']
    df = pd.DataFrame(rows)[['TIME', 'DATA_VALUE']].copy()
    
    # 날짜 포맷팅 통일 (월별: YYYY-MM, 일별: YYYY-MM-DD, 분기별: 분기 시작월 YYYY-MM)
    if period == 'M':
        df['date'] = df['TIME'].str[:4] + '-' + df['TIME'].str[4:6]
    elif period == 'D':
        df['date'] = df['TIME'].str[:4] + '-' + df['TIME'].str[4:6] + '-' + df['TIME'].str[6:8]
    elif period == 'Q':
        # ECOS의 분기 데이터(예: 2023Q1)를 월별 포맷(2023-01)으로 매핑
        quarter_to_month = {'1': '01', '2': '04', '3': '07', '4': '10'}
        df['date'] = df['TIME'].str[:4] + '-' + df['TIME'].str[5].map(quarter_to_month)
        
    df[col_name] = pd.to_numeric(df['DATA_VALUE'], errors='coerce')
    return df[['date', col_name]]

# ═══════════════════════════════════════════════════
#  미연준 FRED API 호출
# ═══════════════════════════════════════════════════
def fetch_fred(api_key, series_id, col_name='value', 
               start='2014-01-01', end='2025-12-31', frequency='m'):
    """
    FRED API 데이터 수집 (m: 월별, d: 일별)
    """
    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': start,
        'observation_end': end,
        'frequency': frequency,
        'aggregation_method': 'avg' if frequency == 'm' else 'eop', # 월별은 평균, 일별은 종가 기준
    }
    resp = requests.get(
        "https://api.stlouisfed.org/fred/series/observations", 
        params=params, timeout=30
    )

    if resp.status_code != 200:
        print(f"     ⚠ HTTP {resp.status_code}")
        return pd.DataFrame()

    body = resp.json()
    if 'observations' not in body:
        return pd.DataFrame()

    df = pd.DataFrame(body['observations'])[['date', 'value']].copy()
    
    # 날짜 포맷팅 통일
    if frequency == 'm':
        df['date'] = df['date'].str[:7] # YYYY-MM
        
    df[col_name] = pd.to_numeric(df['value'], errors='coerce')
    # 결측치(.) 제거
    df = df.dropna(subset=[col_name])
    return df[['date', col_name]]

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
#  메인 실행 함수
# ═══════════════════════════════════════════════════
def collect_all():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    load_dotenv(dotenv_path=os.path.join(base_dir, '.env'))

    ecos_key = os.getenv('ECOS_API_KEY')
    fred_key = os.getenv('FRED_API_KEY')

    if not ecos_key or not fred_key:
        print("❌ .env 파일에서 ECOS_API_KEY 또는 FRED_API_KEY를 찾을 수 없습니다.")
        return

    # 결과 저장 디렉토리 (data/ml)
    save_dir = os.path.join(base_dir, 'data', 'ml')
    os.makedirs(save_dir, exist_ok=True)

    # 지표를 담을 리스트
    ecos_m_dfs, ecos_d_dfs = [], []
    fred_m_dfs, fred_d_dfs = [], []

    # ─────────────────────────────────────────
    # 1. ECOS 데이터 수집
    # ─────────────────────────────────────────
    print("\n🇰🇷 한국은행(ECOS) 데이터 수집 시작...")
    
    # [월별/분기별] 통계표코드, 항목코드, 컬럼명, 설명, 주기(Period)
    ecos_indicators = [
        ('722Y001', '0101000', 'kr_base_rate', '한국 기준금리', 'M'),
        ('901Y009', '0', 'kr_cpi', '한국 CPI(소비자물가지수)', 'M'),
        ('121Y006', 'BECBLA0302', 'kr_mortgage_rate', '주택담보대출금리', 'M'),
        ('161Y005', 'BBHS00', 'kr_m2', '한국 M2 통화량', 'M'),
        ('200Y102', '10111', 'kr_gdp', '한국 GDP(실질,계절조정,전기비)', 'Q') # ✨ 추가된 GDP 데이터
    ]
    
    for stat, item, col, desc, prd in ecos_indicators:
        print(f"  📊 {desc} ({'월별' if prd == 'M' else '분기별'})...")
        
        # 주기(M, Q)에 따라 요청 날짜 포맷 변경
        req_start = '201401' if prd == 'M' else '2014Q1'
        req_end = '202512' if prd == 'M' else '2025Q4'
        
        df = fetch_ecos(ecos_key, stat, item, period=prd, start=req_start, end=req_end, col_name=col)
        
        if not df.empty: 
            ecos_m_dfs.append(df) # 분기 데이터도 date가 YYYY-MM이므로 월별 리스트에 함께 병합
        time.sleep(0.3)

    # ─────────────────────────────────────────
    # 2. FRED 데이터 수집
    # ─────────────────────────────────────────
    print("\n🌐 FRED 데이터 수집 시작...")
    
    # [월별] 시리즈ID, 컬럼명, 설명
    fred_m_indicators = [
        ('DEXKOUS', 'kr_usd_exchange', '원/달러 환율'),
        ('VIXCLS', 'vix', 'VIX 변동성 지수'),
        ('DCOILWTICO', 'wti_oil', 'WTI 유가'),
        ('FEDFUNDS', 'us_fed_rate', '미국 연방기금금리'),
        ('LRHUTTTTKRM156S', 'kr_unemployment', '한국 실업률(OECD)'),
        # ECOS로 대체했으므로 M2와 GDP는 제외
    ]
    
    # [일별] 시리즈ID, 컬럼명, 설명
    fred_d_indicators = [
        ('DEXKOUS', 'kr_usd_exchange', '원/달러 환율'),
        ('VIXCLS', 'vix', 'VIX 변동성 지수'),
        ('DTWEXBGS', 'dxy_proxy', '달러 인덱스(FRED Broad)'),
        ('DCOILWTICO', 'wti_oil', 'WTI 유가')
    ]

    for series_id, col, desc in fred_m_indicators:
        print(f"  📊 {desc} (월별)...")
        df = fetch_fred(fred_key, series_id, col_name=col, frequency='m')
        if not df.empty: fred_m_dfs.append(df)
        time.sleep(0.3)

    for series_id, col, desc in fred_d_indicators:
        print(f"  📊 {desc} (일별)...")
        df = fetch_fred(fred_key, series_id, col_name=col, frequency='d')
        if not df.empty: fred_d_dfs.append(df)
        time.sleep(0.3)

    # ─────────────────────────────────────────
    # 3. 데이터 병합 및 CSV 저장
    # ─────────────────────────────────────────
    print("\n🔗 데이터를 파일별로 분리하여 저장합니다...")
    
    merge_and_save(ecos_m_dfs, os.path.join(save_dir, 'ecos_m.csv'), "ECOS 월별 데이터")
    merge_and_save(ecos_d_dfs, os.path.join(save_dir, 'ecos_d.csv'), "ECOS 일별 데이터")
    merge_and_save(fred_m_dfs, os.path.join(save_dir, 'fred_m.csv'), "FRED 월별 데이터")
    merge_and_save(fred_d_dfs, os.path.join(save_dir, 'fred_d.csv'), "FRED 일별 데이터")
    
    print("\n🎉 모든 처리가 완료되었습니다!")

if __name__ == '__main__':
    collect_all()