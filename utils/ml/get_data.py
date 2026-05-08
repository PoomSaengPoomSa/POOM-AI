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
    ECOS 통계 검색 API로 단일 지표를 조회합니다.
    - period: 'M'(월), 'Q'(분기), 'A'(연)
    - 반환: DataFrame [date_ym, col_name]
    """
    url = (
        f"https://ecos.bok.or.kr/api/StatisticSearch"
        f"/{api_key}/json/kr/1/1000"
        f"/{stat_code}/{period}/{start}/{end}/{item_code}"
    )
    resp = requests.get(url, timeout=30)

    if resp.status_code != 200:
        print(f"     ⚠ HTTP {resp.status_code}")
        return pd.DataFrame()

    body = resp.json()
    if 'StatisticSearch' not in body:
        msg = body.get('RESULT', {}).get('MESSAGE', 'Unknown error')
        print(f"     ⚠ API 오류: {msg}")
        return pd.DataFrame()

    rows = body['StatisticSearch']['row']
    df = pd.DataFrame(rows)[['TIME', 'DATA_VALUE']].copy()
    df.columns = ['date_ym', col_name]
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
    return df


# ═══════════════════════════════════════════════════
#  미연준 FRED API 호출
# ═══════════════════════════════════════════════════
def fetch_fred(api_key, series_id, col_name='value',
               start='2014-01-01', end='2025-12-31', frequency='m'):
    """
    FRED API로 단일 지표를 월별로 조회합니다.
    일별 데이터는 월평균으로 자동 변환됩니다.
    """
    params = {
        'series_id':         series_id,
        'api_key':           api_key,
        'file_type':         'json',
        'observation_start': start,
        'observation_end':   end,
        'frequency':         frequency,
        'aggregation_method': 'avg',
    }
    resp = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params=params, timeout=30,
    )

    if resp.status_code != 200:
        print(f"     ⚠ HTTP {resp.status_code}")
        return pd.DataFrame()

    body = resp.json()
    if 'observations' not in body:
        print(f"     ⚠ 데이터 없음")
        return pd.DataFrame()

    df = pd.DataFrame(body['observations'])[['date', 'value']].copy()
    # "2014-01-01" → "201401"
    df['date_ym'] = df['date'].str[:4] + df['date'].str[5:7]
    df[col_name] = pd.to_numeric(df['value'], errors='coerce')
    return df[['date_ym', col_name]]


# ═══════════════════════════════════════════════════
#  메인 수집 함수
# ═══════════════════════════════════════════════════
def collect_all():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(dotenv_path=os.path.join(base_dir, '.env'))

    ecos_key = os.getenv('ECOS_API_KEY')
    fred_key = os.getenv('FRED_API_KEY')

    if not ecos_key:
        print("❌ .env에서 ECOS_KEY를 찾을 수 없습니다.")
        return
    if not fred_key:
        print("❌ .env에서 FRED_KEY를 찾을 수 없습니다.")
        return

    all_dfs = []   # 모든 지표를 모아두는 리스트

    # ─────────────────────────────────────────
    # 🇰🇷 ECOS: 한국은행 직접 제공 지표
    # ─────────────────────────────────────────
    print("=" * 55)
    print("🇰🇷  한국은행(ECOS) 데이터 수집")
    print("=" * 55)

    # (통계표코드, 항목코드, 주기, 시작, 종료, 컬럼명, 설명)
    ecos_indicators = [
        ('722Y001', '0101000', 'M', '201401', '202512', 'kr_base_rate',  '한국 기준금리'),
        ('901Y010', '00',      'M', '201401', '202512', 'kr_cpi',        '소비자물가지수(CPI)'),
    ]

    for stat, item, period, start, end, col, desc in ecos_indicators:
        print(f"  📊 {desc} ({stat}/{item})...")
        df = fetch_ecos(ecos_key, stat, item, period=period,
                        start=start, end=end, col_name=col)
        if not df.empty:
            all_dfs.append(df)
            print(f"     ✅ {len(df)}건")
        else:
            print(f"     ❌ 실패")
        time.sleep(0.3)

    # ─────────────────────────────────────────
    # 🇰🇷🇺🇸 FRED: 한국 + 미국 거시경제 지표
    # ─────────────────────────────────────────
    print("\n" + "=" * 55)
    print("🌐  FRED 데이터 수집 (한국 + 미국)")
    print("=" * 55)

    # (시리즈ID, 컬럼명, 설명, 주기)
    fred_indicators = [
        # ── 한국 지표 (OECD/BOK 경유) ──
        ('LRHUTTTTKRM156S', 'kr_unemployment',  '한국 실업률(OECD)',          'm'),
        ('DEXKOUS',         'kr_usd_exchange',   '원/달러 환율',               'm'),
        ('MYAGM2KRM189N',   'kr_m2',             '한국 M2 통화량',             'm'),
        ('NAEXKP01KRQ661S', 'kr_gdp_index',      '한국 GDP 지수(분기→월보간)', 'q'),

        # ── 미국 지표 ──
        ('FEDFUNDS',        'us_fed_rate',       '미국 연방기금금리',           'm'),
        ('CPIAUCSL',        'us_cpi',            '미국 CPI',                   'm'),
        ('UNRATE',          'us_unemployment',   '미국 실업률',                 'm'),
        ('GS10',            'us_treasury_10y',   '미국 10년 국채금리',          'm'),
        ('VIXCLS',          'vix',               'VIX 변동성 지수',            'm'),
        ('DCOILWTICO',      'wti_oil',           'WTI 유가',                   'm'),
    ]

    for series_id, col, desc, freq in fred_indicators:
        print(f"  📊 {desc} ({series_id})...")
        df = fetch_fred(fred_key, series_id, col_name=col, frequency=freq)
        if not df.empty:
            all_dfs.append(df)
            print(f"     ✅ {len(df)}건")
        else:
            print(f"     ❌ 실패")
        time.sleep(0.3)

    # ─────────────────────────────────────────
    # 🔗 전체 병합 및 저장
    # ─────────────────────────────────────────
    if len(all_dfs) < 2:
        print("❌ 수집된 데이터가 부족합니다. 중단합니다.")
        return

    print("\n🔗 전체 데이터 병합 중...")
    merged = all_dfs[0]
    for df in all_dfs[1:]:
        merged = pd.merge(merged, df, on='date_ym', how='outer')

    merged = merged.sort_values('date_ym').reset_index(drop=True)

    # 저장
    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    save_path = os.path.join(data_dir, 'raw_data.csv')
    merged.to_csv(save_path, index=False, encoding='utf-8-sig')

    # ─────────────────────────────────────────
    # 📋 메타데이터 저장 (컬럼 한글명 + 설명)
    # ─────────────────────────────────────────
    metadata = [
        ('date_ym',          '기준연월',          'ECOS/FRED', '데이터 기준 연월 (YYYYMM 형식)'),
        ('kr_base_rate',     '한국 기준금리',     'ECOS',      '한국은행 기준금리 (%, 722Y001). 예측 타겟의 원천'),
        ('kr_cpi',           '한국 소비자물가지수', 'ECOS',    '소비자물가지수 총지수 (2020=100, 901Y010). 인플레이션 지표'),
        ('kr_unemployment',  '한국 실업률',       'FRED/OECD', '한국 월별 실업률 (%, LRHUTTTTKRM156S). 노동시장 지표'),
        ('kr_usd_exchange',  '원/달러 환율',      'FRED',      '원/미달러 월평균 환율 (원, DEXKOUS). 외환시장 지표'),
        ('kr_m2',            '한국 M2 통화량',    'FRED',      '한국 광의통화 M2 (원, MYAGM2KRM189N). 유동성 지표'),
        ('kr_gdp_index',     '한국 GDP 지수',     'FRED/OECD', '한국 실질GDP 지수 (분기→월 보간, NAEXKP01KRQ661S). 경기 지표'),
        ('us_fed_rate',      '미국 연방기금금리',  'FRED',      '미국 연방기금 목표금리 (%, FEDFUNDS). 글로벌 금리 기준'),
        ('us_cpi',           '미국 소비자물가지수', 'FRED',     '미국 CPI (1982-84=100, CPIAUCSL). 글로벌 인플레이션'),
        ('us_unemployment',  '미국 실업률',       'FRED',      '미국 월별 실업률 (%, UNRATE). 미국 경기 지표'),
        ('us_treasury_10y',  '미국 10년 국채금리', 'FRED',      '미국 10년 만기 국채 수익률 (%, GS10). 장기금리 지표'),
        ('vix',              'VIX 변동성 지수',   'FRED',      'CBOE 변동성 지수 월평균 (VIXCLS). 시장 불안 심리 지표'),
        ('wti_oil',          'WTI 유가',          'FRED',      'WTI 원유 현물가격 월평균 ($, DCOILWTICO). 원자재·에너지 지표'),
    ]

    meta_df = pd.DataFrame(metadata, columns=['컬럼영문명', '컬럼한글명', '출처', '설명'])
    # 실제 존재하는 컬럼만 필터링
    meta_df = meta_df[meta_df['컬럼영문명'].isin(merged.columns)]
    meta_path = os.path.join(data_dir, 'metadata.csv')
    meta_df.to_csv(meta_path, index=False, encoding='utf-8-sig')
    print(f"\n📋 메타데이터 저장: {meta_path}")

    print("\n" + "=" * 55)
    print("✅ 원천 데이터 수집 완료!")
    print("=" * 55)
    print(f"   저장 경로 : {save_path}")
    print(f"   데이터 크기: {merged.shape[0]}행 × {merged.shape[1]}열")
    print(f"   기간      : {merged['date_ym'].min()} ~ {merged['date_ym'].max()}")
    print(f"   컬럼      : {list(merged.columns)}")
    print(f"\n   결측치 현황:")
    na_found = False
    for c in merged.columns:
        n = merged[c].isna().sum()
        if n > 0:
            print(f"     {c}: {n}건")
            na_found = True
    if not na_found:
        print(f"     결측치 없음! 🎉")


if __name__ == '__main__':
    collect_all()