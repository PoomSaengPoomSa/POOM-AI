"""
수빈 담당: R-ONE 부동산 데이터 수집
기간: 2014-01 ~ 2025-02
출력: realestate_m.csv (월별)

수집 항목:
  A_2024_00045  매매가격지수_아파트 (전국)
  A_2024_00554  아파트매매거래현황 (전국, 동호수=거래건수)
  A_2024_00076  매매수급동향_아파트 (전국) ← 매수우위지수

R-ONE API 특성:
  - WRTTIME_IDTFR_ID에 시작월만 입력하면 해당 월 데이터만 반환
  - 기간 범위 파라미터가 작동하지 않아 월별로 반복 호출
"""

import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

# .env 파일 로드 (utils/ml/ 기준 3단계 상위 = POOM/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, '.env'))

API_KEY = os.getenv('REB_API_KEY')
START   = '201401'
END     = '202502'


def get_period_list(start, end):
    """YYYYMM 형식의 월 리스트 생성 (예: ['201401', '201402', ...])"""
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


# ── STEP 1. R-ONE API 월별 반복 호출 ───────────────────────
# API가 기간 범위를 지원하지 않아 월별로 1건씩 호출
def fetch_reb(stat_code, item_code, col_name, region_code=None):
    print(f"  📥 {col_name} 수집 중...")

    periods = get_period_list(START, END)
    rows_all = []

    for period in periods:
        params = {
            'KEY':              API_KEY,
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
            resp = requests.get(
                'https://www.reb.or.kr/r-one/openapi/SttsApiTblData.do',
                params=params,
                timeout=30
            )
            body = resp.json()

            if 'SttsApiTblData' not in body:
                continue

            rows = body['SttsApiTblData'][1].get('row', [])
            if rows:
                rows_all.extend(rows)

        except Exception as e:
            print(f"     ❌ {period} 오류: {e}")

        time.sleep(0.1)  # API 호출 간격

    if not rows_all:
        print(f"     ⚠️  데이터 없음")
        return pd.DataFrame()

    df = pd.DataFrame(rows_all)

    # 전국(CLS_ID=500001) + 동호수(ITM_ID=100001)만 필터링
    # ITM_ID=100001: 동(호)수=거래건수, ITM_ID=100002: 면적 (불필요)
    if 'CLS_ID' in df.columns:
        df = df[df['CLS_ID'] == 500001].copy()
    if 'ITM_ID' in df.columns:
        df = df[df['ITM_ID'] == 100001].copy()

    df = df[['WRTTIME_IDTFR_ID', 'DTA_VAL']].copy()

    # 날짜 포맷: "202401" → "2024-01"
    df['date'] = df['WRTTIME_IDTFR_ID'].str[:4] + '-' + df['WRTTIME_IDTFR_ID'].str[4:6]

    # 문자열 → 숫자 변환
    df[col_name] = pd.to_numeric(df['DTA_VAL'], errors='coerce')

    print(f"     ✅ {len(df)}건 수집 완료")
    return df[['date', col_name]]


# ── 메인 실행 ───────────────────────────────────────────────
if __name__ == "__main__":
    if not API_KEY:
        print("❌ .env 파일에서 REB_API_KEY를 찾을 수 없습니다.")
        raise SystemExit(0)

    print("=" * 50)
    print(f"  R-ONE 부동산 데이터 수집  |  {START} ~ {END}")
    print(f"  총 {len(get_period_list(START, END))}개월 × 3개 항목 호출")
    print("=" * 50)

    # (통계표코드, 항목코드, 컬럼명, 지역코드)
    indicators = [
        ('A_2024_00045', '100001', 'house_price_idx', None),      # 매매가격지수
        ('A_2024_00554', '100001', 'apt_trade_count', '500001'),  # 아파트 거래건수
        ('A_2024_00076', '100001', 'buyer_dominance', None),      # 매수우위지수
    ]

    dfs = []
    for stat_code, item_code, col_name, region_code in indicators:
        df = fetch_reb(stat_code, item_code, col_name, region_code)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        print("❌ 수집된 데이터가 없습니다.")
        raise SystemExit(0)

    # 날짜 기준으로 병합
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on='date', how='outer')

    merged = merged.sort_values('date').reset_index(drop=True)

    # 매매가격지수 전월 대비 변화율 추가
    merged['house_price_idx_pct'] = merged['house_price_idx'].pct_change() * 100

    # 컬럼 순서 정리
    cols = [
        'date',
        'house_price_idx',      # 아파트 매매가격지수 (전국)
        'house_price_idx_pct',  # 전월 대비 변화율 (%)
        'apt_trade_count',      # 아파트 매매 거래건수 (전국)
        'buyer_dominance',      # 매수우위지수 (전국)
    ]
    merged = merged[[c for c in cols if c in merged.columns]]

    # CSV 저장
    save_dir = os.path.join(BASE_DIR, 'data', 'ml')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'realestate_m.csv')
    merged.to_csv(save_path, index=False, encoding='utf-8-sig')

    print(f"\n✅ realestate_m.csv 저장 완료 ({len(merged)}개월)")
    print(merged.to_string(index=False))