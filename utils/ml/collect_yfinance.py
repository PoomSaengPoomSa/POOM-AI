"""
yfinance 데이터 수집
기간: 2014-01-01 ~ 2025-02-28
출력: yfinance_d.csv (일별), yfinance_m.csv (월별 평균)

수집 항목:
  GC=F      금 선물 종가 (USD/oz)
  ^GSPC     S&P 500
  DX-Y.NYB  달러 인덱스
  ^KS200    KOSPI 200
"""

import yfinance as yf
import pandas as pd

START = "2014-01-01"
END   = "2025-02-28"

# 컬럼명: 티커 심볼
TICKERS = {
    "gold":     "GC=F",
    "sp500":    "^GSPC",
    "dxy":      "DX-Y.NYB",
    "kospi200": "^KS200",
}


# ── STEP 1. 다운로드 ────────────────────────────────────────
# 각 티커 일별 종가를 받아서 날짜 기준으로 병합
# 미국/한국 휴장일이 달라 한쪽 컬럼만 NaN인 행 발생 → 그대로 유지
def download_all(tickers: dict, start: str, end: str) -> pd.DataFrame:
    frames = []

    for col_name, ticker in tickers.items():
        print(f"  📥 {ticker:12s} → {col_name}")
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval="1d",    # 일별
                auto_adjust=True, # 분할·배당 보정
                progress=False,
            )
            if df.empty:
                print(f"     ⚠️  데이터 없음: {ticker}")
                continue

            # yfinance 1.x 이상은 MultiIndex 컬럼으로 반환될 수 있음 → Close만 추출
            if isinstance(df.columns, pd.MultiIndex):
                close = df["Close"].iloc[:, 0]
            else:
                close = df["Close"]

            close.name = col_name
            frames.append(close)

        except Exception as e:
            print(f"     ❌ 오류 ({ticker}): {e}")

    if not frames:
        raise RuntimeError("수집된 데이터가 없습니다.")

    result = pd.concat(frames, axis=1)  # 날짜 기준으로 옆으로 붙이기
    result.index.name = "date"
    return result


# ── STEP 2. 일별 전처리 → yfinance_d.csv ───────────────────
def make_daily(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # timezone 제거 (yfinance가 UTC 붙여서 줄 때 있음)
    if d.index.tz is not None:
        d.index = d.index.tz_localize(None)
    else:
        d.index = pd.to_datetime(d.index)

    # 금값 전일 대비 변화율 (%) = (오늘 - 어제) / 어제 × 100
    d["gold_pct_change"] = d["gold"].pct_change() * 100

    # NaN 현황 출력 (주말·공휴일로 인한 정상 결측, 제거 안 함)
    nan_counts = d.isnull().sum()
    if nan_counts.any():
        print("     [NaN 현황]")
        for col, cnt in nan_counts[nan_counts > 0].items():
            print(f"       - {col}: {cnt}개")

    # 날짜 포맷 통일: datetime → "YYYY-MM-DD"
    d = d.reset_index()
    d["date"] = d["date"].dt.strftime("%Y-%m-%d")

    cols = ["date", "gold", "gold_pct_change",
            "sp500", "dxy", "kospi200"]
    return d[[c for c in cols if c in d.columns]]


# ── STEP 3. 월별 변환 → yfinance_m.csv ─────────────────────
# 일별 → 월평균 집계 (FRED aggregation_method='avg' 와 동일)
# gold_pct_change는 월평균 기준으로 재계산
def make_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    m = daily.copy()

    # resample()은 DatetimeIndex 필요
    m["date"] = pd.to_datetime(m["date"])
    m = m.set_index("date")

    # 수치 컬럼만 선택 후 월평균 집계
    numeric_cols = m.select_dtypes(include="number").columns.tolist()
    monthly = m[numeric_cols].resample("ME").mean()

    # 일별 변화율 제거 후 월평균 기준으로 재계산
    # = (이번달 평균 - 지난달 평균) / 지난달 평균 × 100
    monthly = monthly.drop(columns=["gold_pct_change"], errors="ignore")
    monthly["gold_pct_change_monthly"] = monthly["gold"].pct_change() * 100

    # 날짜 포맷: "YYYY-MM"
    monthly = monthly.reset_index()
    monthly["date"] = monthly["date"].dt.strftime("%Y-%m")

    cols = ["date", "gold", "gold_pct_change_monthly",
            "sp500", "dxy", "kospi200"]
    return monthly[[c for c in cols if c in monthly.columns]]


# ── 메인 실행 ───────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print(f"  yfinance 데이터 수집  |  {START} ~ {END}")
    print("=" * 50)

    print("\n[1/3] 다운로드...")
    raw = download_all(TICKERS, START, END)
    print(f"  → shape: {raw.shape}")

    print("\n[2/3] 일별 CSV 저장...")
    daily_df = make_daily(raw)
    daily_df.to_csv("yfinance_d.csv", index=False, encoding="utf-8-sig")
    print(f"  ✅ yfinance_d.csv ({len(daily_df):,}행)")

    print("\n[3/3] 월별 CSV 저장...")
    monthly_df = make_monthly(daily_df)
    monthly_df.to_csv("yfinance_m.csv", index=False, encoding="utf-8-sig")
    print(f"  ✅ yfinance_m.csv ({len(monthly_df):,}개월)")

    print("\n── yfinance_d.csv 최근 3행 ──")
    print(daily_df.tail(3).to_string(index=False))
    print("\n── yfinance_m.csv 최근 3행 ──")
    print(monthly_df.tail(3).to_string(index=False))
    print("\n✅ 완료!")
