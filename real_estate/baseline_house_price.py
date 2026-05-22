# =============================================================================
# 📌 매매 가격 지수 변화율 예측 | Baseline
# =============================================================================
# Target : next_change_rate = (idx[t+1] - idx[t]) / idx[t] * 100
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =============================================================================
# ⚙️  설정
# =============================================================================

SEED        = 42
DATA_PATH   = "/mnt/user-data/uploads/realestate_raw1.csv"
TEST_MONTHS = 24   # 최근 N개월 → Test

FEATURES = [
    "house_price_idx",   # 이번달 매매가격지수
    "kr_cpi",            # 소비자물가지수 (CPI)
    "kr_unemployment",   # 실업률
    "kr_base_rate",      # 한국 기준금리
    "kr_mortgage_rate",  # 주택담보대출 금리
    "kospi200",          # KOSPI200
    "apt_trade_count",   # 아파트 거래건수
    "kr_m2",             # M2 통화량
    "buyer_dominance",   # 매수우위지수
]
TARGET = "next_change_rate"

random.seed(SEED)
np.random.seed(SEED)


# =============================================================================
# 📂  데이터 로드
# =============================================================================

df = pd.read_csv(DATA_PATH)
df["loaded_date"] = pd.to_datetime(df["loaded_date"])
df = df.sort_values("loaded_date").reset_index(drop=True)

print("Shape :", df.shape)
print("Period:", df["loaded_date"].min().strftime("%Y-%m"), "~", df["loaded_date"].max().strftime("%Y-%m"))
print("\n[결측값]")
print(df.isnull().sum()[df.isnull().sum() > 0])


# =============================================================================
# 🛠️  전처리
# =============================================================================

# 타겟 생성: shift(-1)로 다음달 지수를 현재 행으로 당겨온 뒤 변화율 계산
df["next_house_price_idx"] = df["house_price_idx"].shift(-1)
df[TARGET] = (df["next_house_price_idx"] - df["house_price_idx"]) / df["house_price_idx"] * 100

# 결측값 처리: ffill(앞값으로 채움) → bfill(남은 앞부분 처리)
df[FEATURES] = df[FEATURES].ffill().bfill()

# 마지막 행은 다음달 데이터 없어 타겟 NaN → 제거
df = df.dropna(subset=[TARGET]).copy()

print(f"\n학습 가용 행: {len(df)}")
print(f"\n[타겟 분포]\n{df[TARGET].describe().round(4)}")


# =============================================================================
# ✂️  Train / Test 분할
# =============================================================================
# 시계열 → 시간 순서 유지 필수, shuffle 금지

train = df.iloc[:-TEST_MONTHS].copy()
test  = df.iloc[-TEST_MONTHS:].copy()

X_train, y_train = train[FEATURES], train[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]

print(f"\nTrain : {train['loaded_date'].min().strftime('%Y-%m')} ~ {train['loaded_date'].max().strftime('%Y-%m')} ({len(train)}개월)")
print(f"Test  : {test['loaded_date'].min().strftime('%Y-%m')} ~ {test['loaded_date'].max().strftime('%Y-%m')} ({len(test)}개월)")


# =============================================================================
# ⚖️  스케일링
# =============================================================================
# fit은 Train에만, Test는 transform만 적용 (데이터 누수 방지)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


# =============================================================================
# 🤖  모델 학습
# =============================================================================

MODELS = {
    "LinearRegression": LinearRegression(),
    "Ridge":            Ridge(alpha=1.0),
    "Lasso":            Lasso(alpha=0.01, max_iter=5000),
}

for name, model in MODELS.items():
    model.fit(X_train_sc, y_train)


# =============================================================================
# 📊  평가
# =============================================================================

def evaluate(y_true, y_pred):
    return {
        "MAE":  round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "R2":   round(r2_score(y_true, y_pred), 4),
    }

results = {}
print(f"\n{'모델':<20} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
print("-" * 46)

for name, model in MODELS.items():
    pred = model.predict(X_test_sc)
    results[name] = {"pred": pred, **evaluate(y_test, pred)}
    r = results[name]
    print(f"{name:<20} {r['MAE']:>8} {r['RMSE']:>8} {r['R2']:>8}")

print("\n✅ Done.")
