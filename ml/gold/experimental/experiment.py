import sys
import io
# Windows 콘솔 한글 깨짐 및 인코딩 오류 방지 설정
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

import os
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def load_raw_data():
    """
    MySQL 데이터베이스에서 ml_gold_raw 테이블을 날짜 순으로 전량 로드합니다.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(dotenv_path=os.path.join(base_dir, '../../.env'))

    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT", "3306")
    db_name = os.getenv("DB_NAME")

    if not all([db_user, db_password, db_host, db_name]):
        raise ValueError(".env 파일의 DB 연결 환경변수가 비어있습니다!")

    db_url = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    engine = create_engine(db_url)
    
    query = "SELECT * FROM ml_gold_raw ORDER BY loaded_date ASC"
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
    
    return df

def preprocess_data_v2(df):
    """
    기존 피처(40차원) + 신규 고급 파생변수 생성 및 피처 선택을 수행합니다.
    """
    df = df.copy()
    
    # 1. 시계열 순차 정렬
    df = df.sort_values('loaded_date').reset_index(drop=True)
    
    # 2. 금값이 NaN인 날(주말/공휴일) 드롭
    df = df.dropna(subset=['gold']).reset_index(drop=True)
    
    # 3. 타 거시경제 지표 결측치 보정 (Forward fill & Backward fill)
    feature_cols = ['kr_usd_exchange', 'wti_oil', 'dxy_proxy', 'vix', 'kospi200', 'sp500', 'kr_cpi']
    df[feature_cols] = df[feature_cols].ffill().bfill()
    
    # 4. 일별 지표들의 변화율 생성
    change_rate_cols = []
    for col in ['gold', 'kr_usd_exchange', 'wti_oil', 'dxy_proxy', 'vix', 'kospi200', 'sp500']:
        col_name = f"{col}_change_rate"
        df[col_name] = df[col].pct_change()
        change_rate_cols.append(col_name)
    
    # 5. [NEW 파생변수] 금융 공학 기술 지표 및 스프레드 지표 설계
    # A. RSI (Relative Strength Index) - 14일 기준
    delta = df['gold_change_rate']
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['gold_rsi_14'] = 100 - (100 / (1 + rs))
    
    # B. MACD (Moving Average Convergence Divergence)
    ema12 = df['gold_change_rate'].ewm(span=12, adjust=False).mean()
    ema26 = df['gold_change_rate'].ewm(span=26, adjust=False).mean()
    df['gold_macd'] = ema12 - ema26
    df['gold_macd_signal'] = df['gold_macd'].ewm(span=9, adjust=False).mean()
    
    # C. EMA (Exponential Moving Average) - 5일/20일 기준
    df['gold_change_rate_ema_5'] = df['gold_change_rate'].ewm(span=5, adjust=False).mean()
    df['gold_change_rate_ema_20'] = df['gold_change_rate'].ewm(span=20, adjust=False).mean()
    
    # D. 미국-한국 주식 스프레드 (SP500 vs KOSPI200 스프레드)
    df['sp500_kospi200_spread'] = df['sp500_change_rate'] - df['kospi200_change_rate']
    
    # E. 금-달러 상호작용 피처 (역상관 강도 측정)
    df['gold_dxy_interaction'] = df['gold_change_rate'] * df['dxy_proxy_change_rate']
    
    new_derived_cols = [
        'gold_rsi_14', 'gold_macd', 'gold_macd_signal', 
        'gold_change_rate_ema_5', 'gold_change_rate_ema_20',
        'sp500_kospi200_spread', 'gold_dxy_interaction'
    ]
    
    # 6. 기존 시차 피처 생성 (1, 2, 3 영업일 시차 수익률)
    lag_features = []
    # 기존 변화율 지표들의 lags
    for col in change_rate_cols:
        for lag in [1, 2, 3]:
            lag_name = f"{col}_lag_{lag}"
            df[lag_name] = df[col].shift(lag)
            lag_features.append(lag_name)
            
    # 신규 파생변수들의 lags 추가생성 (Look-ahead bias 방지 및 최신정보 피팅)
    for col in new_derived_cols:
        for lag in [1, 2, 3]:
            lag_name = f"{col}_lag_{lag}"
            df[lag_name] = df[col].shift(lag)
            lag_features.append(lag_name)
            
    # 7. 기술 지표 생성 (5일 및 20일 이동평균, 변동성)
    df['gold_change_rate_sma_5'] = df['gold_change_rate'].rolling(5).mean()
    df['gold_change_rate_sma_20'] = df['gold_change_rate'].rolling(20).mean()
    df['gold_change_rate_std_5'] = df['gold_change_rate'].rolling(5).std()
    df['gold_change_rate_std_20'] = df['gold_change_rate'].rolling(20).std()
    
    rolling_features = [ 
        'gold_change_rate_sma_5', 'gold_change_rate_sma_20',
        'gold_change_rate_std_5', 'gold_change_rate_std_20'
    ]
    
    # 8. 예측용 타겟 변수 생성 (하루 뒤로 시프트하여 미래 정보 누수 방지)
    df['target_tomorrow_gold_change_rate'] = df['gold_change_rate'].shift(-1)
    df['target_tomorrow_gold_direction'] = (df['target_tomorrow_gold_change_rate'] > 0).astype(int)
    
    # Lags, Rolling, Shift로 인해 발생하는 결측치 행 정리
    df = df.dropna().reset_index(drop=True)
    
    # 학습용 피처 목록 정의 (고도화 피처셋)
    base_features = [
        'gold', 'gold_change_rate', 'kr_cpi', 'kr_usd_exchange', 'wti_oil', 
        'dxy_proxy', 'vix', 'kospi200', 'sp500', 
        'kr_usd_exchange_change_rate', 'wti_oil_change_rate', 'dxy_proxy_change_rate', 
        'vix_change_rate', 'kospi200_change_rate', 'sp500_change_rate'
    ]
    
    # 총 피처 목록 조립
    features = base_features + new_derived_cols + lag_features + rolling_features
    
    X = df[features]
    y_cls = df['target_tomorrow_gold_direction']
    dates = df['loaded_date']
    
    return X, y_cls, dates

def run_experiments():
    print("="*60)
    print("[실험 개시] 데이터 로딩 및 전처리 (파생변수 주입 버전)")
    print("="*60)
    raw_df = load_raw_data()
    X, y_cls, dates = preprocess_data_v2(raw_df)
    print(f"   [OK] 데이터 셋 형상: {X.shape}")
    
    # 피처 목록 확인 및 저장
    feature_list = X.columns.tolist()
    
    # 8:2 시계열 연대기 순 분할
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y_cls.iloc[:split_idx], y_cls.iloc[split_idx:]
    dates_train, dates_test = dates.iloc[:split_idx], dates.iloc[split_idx:]
    
    # 피처 표준화 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_list)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_list)
    
    # 클래스 가중치 계산
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_val = float(num_neg) / float(num_pos)
    print(f"   [샘플 비율] 하락(0): {num_neg} | 상승(1): {num_pos}")
    print(f"   [가중치 계산] scale_pos_weight: {scale_pos_val:.4f}\n")
    
    # 실험할 모델 및 하이퍼파라미터 세팅
    models = {
        "1. Random Forest (Baseline)": RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        "2. Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.015,
            max_depth=3,
            subsample=0.8,
            random_state=42
        ),
        "3. XGBoost (Tuned V2)": xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.012,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.7,
            scale_pos_weight=scale_pos_val,
            reg_alpha=1.2,
            reg_lambda=3.0,
            random_state=42,
            eval_metric='logloss'
        ),
        "4. LightGBM (Tuned)": lgb.LGBMClassifier(
            n_estimators=180,
            learning_rate=0.012,
            max_depth=4,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.7,
            scale_pos_weight=scale_pos_val,
            reg_alpha=1.0,
            reg_lambda=2.5,
            random_state=42,
            verbosity=-1,
            n_jobs=-1
        )
    }
    
    results = []
    
    print("="*60)
    print("[모델 훈련 및 교차 검증]")
    print("="*60)
    
    for name, model in models.items():
        print(f"👉 모델 학습 중: {name}...")
        model.fit(X_train_scaled_df, y_train)
        y_pred = model.predict(X_test_scaled_df)
        
        # 성능 지표 도출
        acc = accuracy_score(y_test, y_pred) * 100
        f1_cls0 = f1_score(y_test, y_pred, pos_label=0)
        f1_cls1 = f1_score(y_test, y_pred, pos_label=1)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        conf = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = conf.ravel()
        
        results.append({
            "Model": name,
            "Accuracy (Hit Rate)": f"{acc:.2f}%",
            "Class 0 (Down) F1": f"{f1_cls0:.4f}",
            "Class 1 (Up) F1": f"{f1_cls1:.4f}",
            "Macro F1-Score": f"{f1_macro:.4f}",
            "TN": tn, "FP": fp, "FN": fn, "TP": tp
        })
        
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("[실험 결과 비교 대조표]")
    print("="*60)
    print(results_df.to_string(index=False))
    print("="*60)
    
    # 가장 높은 Macro F1-Score를 가진 최적 모델 선정
    results_df['Macro_F1_val'] = results_df['Macro F1-Score'].astype(float)
    best_row = results_df.loc[results_df['Macro_F1_val'].idxmax()]
    print(f"\n[최적 모델 선정 완료]")
    print(f"   선정 모델: {best_row['Model']}")
    print(f"   종합 Macro F1-Score: {best_row['Macro F1-Score']}")
    print(f"   예측 정확도 (Hit Rate): {best_row['Accuracy (Hit Rate)']}")
    print(f"   오차행렬 - TN: {best_row['TN']} (하락 예측 성공) | TP: {best_row['TP']} (상승 예측 성공)\n")

if __name__ == '__main__':
    run_experiments()
