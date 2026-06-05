import os
import sys
import numpy as np
import pandas as pd
import pymysql
from dotenv import load_dotenv, find_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# utils/ 하위에서 직접 실행 시 상위 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



def load_data_from_mysql():
    load_dotenv(find_dotenv())

    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    DB_NAME = os.getenv('DB_NAME')

    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
        raise ValueError("Missing database credentials in .env file.")

    DB_PORT = int(DB_PORT)

    connection = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

    try:
        with connection.cursor() as cursor:
            sql = "SELECT * FROM ml_realestate_raw ORDER BY loaded_date ASC"
            cursor.execute(sql)
            rows = cursor.fetchall()
    finally:
        connection.close()

    df = pd.DataFrame(rows)
    df['date_ym'] = pd.to_datetime(df['loaded_date']).dt.strftime('%Y%m')
    df = df.drop(columns=[c for c in ['rr_id', 'loaded_date'] if c in df.columns])

    numeric_cols = [col for col in df.columns if col != 'date_ym']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"[DB] Loaded data successfully from MySQL table 'ml_realestate_raw': {len(df)} rows")
    return df


def calculate_vif_custom(df, features):
    vif_dict = {}
    for feature in features:
        other_features = [f for f in features if f != feature]
        if not other_features:
            vif_dict[feature] = 1.0
            continue

        X = df[other_features].values
        y = df[feature].values

        reg = LinearRegression().fit(X, y)
        r2 = reg.score(X, y)

        if r2 >= 1.0:
            vif = float('inf')
        else:
            vif = 1.0 / (1.0 - r2)
        vif_dict[feature] = vif

    return pd.Series(vif_dict)


def filter_features_by_vif(df, features, threshold=5.0, max_features=6):
    current_features = list(features)
    print("  [Aggressive VIF Feature Pruning for Production]")

    protected_features = ["buyer_dominance_change", "kr_mortgage_rate_change"]

    while True:
        if len(current_features) <= 4:
            break

        vif_series = calculate_vif_custom(df, current_features)

        candidates = vif_series.drop(labels=[f for f in protected_features if f in vif_series.index])
        if candidates.empty:
            break

        max_vif = candidates.max()
        max_feature = candidates.idxmax()

        if max_vif > threshold or len(current_features) > max_features:
            print(f"    - Dropping '{max_feature}' with VIF = {max_vif:.4f}")
            current_features.remove(max_feature)
        else:
            break

    print(f"  Final selected features ({len(current_features)}): {current_features}")

    final_vifs = calculate_vif_custom(df, current_features)
    for feat, v in final_vifs.items():
        print(f"    * {feat:<25}: VIF = {v:.4f}")

    return current_features


def preprocess_data(vif_threshold=5.0, valid_mode=False):
    # MySQL에서 직접 데이터 로드
    df = load_data_from_mysql()
    df = df.dropna(subset=["house_price_idx"]).copy()
    df = df.sort_values("date_ym").reset_index(drop=True)

    # 1. Target creation
    df["next_house_price_idx"] = df["house_price_idx"].shift(-1)
    TARGET = "next_change_rate"
    df[TARGET] = (df["next_house_price_idx"] - df["house_price_idx"]) / df["house_price_idx"] * 100

    # 2. Impute missing values
    raw_features = [
        "house_price_idx", "kr_cpi", "kr_unemployment",
        "kr_base_rate", "kr_mortgage_rate", "kospi200",
        "apt_trade_count", "kr_m2", "buyer_dominance"
    ]
    df[raw_features] = df[raw_features].ffill().bfill()

    # 3. 파생 변수 엔지니어링 (변수 특성에 맞게 pct_change / diff 구분)
    # 절댓값 스케일이 큰 변수 → 변화율(pct_change)
    rate_cols = ["house_price_idx", "kr_cpi", "kospi200", "apt_trade_count", "kr_m2"]
    # 금리·지수처럼 이미 % 단위인 변수 → 차분(diff)
    diff_cols  = ["kr_unemployment", "kr_base_rate", "kr_mortgage_rate", "buyer_dominance"]

    stationary_cols = []
    for col in rate_cols:
        df[f"{col}_change"] = df[col].pct_change() * 100
        stationary_cols.append(f"{col}_change")
    for col in diff_cols:
        df[f"{col}_change"] = df[col].diff()
        stationary_cols.append(f"{col}_change")

    # Lag 변수 (1~3개월)
    lagged_features = []
    for col in stationary_cols:
        for lag in [1, 2, 3]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
            lagged_features.append(f"{col}_lag{lag}")

    # 이동평균 (3개월, 6개월)
    rolling_features = []
    for col in ["house_price_idx_change", "buyer_dominance_change",
                "apt_trade_count_change", "kr_mortgage_rate_change"]:
        if col in df.columns:
            df[f"{col}_ma3"] = df[col].rolling(window=3).mean()
            df[f"{col}_ma6"] = df[col].rolling(window=6).mean()
            rolling_features.extend([f"{col}_ma3", f"{col}_ma6"])

    # 계절성: sin/cos 인코딩 (주기 연속성 보존)
    month_series = pd.to_datetime(df['date_ym'], format='%Y%m').dt.month
    df['month_sin'] = np.sin(2 * np.pi * month_series / 12)
    df['month_cos'] = np.cos(2 * np.pi * month_series / 12)
    seasonality_features = ['month_sin', 'month_cos']

    candidate_features = stationary_cols + lagged_features + rolling_features + seasonality_features

    # candidate_features에 결측치(NaN)가 있는 앞부분 행들을 제거 (ffill/bfill 대신 dropna 적용)
    df = df.dropna(subset=candidate_features).reset_index(drop=True)

    # Train/Test split (Fixed Date Split)
    from model import RealEstateEnsembleRegressor as cfg
    df['date_ym'] = df['date_ym'].astype(str).str.strip()
    if valid_mode:
        train_df = df[df['date_ym'] <= cfg.TRAIN_END].copy()
        test_df  = df[df['date_ym'].between(cfg.VALID_START, cfg.VALID_END)].copy()
        eval_name = "Validation"
    else:
        train_df = df[df['date_ym'] <= cfg.VALID_END].copy()
        test_df  = df[df['date_ym'].between(cfg.TEST_START, cfg.TEST_END)].copy()
        eval_name = "Test"

    # 모델 학습/평가 세트 추출 시에만 TARGET 결측치를 드롭함
    train_clean = train_df.dropna(subset=[TARGET]).copy()
    test_clean  = test_df.dropna(subset=[TARGET]).copy()

    print("=" * 55)
    print(f"Data Preprocessing & Train/{eval_name} Splitting")
    print("=" * 55)
    print(f"  Total samples (raw)     : {len(df)}")
    print(f"  Train period (clean)    : {train_clean['date_ym'].min()} ~ {train_clean['date_ym'].max()} ({len(train_clean)} months)")
    print(f"  {eval_name} period (clean)     : {test_clean['date_ym'].min()} ~ {test_clean['date_ym'].max()} ({len(test_clean)} months)")

    selected_features = filter_features_by_vif(train_clean, candidate_features, threshold=vif_threshold, max_features=6)

    X_train = train_clean[selected_features]
    y_train = train_clean[TARGET]
    X_test = test_clean[selected_features]
    y_test = test_clean[TARGET]

    # 트리 기반 앙상블 모델은 피처 스케일링이 필요 없으므로 StandardScaler를 비활성화함
    X_train_sc = X_train.values
    X_test_sc = X_test.values

    preprocessed_data = {
        'df': df,
        'train_df': train_clean,
        'test_df': test_clean,
        'X_train_sc': X_train_sc,
        'X_test_sc': X_test_sc,
        'y_train': y_train,
        'y_test': y_test,
        'features': selected_features,
        'scaler': None
    }

    # -----------------------------------------
    # DB 적재: ml_realestate_preprocessed (gold/base_rate와 동일한 방식)
    # -----------------------------------------
    all_feature_cols = ['house_price_idx'] + candidate_features
    final_order = ['date_ym'] + all_feature_cols + [TARGET]

    # 저장할 컬럼만 추려서 결측치 처리
    export_df = df[final_order].copy()
    numeric_cols = export_df.select_dtypes(include=[np.number]).columns
    export_df[numeric_cols] = export_df[numeric_cols].round(6)

    print("\n[Database Export] Loading preprocessed real estate data into MySQL...")

    load_dotenv(find_dotenv())
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    DB_NAME = os.getenv('DB_NAME')

    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
        print("   [Warning] Missing DB configuration. Skipping database export.")
    else:
        try:
            connection = pymysql.connect(
                host=DB_HOST,
                user=DB_USER,
                password=DB_PASSWORD,
                database=DB_NAME,
                port=int(DB_PORT),
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            print("   DB Connection successful!")

            with connection.cursor() as cursor:
                table_name = "ml_realestate_preprocessed"

                # DROP & CREATE (항상 최신 피처 구조로 갱신)
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

                columns_def = ["date_ym VARCHAR(10) PRIMARY KEY"]
                for col in all_feature_cols:
                    columns_def.append(f"`{col}` DECIMAL(15, 6)")
                columns_def.append(f"`{TARGET}` DECIMAL(15, 6)")

                create_sql = f"CREATE TABLE {table_name} ({', '.join(columns_def)}) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;"
                cursor.execute(create_sql)
                print(f"   Created table '{table_name}' with {len(all_feature_cols)} feature columns.")

                # Batch INSERT
                db_data = export_df.replace({np.nan: None}).values.tolist()
                placeholders = ", ".join(["%s"] * len(final_order))
                col_names_quoted = ", ".join([f"`{c}`" for c in final_order])
                insert_sql = f"INSERT INTO {table_name} ({col_names_quoted}) VALUES ({placeholders})"

                cursor.executemany(insert_sql, db_data)
                connection.commit()
                print(f"   Successfully uploaded {len(db_data)} rows into '{table_name}'!")

            connection.close()
        except Exception as e:
            print(f"   [Error] MySQL Export failed: {e}")

    return preprocessed_data



if __name__ == '__main__':
    print("=" * 55)
    print("[Preprocess] 부동산 전처리 검증 실행")
    print("=" * 55)
    result = preprocess_data(vif_threshold=20.0)
    if result:
        print(f"\n[Preprocess] 완료!")
        print(f"  Train : {result['train_df']['date_ym'].min()} ~ {result['train_df']['date_ym'].max()} ({len(result['X_train_sc'])} rows)")
        print(f"  Test  : {result['test_df']['date_ym'].min()} ~ {result['test_df']['date_ym'].max()} ({len(result['X_test_sc'])} rows)")
        print(f"  Features ({len(result['features'])}): {result['features']}")
    else:
        print("[Preprocess] 전처리 실패")

