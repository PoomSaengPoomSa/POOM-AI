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

NUM_FEATURES = 7



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


def filter_features_by_vif(df, features, threshold=10.0):
    current_features = list(features)
    print("=" * 55)
    print(f"  [VIF Feature Pruning] Threshold = {threshold}")
    print("=" * 55)
    
    protected_features = ["buyer_dominance_change", "kr_mortgage_rate_change"]

    while True:
        if len(current_features) <= NUM_FEATURES:
            break

        vif_series = calculate_vif_custom(df, current_features)
        
        candidates = vif_series.drop(labels=[f for f in protected_features if f in vif_series.index], errors='ignore')
        if candidates.empty:
            break

        max_vif = candidates.max()
        max_feature = candidates.idxmax()

        if max_vif > threshold:
            print(f"    - Dropping '{max_feature}' with VIF = {max_vif:.4f}")
            current_features.remove(max_feature)
        else:
            break

    print(f"  Final features remaining after VIF pruning ({len(current_features)}): {current_features}")
    return current_features


def preprocess():
    import pickle
    
    # 1. MySQL에서 직접 데이터 로드
    df = load_data_from_mysql()
    df = df.dropna(subset=["house_price_idx"]).copy()
    df = df.sort_values("date_ym").reset_index(drop=True)

    # Target creation
    df["next_house_price_idx"] = df["house_price_idx"].shift(-1)
    TARGET = "next_change_rate"
    df[TARGET] = (df["next_house_price_idx"] - df["house_price_idx"]) / df["house_price_idx"] * 100

    # Impute missing values
    raw_features = [
        "house_price_idx", "kr_cpi", "kr_unemployment",
        "kr_base_rate", "kr_mortgage_rate", "kospi200",
        "apt_trade_count", "kr_m2", "buyer_dominance"
    ]
    df[raw_features] = df[raw_features].ffill().bfill()

    # 파생 변수 엔지니어링 (변수 특성에 맞게 pct_change / diff 구분)
    rate_cols = ["house_price_idx", "kr_cpi", "kospi200", "apt_trade_count", "kr_m2"]
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

    # 계절성: sin/cos 인코딩
    month_series = pd.to_datetime(df['date_ym'], format='%Y%m').dt.month
    df['month_sin'] = np.sin(2 * np.pi * month_series / 12)
    df['month_cos'] = np.cos(2 * np.pi * month_series / 12)
    seasonality_features = ['month_sin', 'month_cos']

    candidate_features = stationary_cols + lagged_features + rolling_features + seasonality_features

    # candidate_features에 결측치(NaN)가 있는 앞부분 행들을 제거
    df = df.dropna(subset=candidate_features).reset_index(drop=True)

    # Train/Test split 기준으로 피처 선택 (Lookahead Bias 차단)
    from model import RealEstateEnsembleRegressor as cfg
    df['date_ym'] = df['date_ym'].astype(str).str.strip()
    
    # 피처 선택용 Train 데이터 분리 (TRAIN_END인 202403 기준)
    train_clean = df[df['date_ym'] <= cfg.TRAIN_END].dropna(subset=[TARGET]).copy()

    # 1) VIF Pruning (Threshold = 10.0)
    vif_filtered_features = filter_features_by_vif(train_clean, candidate_features, threshold=10.0)

    # 2) RF Feature Selection from the VIF-filtered features
    print("=" * 55)
    print(f"Static Random Forest Feature Selection (on VIF-filtered pool)")
    print("=" * 55)
    print(f"  Total samples (raw)     : {len(df)}")
    print(f"  RF Fit Train period     : {train_clean['date_ym'].min()} ~ {train_clean['date_ym'].max()} ({len(train_clean)} months)")

    from sklearn.ensemble import RandomForestRegressor
    X_train_all = train_clean[vif_filtered_features]
    y_train = train_clean[TARGET].values

    print(f"  Selecting top {NUM_FEATURES} features using 20 Random Forest Regressor ensembles from {X_train_all.shape[1]} features...")
    importances_list = []
    for i in range(20):
        rf = RandomForestRegressor(
            n_estimators=100,
            random_state=i,
            max_depth=5,
            n_jobs=-1
        )
        rf.fit(X_train_all, y_train)
        importances_list.append(rf.feature_importances_)
        
    avg_importances = np.mean(importances_list, axis=0)
    
    feat_imp = pd.DataFrame({
        'feature': vif_filtered_features,
        'importance': avg_importances
    }).sort_values('importance', ascending=False)
    
    selected_features = feat_imp.head(NUM_FEATURES)['feature'].tolist()

    print(f"  Selected {NUM_FEATURES} Features:")
    for idx, r in feat_imp.head(NUM_FEATURES).iterrows():
        print(f"    - {r['feature']}: importance = {r['importance']:.4f}")

    # -----------------------------------------
    # Save Selected Features List
    # -----------------------------------------
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    features_path = os.path.join(models_dir, 'selected_features.pkl')
    with open(features_path, 'wb') as f:
        pickle.dump(selected_features, f)
        
    txt_features_path = os.path.join(models_dir, 'selected_features.txt')
    with open(txt_features_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(selected_features))
        
    print(f"  Saved selected features list to: {features_path} and .txt")

    # -----------------------------------------
    # DB/CSV 적재 컬럼 추출
    # -----------------------------------------
    # house_price_idx는 최신 실제 지표 확인을 위해 반드시 내보내야 함
    all_cols = ['date_ym', 'house_price_idx'] + selected_features + [TARGET]
    df_export = df[all_cols].copy()

    # 소수점 6자리 반올림
    numeric_cols = df_export.select_dtypes(include=[np.number]).columns
    df_export[numeric_cols] = df_export[numeric_cols].round(6)

    # Save to local CSV
    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    save_path = os.path.join(data_dir, 'final_dataset.csv')
    df_export.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"  Saved preprocessed dataset to: {save_path}")

    # -----------------------------------------
    # DB 적재: ml_realestate_preprocessed
    # -----------------------------------------
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

                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

                columns_def = [
                    "date_ym VARCHAR(10) PRIMARY KEY",
                    "house_price_idx DECIMAL(15, 6)"
                ]
                for col in selected_features:
                    columns_def.append(f"`{col}` DECIMAL(15, 6)")
                columns_def.append(f"`{TARGET}` DECIMAL(15, 6)")

                create_sql = f"CREATE TABLE {table_name} ({', '.join(columns_def)}) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;"
                cursor.execute(create_sql)
                print(f"   Created table '{table_name}' with {len(selected_features)} selected feature columns.")

                # Batch INSERT
                db_data = df_export.replace({np.nan: None}).values.tolist()
                placeholders = ", ".join(["%s"] * len(all_cols))
                col_names_quoted = ", ".join([f"`{c}`" for c in all_cols])
                insert_sql = f"INSERT INTO {table_name} ({col_names_quoted}) VALUES ({placeholders})"

                cursor.executemany(insert_sql, db_data)
                connection.commit()
                print(f"   Successfully uploaded {len(db_data)} rows into '{table_name}'!")

            connection.close()
        except Exception as e:
            print(f"   [Error] MySQL Export failed: {e}")

    return df_export


if __name__ == '__main__':
    print("=" * 55)
    print("[Preprocess] 부동산 전처리 정적 실행")
    print("=" * 55)
    result = preprocess()
    if result is not None:
        print(f"\n[Preprocess] 완료! 최종 데이터셋 행 개수: {len(result)}")
        print(f"  Columns: {list(result.columns)}")
    else:
        print("[Preprocess] 전처리 실패")

