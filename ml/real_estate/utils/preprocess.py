import os
import numpy as np
import pandas as pd
import pymysql
from dotenv import load_dotenv, find_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


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

    protected_features = ["buyer_dominance_change", "kr_mortgage_rate_change", "house_price_idx_change"]

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


def preprocess_data(test_months=24, vif_threshold=5.0):
    # MySQL에서 직접 데이터 로드
    df = load_data_from_mysql()
    df = df.sort_values("date_ym").reset_index(drop=True)

    # 1. Target creation
    df["next_house_price_idx"] = df["house_price_idx"].shift(-1)
    TARGET = "next_change_rate"
    df[TARGET] = (df["next_house_price_idx"] - df["house_price_idx"]) / df["house_price_idx"] * 100

    df = df.dropna(subset=[TARGET]).copy()

    # 2. Impute missing values
    raw_features = [
        "house_price_idx", "kr_cpi", "kr_unemployment",
        "kr_base_rate", "kr_mortgage_rate", "kospi200",
        "apt_trade_count", "kr_m2", "buyer_dominance"
    ]
    df[raw_features] = df[raw_features].ffill().bfill()

    # Feature Engineering
    df["house_price_idx_change"] = df["house_price_idx"].pct_change() * 100
    df["kr_cpi_change"] = df["kr_cpi"].pct_change() * 100
    df["kr_unemployment_change"] = df["kr_unemployment"].diff()
    df["kr_base_rate_change"] = df["kr_base_rate"].diff()
    df["kr_mortgage_rate_change"] = df["kr_mortgage_rate"].diff()
    df["kospi200_change"] = df["kospi200"].pct_change() * 100
    df["apt_trade_count_change"] = df["apt_trade_count"].pct_change() * 100
    df["kr_m2_change"] = df["kr_m2"].pct_change() * 100
    df["buyer_dominance_change"] = df["buyer_dominance"].diff()

    # Seasonality Features
    month_series = pd.to_datetime(df['date_ym'], format='%Y%m').dt.month
    df['month_sin'] = np.sin(2 * np.pi * month_series / 12)
    df['month_cos'] = np.cos(2 * np.pi * month_series / 12)
    seasonality_features = ['month_sin', 'month_cos']

    # Lags
    stationary_cols = [
        "house_price_idx_change", "kr_cpi_change", "kr_unemployment_change",
        "kr_mortgage_rate_change", "kospi200_change", "buyer_dominance_change",
        "apt_trade_count_change"
    ]

    lagged_features = []
    for col in stationary_cols:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag2"] = df[col].shift(2)
        df[f"{col}_lag3"] = df[col].shift(3)
        lagged_features.extend([f"{col}_lag1", f"{col}_lag2", f"{col}_lag3"])

    # Moving Averages
    rolling_features = []
    for col in ["house_price_idx_change", "buyer_dominance_change", "apt_trade_count_change", "kr_mortgage_rate_change"]:
        df[f"{col}_ma3"] = df[col].rolling(window=3).mean()
        df[f"{col}_ma6"] = df[col].rolling(window=6).mean()
        rolling_features.extend([f"{col}_ma3", f"{col}_ma6"])

    candidate_features = [
        "house_price_idx_change", "kr_cpi_change", "kr_unemployment_change",
        "kr_base_rate_change", "kr_mortgage_rate_change", "kospi200_change",
        "apt_trade_count_change", "kr_m2_change", "buyer_dominance_change"
    ] + lagged_features + rolling_features + seasonality_features

    # Train/Test split
    train_df = df.iloc[:-test_months].copy()
    test_df = df.iloc[-test_months:].copy()

    train_df[candidate_features] = train_df[candidate_features].ffill().bfill()
    test_df[candidate_features] = test_df[candidate_features].ffill().bfill()

    print("=" * 55)
    print("Data Preprocessing & Train/Test Splitting")
    print("=" * 55)
    print(f"  Total samples     : {len(df)}")
    print(f"  Train period      : {train_df['date_ym'].min()} ~ {train_df['date_ym'].max()} ({len(train_df)} months)")
    print(f"  Test period       : {test_df['date_ym'].min()} ~ {test_df['date_ym'].max()} ({len(test_df)} months)")

    selected_features = filter_features_by_vif(train_df, candidate_features, threshold=vif_threshold, max_features=6)

    X_train = train_df[selected_features]
    y_train = train_df[TARGET]
    X_test = test_df[selected_features]
    y_test = test_df[TARGET]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # -------------------------------------------------------------
    # ml_realestate_preprocessed 테이블 동적 생성 및 업로드 추가
    # -------------------------------------------------------------
    import pymysql
    
    # numeric_cols 반올림
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(6)

    load_dotenv(find_dotenv())
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    DB_NAME = os.getenv('DB_NAME')

    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
        print("   [Warning] Missing DB configuration in .env. Skipping database export.")
    else:
        DB_PORT = int(DB_PORT)
        try:
            connection = pymysql.connect(
                host=DB_HOST,
                user=DB_USER,
                password=DB_PASSWORD,
                database=DB_NAME,
                port=DB_PORT,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            print("   [RealEstate] DB Connection successful!")
            
            with connection.cursor() as cursor:
                table_name = "ml_realestate_preprocessed"
                
                # Drop existing table
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                
                # Dynamic CREATE TABLE based on df columns
                columns_def = []
                final_order = []
                for col in df.columns:
                    final_order.append(col)
                    if col == 'date_ym':
                        columns_def.append("date_ym VARCHAR(10) PRIMARY KEY")
                    else:
                        columns_def.append(f"`{col}` DOUBLE")
                
                create_table_sql = f"CREATE TABLE {table_name} ({', '.join(columns_def)}) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;"
                cursor.execute(create_table_sql)
                print(f"   Created dynamic table '{table_name}' with {len(df.columns)} columns.")

                # Batch INSERT preprocessed DataFrame rows (NaN to None for NULL binding)
                df_ordered = df[final_order]
                db_data = df_ordered.replace({np.nan: None}).values.tolist()
                
                placeholders = ", ".join(["%s"] * len(final_order))
                col_names_quoted = ", ".join([f"`{c}`" for c in final_order])
                
                insert_sql = f"INSERT INTO {table_name} ({col_names_quoted}) VALUES ({placeholders})"
                
                cursor.executemany(insert_sql, db_data)
                connection.commit()
                print(f"   Successfully uploaded {len(db_data)} preprocessed rows into MySQL table '{table_name}'!")
                
            connection.close()
        except Exception as e:
            print(f"   [Error] MySQL Export failed: {e}")

    preprocessed_data = {
        'df': df,
        'train_df': train_df,
        'test_df': test_df,
        'X_train_sc': X_train_sc,
        'X_test_sc': X_test_sc,
        'y_train': y_train,
        'y_test': y_test,
        'features': selected_features,
        'scaler': scaler
    }

    return preprocessed_data


if __name__ == '__main__':
    preprocess_data()
