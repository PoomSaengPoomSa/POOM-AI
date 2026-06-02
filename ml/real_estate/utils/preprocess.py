import os
import pandas as pd
import numpy as np
import pymysql
import sys
from dotenv import load_dotenv, find_dotenv

N_FEATURES = 6


def load_raw_data_from_mysql():
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
    
    if 'rr_id' in df.columns:
        df = df.drop(columns=['rr_id'])
    if 'loaded_date' in df.columns:
        df = df.drop(columns=['loaded_date'])
        
    cols = ['date_ym'] + [col for col in df.columns if col != 'date_ym']
    df = df[cols]
    
    numeric_cols = [col for col in df.columns if col != 'date_ym']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    return df


def prune_features_by_vif(df, features, threshold=10.0):
    current_features = list(features)
    print(f"   [VIF Pruning] Starting VIF pruning on {len(current_features)} candidate features (threshold={threshold})...")
    
    from sklearn.linear_model import LinearRegression
    
    while True:
        # If we have 10 features left, let's stop pruning to allow RF to choose from a decent pool of clean features.
        if len(current_features) <= 10:
            break
            
        vifs = {}
        for feature in current_features:
            other_features = [f for f in current_features if f != feature]
            X = df[other_features].values
            y = df[feature].values
            
            reg = LinearRegression().fit(X, y)
            r2 = reg.score(X, y)
            
            vif = 1.0 / (1.0 - r2) if r2 < 1.0 else float('inf')
            vifs[feature] = vif
            
        vif_series = pd.Series(vifs)
        max_vif = vif_series.max()
        max_feature = vif_series.idxmax()
        
        if max_vif > threshold:
            print(f"      - VIF Pruning: Dropping '{max_feature}' with VIF = {max_vif:.2f}")
            current_features.remove(max_feature)
        else:
            break
            
    print(f"   [VIF Pruning] Completed. Remaining features: {len(current_features)}")
    return current_features



def preprocess():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')

    # -----------------------------------------
    # 1. Load Raw Data (MySQL Database)
    # -----------------------------------------
    print("Loading raw data from MySQL...")
    df = load_raw_data_from_mysql()
    print(f"   Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"   Period: {df['date_ym'].min()} ~ {df['date_ym'].max()}")

    df['date_ym'] = df['date_ym'].astype(str).str.strip()
    df = df.sort_values('date_ym').reset_index(drop=True)

    # -----------------------------------------
    # 1-1. Target creation (Before Imputation to drop fake future rows)
    # -----------------------------------------
    df["next_house_price_idx"] = df["house_price_idx"].shift(-1)
    df["next_change_rate"] = (df["next_house_price_idx"] - df["house_price_idx"]) / df["house_price_idx"] * 100
    df = df.dropna(subset=["next_change_rate"]).copy()
    df = df.reset_index(drop=True)

    # -----------------------------------------
    # 2. Impute Missing Values (IterativeImputer MICE)
    # -----------------------------------------
    print("\nImputing missing values using IterativeImputer (MICE)...")
    before_na = df.isna().sum().sum()

    # Extract numeric columns for imputation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if df[numeric_cols].isna().sum().sum() > 0:
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        from sklearn.linear_model import BayesianRidge
        
        imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    df = df.ffill().bfill()
    after_na = df.isna().sum().sum()
    print(f"   Missing values: {before_na} -> {after_na}")

    # -----------------------------------------
    # 3. Feature Engineering
    # -----------------------------------------
    print("\nGenerating derived features...")

    # -- Target: Next Month's Change Rate is already calculated --

    # -- 3-1) Month-on-Month Change Rate / Diff --
    df["house_price_idx_change"] = df["house_price_idx"].pct_change() * 100
    df["kr_cpi_change"] = df["kr_cpi"].pct_change() * 100
    df["kr_unemployment_change"] = df["kr_unemployment"].diff()
    df["kr_base_rate_change"] = df["kr_base_rate"].diff()
    df["kr_mortgage_rate_change"] = df["kr_mortgage_rate"].diff()
    df["kospi200_change"] = df["kospi200"].pct_change() * 100
    df["apt_trade_count_change"] = df["apt_trade_count"].pct_change() * 100
    df["kr_m2_change"] = df["kr_m2"].pct_change() * 100
    df["buyer_dominance_change"] = df["buyer_dominance"].diff()

    # -- 3-2) Seasonality Features --
    month_series = pd.to_datetime(df['date_ym'], format='%Y%m').dt.month
    df['month_sin'] = np.sin(2 * np.pi * month_series / 12)
    df['month_cos'] = np.cos(2 * np.pi * month_series / 12)
    seasonality_features = ['month_sin', 'month_cos']

    # -- 3-3) Lag Variables (1-Month, 2-Month, 3-Month Lag) --
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

    # -- 3-4) Moving Averages (3-Month, 6-Month) --
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

    print(f"   Total columns including derived features: {len(df.columns)}")

    # -----------------------------------------
    # 4. Cleanup NaN Rows (Due to rolling/shift)
    # -----------------------------------------
    # Drop initial rolling windows
    first_valid = 6  # 6-month window for rolling ma6
    df = df.iloc[first_valid:].reset_index(drop=True)
    df = df.ffill().bfill()

    # -----------------------------------------
    # 5. Feature Selection (Random Forest Regressor Voting)
    # -----------------------------------------
    print("\n[Feature Selection] Running Random Forest Regressor Feature Selection...")
    
    sys.path.insert(0, base_dir)
    from model import RealEstateEnsembleRegressor
    from sklearn.ensemble import RandomForestRegressor
    import joblib
    
    cfg = RealEstateEnsembleRegressor
    
    # 1) Slice Train Dataset down to ~cfg.TRAIN_END to avoid Data Leakage
    train_df = df[df['date_ym'] <= cfg.TRAIN_END].copy()
    
    drop_cols = [c for c in cfg.DROP_COLS if c in train_df.columns]
    X_train_all = train_df.drop(columns=drop_cols)
    y_train_target = train_df['next_change_rate']
    
    # 2) Random Forest Regressor Feature Selection using Train set
    # Run VIF pruning on candidate features first!
    vif_pruned_features = prune_features_by_vif(X_train_all, X_train_all.columns.tolist(), threshold=10.0)
    
    print(f"   Selecting top {N_FEATURES} features using 20 Random Forest ensembles from {len(vif_pruned_features)} VIF-pruned features...")
    X_train_vif = X_train_all[vif_pruned_features]
    importances_list = []
    for i in range(20):
        rf = RandomForestRegressor(
            n_estimators=100,
            random_state=i,
            max_depth=5,
            max_features=0.8,
            n_jobs=-1
        )
        rf.fit(X_train_vif, y_train_target)
        importances_list.append(rf.feature_importances_)
        
    avg_importances = np.mean(importances_list, axis=0)
    
    feat_imp = pd.DataFrame({
        'feature': vif_pruned_features,
        'importance': avg_importances
    }).sort_values('importance', ascending=False)
    
    selected_features = feat_imp.head(N_FEATURES)['feature'].tolist()
    
    print(f"   Selected {N_FEATURES} Features:")
    for i, r in feat_imp.head(N_FEATURES).iterrows():
        print(f"     - {r['feature']}: {r['importance']:.4f}")
        
    # 3) Dump selected features list to pkl for downstream scripts
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(selected_features, os.path.join(models_dir, 'selected_features.pkl'))
    print(f"   Feature names saved to: models/selected_features.pkl")

    # -----------------------------------------
    # 6. Final Reordering (Slice to Selected Features + Targets Only)
    # -----------------------------------------
    id_cols = ['date_ym']
    target_cols = ['next_change_rate']
    final_order = id_cols + selected_features + target_cols
    df = df[final_order]

    # Save to CSV local cache
    save_path = os.path.join(data_dir, 'final_dataset.csv')
    df.to_csv(save_path, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 55)
    print("Preprocessing & Feature Selection completed successfully!")
    print("=" * 55)
    print(f"   Save Path   : {save_path}")
    print(f"   Final Size  : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"   Train Period: {df[df['date_ym'] <= cfg.TRAIN_END].shape[0]} months (~{cfg.TRAIN_END})")
    print(f"   Test Period : {df[df['date_ym'] >= cfg.TEST_START].shape[0]} months ({cfg.TEST_START}~)")

    # -----------------------------------------
    # 7. Load Preprocessed Data into MySQL Database
    # -----------------------------------------
    print("\n[Database Export] Loading preprocessed data into MySQL...")
    
    # 1) Round all numeric values to 4 decimal places
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(4)
    print("   All numeric columns rounded to 4 decimal places.")

    # 2) DB connection
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
            print("   DB Connection successful!")
            
            with connection.cursor() as cursor:
                table_name = "ml_realestate_preprocessed"
                
                # Drop existing table
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                
                # CREATE TABLE dynamically
                columns_def = ["date_ym VARCHAR(10) PRIMARY KEY"]
                for col in selected_features:
                    columns_def.append(f"`{col}` DECIMAL(15, 4)")
                columns_def.append("next_change_rate DECIMAL(15, 4)")
                
                create_table_sql = f"CREATE TABLE {table_name} ({', '.join(columns_def)}) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;"
                cursor.execute(create_table_sql)
                print(f"   Created dynamic table '{table_name}' with {len(selected_features)} features.")

                # Batch INSERT
                db_data = df.replace({np.nan: None}).values.tolist()
                all_cols = ['date_ym'] + selected_features + ['next_change_rate']
                placeholders = ", ".join(["%s"] * len(all_cols))
                col_names_quoted = ", ".join([f"`{c}`" for c in all_cols])
                
                insert_sql = f"INSERT INTO {table_name} ({col_names_quoted}) VALUES ({placeholders})"
                
                cursor.executemany(insert_sql, db_data)
                connection.commit()
                print(f"   Successfully uploaded {len(db_data)} preprocessed rows into MySQL table '{table_name}'!")
                
            connection.close()
        except Exception as e:
            print(f"   [Error] MySQL Export failed: {e}")

    # Data Preview
    print(f"\n   Data Preview (last 5 rows):")
    pd.set_option('display.max_columns', 10)
    print(df.tail().to_string(index=False))


if __name__ == '__main__':
    preprocess()
