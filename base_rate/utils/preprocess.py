import os
import pandas as pd
import numpy as np
import pymysql
from dotenv import load_dotenv

N_FEATURES = 10


def load_raw_data_from_mysql():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(dotenv_path=os.path.join(base_dir, '.env'))
    
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
            sql = "SELECT * FROM ml_baserate_raw ORDER BY loaded_date ASC"
            cursor.execute(sql)
            rows = cursor.fetchall()
    finally:
        connection.close()
        
    df = pd.DataFrame(rows)
    df['date_ym'] = pd.to_datetime(df['loaded_date']).dt.strftime('%Y%m')
    
    column_mapping = {
        'kr_gdp': 'kr_gdp_index'
    }
    df = df.rename(columns=column_mapping)
    
    if 'br_id' in df.columns:
        df = df.drop(columns=['br_id'])
    if 'loaded_date' in df.columns:
        df = df.drop(columns=['loaded_date'])
        
    cols = ['date_ym'] + [col for col in df.columns if col != 'date_ym']
    df = df[cols]
    
    numeric_cols = [col for col in df.columns if col != 'date_ym']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    return df


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
    # 2. Impute Missing Values (IterativeImputer MICE)
    # -----------------------------------------
    print("\nImputing missing values using IterativeImputer (MICE)...")
    before_na = df.isna().sum().sum()

    if 'kr_gdp_index' in df.columns:
        df['kr_gdp_index'] = df['kr_gdp_index'].interpolate(method='linear')

    # Extract numeric columns for imputation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if df[numeric_cols].isna().sum().sum() > 0:
        # Use more stable IterativeImputer (MICE) from scikit-learn
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        from sklearn.linear_model import BayesianRidge
        
        imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    # Fill remaining NaNs for categorical/identifier columns if any
    df = df.ffill().bfill()
    
    after_na = df.isna().sum().sum()
    print(f"   Missing values: {before_na} -> {after_na}")

    # -----------------------------------------
    # 3. Feature Engineering
    # -----------------------------------------
    print("\nGenerating derived features...")

    # -- Target: MoM Base Rate Change --
    df['kr_base_rate_change'] = df['kr_base_rate'].diff()

    # -- 3-1) Month-on-Month Change/Diff --
    # Adjusted change_pairs to match the updated raw_data.csv (ml_baserate_raw table schema)
    change_pairs = [
        ('kr_cpi',          'kr_cpi_change',       'pct'),
        ('kr_m2',           'kr_m2_change',        'pct'),
        ('kr_usd_exchange', 'kr_exchange_change',  'pct'),
        ('us_fed_rate',     'us_fed_rate_change',  'diff'),
        ('wti_oil',         'wti_change',          'pct'),
        ('vix',             'vix_change',          'pct'),
    ]
    for src, dst, method in change_pairs:
        if src in df.columns:
            if method == 'pct':
                df[dst] = df[src].pct_change() * 100
            else:
                df[dst] = df[src].diff()

    # -- 3-2) YoY (Year-over-Year) Change --
    yoy_targets = ['kr_cpi', 'kr_usd_exchange', 'wti_oil']
    for col in yoy_targets:
        if col in df.columns:
            df[f'{col}_yoy'] = df[col].pct_change(periods=12) * 100

    # -- 3-3) KR-US Interest Rate Spread --
    if 'kr_base_rate' in df.columns and 'us_fed_rate' in df.columns:
        df['rate_spread'] = df['kr_base_rate'] - df['us_fed_rate']
        df['rate_spread_change'] = df['rate_spread'].diff()

    # -- 3-4) Moving Averages (3-Month, 6-Month) --
    ma_targets = [
        'kr_base_rate', 'kr_cpi', 'kr_unemployment',
        'kr_usd_exchange', 'us_fed_rate', 'vix', 'wti_oil'
    ]
    for col in ma_targets:
        if col in df.columns:
            df[f'{col}_ma3'] = df[col].rolling(window=3, min_periods=1).mean()
            df[f'{col}_ma6'] = df[col].rolling(window=6, min_periods=1).mean()

    # -- 3-5) Cumulative Change / Momentum (3-Month, 6-Month) --
    momentum_targets = ['kr_cpi', 'kr_usd_exchange', 'us_fed_rate',
                        'vix', 'wti_oil', 'kr_unemployment']
    for col in momentum_targets:
        if col in df.columns:
            df[f'{col}_mom3'] = df[col].diff(3)
            df[f'{col}_mom6'] = df[col].diff(6)

    # -- 3-6) Lag Variables (1-Month, 2-Month, 3-Month Lag) --
    lag_targets = ['kr_cpi_change', 'kr_unemployment', 'us_fed_rate_change',
                   'kr_exchange_change', 'vix', 'rate_spread']
    for col in lag_targets:
        if col in df.columns:
            df[f'{col}_lag1'] = df[col].shift(1)
            df[f'{col}_lag2'] = df[col].shift(2)
            df[f'{col}_lag3'] = df[col].shift(3)

    # -- 3-7) Rate Hold Duration (Crucial Feature) --
    hold_months = []
    count = 0
    for val in df['kr_base_rate_change'].values:
        if pd.isna(val) or val == 0:
            count += 1
        else:
            count = 0
        hold_months.append(count)
    df['hold_duration'] = hold_months

    # -- 3-8) Rate Decision Cycle Direction --
    # 1: Hike, -1: Cut, 0: Hold (Forward filled)
    df['last_change_dir'] = df['kr_base_rate_change'].apply(
        lambda x: np.sign(x) if pd.notna(x) and x != 0 else np.nan
    )
    df['last_change_dir'] = df['last_change_dir'].ffill().fillna(0)

    print(f"   Total columns including derived features: {len(df.columns)}")

    # -----------------------------------------
    # 4. Generate Label Columns (Rate Decisions)
    # -----------------------------------------
    print("\nGenerating label columns for interest rate direction...")

    def classify_direction(change):
        if pd.isna(change):
            return None
        elif change > 0:
            return '인상'
        elif change < 0:
            return '인하'
        else:
            return '동결'

    df['label'] = df['kr_base_rate_change'].apply(classify_direction)

    # Encoded labels for classification model (Cut: 0, Hold: 1, Hike: 2)
    label_map = {'인하': 0, '동결': 1, '인상': 2}
    df['label_encoded'] = df['label'].map(label_map)

    label_counts = df['label'].value_counts()
    print("   [Label Distribution]")
    for lbl in ['인상', '동결', '인하']:
        cnt = label_counts.get(lbl, 0)
        ratio = cnt / len(df) * 100
        print(f"     {lbl}: {cnt} rows ({ratio:.1f}%)")

    # -----------------------------------------
    # 5. Forecasting Setup: Shift Targets by -1 Month
    # -----------------------------------------
    print("\nShifting target variables by -1 month for forecasting framework...")
    id_cols = ['date_ym']
    target_cols = ['kr_base_rate_change', 'label', 'label_encoded']
    feature_cols = [c for c in df.columns if c not in id_cols + target_cols]
    
    # Predict next month's rate decisions using current month's features
    df[target_cols] = df[target_cols].shift(-1)

    # -----------------------------------------
    # 6. Cleanup Initial NaN Rows (Due to YoY diff/shift)
    # -----------------------------------------
    first_valid = 12  # 12-month lag due to YoY pct change
    # Exclude the last row as its target variable is shifted to NaN
    df = df.iloc[first_valid:-1].reset_index(drop=True)
    df = df.ffill().bfill()

    # -----------------------------------------
    # 7. Offloaded Dimensionality Reduction & Feature Selection (Data Leakage Free)
    # -----------------------------------------
    print("\n[Feature Selection] Running VIF and Random Forest Feature Selection...")
    
    import sys
    sys.path.insert(0, base_dir)
    from model import InterestRateEnsembleModel
    from utils.dimensionality_reduction import calculate_vif_iteratively
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    
    cfg = InterestRateEnsembleModel
    
    
    # 1) Slice Train Dataset down to ~cfg.TRAIN_END to avoid Data Leakage
    train_df = df[df['date_ym'] <= cfg.TRAIN_END].copy()
    
    drop_cols = [c for c in cfg.DROP_COLS if c in train_df.columns]
    X_train_all = train_df.drop(columns=drop_cols)
    y_train_label = train_df['label_encoded']
    
    # 2) Calculate VIF iteratively using only Train set
    print(f"   Removing multicollinearity based on VIF on Train set (~{cfg.TRAIN_END})...")
    removed_features, _ = calculate_vif_iteratively(X_train_all, n_features=N_FEATURES, threshold=10.0)
    
    X_train_vif = X_train_all.drop(columns=removed_features)
    print(f"   VIF removal completed! Remaining features: {X_train_vif.shape[1]}")
    
    # 3) Random Forest Ensemble Feature Selection using Train set
    print(f"   Selecting top {N_FEATURES} features using 20 Random Forest ensembles...")
    importances_list = []
    for i in range(20):
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=i,
            max_depth=5,
            class_weight='balanced',
            n_jobs=-1
        )
        rf.fit(X_train_vif, y_train_label)
        importances_list.append(rf.feature_importances_)
        
    avg_importances = np.mean(importances_list, axis=0)
    
    feat_imp = pd.DataFrame({
        'feature': X_train_vif.columns,
        'importance': avg_importances
    }).sort_values('importance', ascending=False)
    
    selected_features = feat_imp.head(N_FEATURES)['feature'].tolist()
    
    print(f"   Selected {N_FEATURES} Features:")
    for i, r in feat_imp.head(N_FEATURES).iterrows():
        print(f"     - {r['feature']}: {r['importance']:.4f}")
        
    # 4) Dump selected features list to pkl for test/explain scripts downstream
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(selected_features, os.path.join(models_dir, 'feature_names.pkl'))
    print(f"   Feature names saved to: models/feature_names.pkl")

    # -----------------------------------------
    # 8. Final Reordering (Slice to Selected Features Only)
    # -----------------------------------------
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
    print(f"   Total Feature Columns: {len(df.columns)}")

    # -----------------------------------------
    # 9. Load Preprocessed Data into MySQL Database
    # -----------------------------------------
    print("\n[Database Export] Loading preprocessed data into MySQL...")
    
    # 1) Round all numeric values to 4 decimal places
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(4)
    print("   All numeric columns rounded to 4 decimal places.")

    # 2) DB connection credentials from environment variables
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(base_dir, '.env'))
    
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
                table_name = "ml_baserate_preprocessed"
                
                # Drop existing table to prevent column count mismatch on dynamic schema change
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                
                # 3) CREATE TABLE dynamically based on selected_features
                columns_def = ["date_ym VARCHAR(10) PRIMARY KEY"]
                for col in selected_features:
                    columns_def.append(f"`{col}` DECIMAL(15, 4)")
                columns_def.append("kr_base_rate_change DECIMAL(15, 4)")
                columns_def.append("label VARCHAR(20)")
                columns_def.append("label_encoded INT")
                
                create_table_sql = f"CREATE TABLE {table_name} ({', '.join(columns_def)}) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;"
                cursor.execute(create_table_sql)
                print(f"   Created dynamic table '{table_name}' with {len(selected_features)} features.")

                # 4) Batch INSERT preprocessed DataFrame rows dynamically (NaN to None for NULL binding)
                db_data = df.replace({np.nan: None}).values.tolist()
                
                all_cols = ['date_ym'] + selected_features + ['kr_base_rate_change', 'label', 'label_encoded']
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
