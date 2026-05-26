import os
import sys
import pandas as pd
import numpy as np
import pymysql
from dotenv import load_dotenv

def load_raw_data_from_mysql():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, 'data', 'raw_data.csv')
    
    env_path = os.path.abspath(os.path.join(base_dir, '../../.env'))
    load_dotenv(dotenv_path=env_path)
    
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    DB_NAME = os.getenv('DB_NAME')
    
    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
        print("[Warning] Missing DB credentials in .env. Falling back to local raw_data.csv...")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"[CSV] Loaded raw data successfully from local raw_data.csv: {csv_path}")
            return df
        raise ValueError("Missing database credentials and no cached raw_data.csv found.")
        
    try:
        DB_PORT = int(DB_PORT)
        connection = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=5
        )
        try:
            with connection.cursor() as cursor:
                sql = "SELECT * FROM ml_gold_raw ORDER BY loaded_date ASC"
                cursor.execute(sql)
                rows = cursor.fetchall()
        finally:
            connection.close()
            
        df = pd.DataFrame(rows)
        df['loaded_date'] = pd.to_datetime(df['loaded_date']).dt.strftime('%Y-%m-%d')
        
        if 'gr_id' in df.columns:
            df = df.drop(columns=['gr_id'])
            
        cols = ['loaded_date'] + [col for col in df.columns if col != 'loaded_date']
        df = df[cols]
        
        numeric_cols = [col for col in df.columns if col != 'loaded_date']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        print("[DB] Loaded raw data successfully from MySQL table 'ml_gold_raw'.")
        return df
        
    except Exception as e:
        print(f"[Warning] MySQL connection/query failed ({e}). Falling back to local raw_data.csv...")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"[CSV] Loaded raw data successfully from local raw_data.csv: {csv_path}")
            return df
        raise RuntimeError(f"Database connection failed and no cached raw_data.csv found at: {csv_path}")

def preprocess():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')

    # 1. Load Raw Data
    print("Loading raw data from MySQL...")
    df = load_raw_data_from_mysql()
    print(f"   Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"   Period: {df['loaded_date'].min()} ~ {df['loaded_date'].max()}")

    df = df.sort_values('loaded_date').reset_index(drop=True)

    # 2. Drop Weekends/Holidays (gold is NaN)
    df = df.dropna(subset=['gold']).reset_index(drop=True)
    print(f"   After dropping weekends/holidays: {df.shape[0]} rows")

    # 3. Impute Missing Macroeconomic Indicators (Forward fill & Backward fill)
    feature_cols = ['kr_usd_exchange', 'wti_oil', 'dxy_proxy', 'vix', 'kospi200', 'sp500', 'kr_cpi']
    df[feature_cols] = df[feature_cols].ffill().bfill()

    # 4. Generate Daily Change Rates
    change_rate_cols = []
    for col in ['gold', 'kr_usd_exchange', 'wti_oil', 'dxy_proxy', 'vix', 'kospi200', 'sp500']:
        col_name = f"{col}_change_rate"
        df[col_name] = df[col].pct_change()
        change_rate_cols.append(col_name)

    # 5. Financial Engineering Indicators
    # A. RSI (Relative Strength Index) - 14-day
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

    # C. EMA (Exponential Moving Average) - 5-day / 20-day
    df['gold_change_rate_ema_5'] = df['gold_change_rate'].ewm(span=5, adjust=False).mean()
    df['gold_change_rate_ema_20'] = df['gold_change_rate'].ewm(span=20, adjust=False).mean()

    # D. SP500 vs KOSPI200 Spread
    df['sp500_kospi200_spread'] = df['sp500_change_rate'] - df['kospi200_change_rate']

    # E. Gold-Dollar Interaction
    df['gold_dxy_interaction'] = df['gold_change_rate'] * df['dxy_proxy_change_rate']

    new_derived_cols = [
        'gold_rsi_14', 'gold_macd', 'gold_macd_signal', 
        'gold_change_rate_ema_5', 'gold_change_rate_ema_20',
        'sp500_kospi200_spread', 'gold_dxy_interaction'
    ]

    # 6. Lag Features (1, 2, 3-day lags)
    lag_features = []
    # Lags for base change rates
    for col in change_rate_cols:
        for lag in [1, 2, 3]:
            lag_name = f"{col}_lag_{lag}"
            df[lag_name] = df[col].shift(lag)
            lag_features.append(lag_name)
            
    # Lags for advanced derived columns (prevents look-ahead bias)
    for col in new_derived_cols:
        for lag in [1, 2, 3]:
            lag_name = f"{col}_lag_{lag}"
            df[lag_name] = df[col].shift(lag)
            lag_features.append(lag_name)

    # 7. Rolling Features (5-day & 20-day SMA/STD)
    df['gold_change_rate_sma_5'] = df['gold_change_rate'].rolling(5).mean()
    df['gold_change_rate_sma_20'] = df['gold_change_rate'].rolling(20).mean()
    df['gold_change_rate_std_5'] = df['gold_change_rate'].rolling(5).std()
    df['gold_change_rate_std_20'] = df['gold_change_rate'].rolling(20).std()

    rolling_features = [ 
        'gold_change_rate_sma_5', 'gold_change_rate_sma_20',
        'gold_change_rate_std_5', 'gold_change_rate_std_20'
    ]

    # 8. Create Target Variables (Shift by -1 day)
    df['target_tomorrow_gold_change_rate'] = df['gold_change_rate'].shift(-1)
    df['target_tomorrow_gold_direction'] = (df['target_tomorrow_gold_change_rate'] > 0).astype(int)

    # Clean up NaNs due to shift, rolling, and lags
    df = df.dropna().reset_index(drop=True)

    # Group features
    base_features = [
        'gold', 'gold_change_rate', 'kr_cpi', 'kr_usd_exchange', 'wti_oil', 
        'dxy_proxy', 'vix', 'kospi200', 'sp500', 
        'kr_usd_exchange_change_rate', 'wti_oil_change_rate', 'dxy_proxy_change_rate', 
        'vix_change_rate', 'kospi200_change_rate', 'sp500_change_rate'
    ]
    features = base_features + new_derived_cols + lag_features + rolling_features

    print(f"   Engineered {len(features)} feature columns.")

    # 9. Slice dataset to final ordering
    final_order = ['loaded_date'] + features + ['target_tomorrow_gold_change_rate', 'target_tomorrow_gold_direction']
    df = df[final_order]

    # Save to CSV
    save_path = os.path.join(data_dir, 'final_dataset.csv')
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"   Saved preprocessed data to: {save_path}")

    # 10. Load Preprocessed Data into MySQL Database
    print("\n[Database Export] Loading preprocessed gold data into MySQL...")
    
    # Round numeric values to 6 decimal places for high fidelity
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(6)

    # DB connection credentials
    env_path = os.path.abspath(os.path.join(base_dir, '../../.env'))
    load_dotenv(dotenv_path=env_path)
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
                table_name = "ml_gold_preprocessed"
                
                # Drop existing table
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                
                # Dynamic CREATE TABLE based on features
                columns_def = ["loaded_date VARCHAR(20) PRIMARY KEY"]
                for col in features:
                    columns_def.append(f"`{col}` DECIMAL(15, 6)")
                columns_def.append("target_tomorrow_gold_change_rate DECIMAL(15, 6)")
                columns_def.append("target_tomorrow_gold_direction INT")
                
                create_table_sql = f"CREATE TABLE {table_name} ({', '.join(columns_def)}) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;"
                cursor.execute(create_table_sql)
                print(f"   Created dynamic table '{table_name}' with {len(features)} features.")

                # Batch INSERT preprocessed DataFrame rows (NaN to None for NULL binding)
                db_data = df.replace({np.nan: None}).values.tolist()
                
                placeholders = ", ".join(["%s"] * len(final_order))
                col_names_quoted = ", ".join([f"`{c}`" for c in final_order])
                
                insert_sql = f"INSERT INTO {table_name} ({col_names_quoted}) VALUES ({placeholders})"
                
                cursor.executemany(insert_sql, db_data)
                connection.commit()
                print(f"   Successfully uploaded {len(db_data)} preprocessed rows into MySQL table '{table_name}'!")
                
            connection.close()
        except Exception as e:
            print(f"   [Error] MySQL Export failed: {e}")

    print("\n" + "=" * 55)
    print("Preprocessing completed successfully!")
    print("=" * 55)
    print(f"   Final Size  : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"   Period     : {df['loaded_date'].min()} ~ {df['loaded_date'].max()}")

if __name__ == '__main__':
    preprocess()
