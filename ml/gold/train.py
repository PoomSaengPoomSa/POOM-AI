import os
import pandas as pd
import numpy as np
import joblib
import pymysql
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from model import GoldModel
import warnings

warnings.filterwarnings('ignore')

def load_data_from_mysql():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'data', 'final_dataset.csv')
    
    env_path = os.path.abspath(os.path.join(base_dir, '../../.env'))
    load_dotenv(dotenv_path=env_path)
    
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    DB_NAME = os.getenv('DB_NAME')
    
    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
        print("[Warning] Missing DB configuration. Falling back to local final_dataset.csv...")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"[CSV] Loaded data successfully from local final_dataset.csv: {csv_path}")
            return df
        raise ValueError(f"No DB credentials and final CSV not found at: {csv_path}")
        
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
                sql = "SELECT * FROM ml_gold_preprocessed ORDER BY loaded_date ASC"
                cursor.execute(sql)
                rows = cursor.fetchall()
        finally:
            connection.close()
            
        df = pd.DataFrame(rows)
        for col in df.columns:
            if col not in ['loaded_date']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        print("[DB] Loaded preprocessed data successfully from MySQL table 'ml_gold_preprocessed'.")
        return df
    except Exception as e:
        print(f"[Warning] MySQL connection/query failed ({e}). Falling back to local final_dataset.csv...")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"[CSV] Loaded preprocessed data successfully from local CSV: {csv_path}")
            return df
        raise RuntimeError(f"Database connection failed and local CSV not found at: {csv_path}")

def train_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load Preprocessed Data
    print("Loading preprocessed gold data from MySQL...")
    df = load_data_from_mysql()
    print(f"   Loaded: {len(df)} rows")

    cfg = GoldModel

    # 2. Split Data Chronologically
    split_idx = int(len(df) * cfg.TRAIN_RATIO)
    train_df = df.iloc[:split_idx].copy()
    test_df  = df.iloc[split_idx:].copy()

    # Drop target/meta columns to isolate features
    drop_cols = [c for c in cfg.DROP_COLS if c in df.columns]
    X_train = train_df.drop(columns=drop_cols)
    X_test  = test_df.drop(columns=drop_cols)
    
    y_train = train_df['target_tomorrow_gold_direction']
    y_test  = test_df['target_tomorrow_gold_direction']

    selected_features = list(X_train.columns)

    print(f"\n{'='*55}")
    print("Gold ML Data Split Results")
    print(f"{'='*55}")
    print(f"   Train: {train_df['loaded_date'].min()} ~ {train_df['loaded_date'].max()}  ({len(X_train)} trading days)")
    print(f"   Test : {test_df['loaded_date'].min()} ~ {test_df['loaded_date'].max()}  ({len(X_test)} trading days)")
    print(f"   Total features: {X_train.shape[1]}")

    # Label Distribution
    print(f"\n   [Train Label Distribution]")
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    print(f"     하락/보합(0): {num_neg} rows ({num_neg/len(y_train)*100:.2f}%)")
    print(f"     상승(1): {num_pos} rows ({num_pos/len(y_train)*100:.2f}%)")

    # 3. Standardize Features
    print("\nScaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=selected_features)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=selected_features)

    # 4. Model Training
    print(f"\n{'='*55}")
    print("XGBoost Binary Classifier Model Training")
    print(f"{'='*55}")

    # Optimal class weight calculation to fight class imbalance
    scale_pos_val = float(num_neg) / float(num_pos)
    print(f"   Calculated scale_pos_weight: {scale_pos_val:.4f}")

    builder = GoldModel(random_state=42, scale_pos_weight=scale_pos_val)
    classifier = builder.get_classifier()
    
    classifier.fit(X_train_scaled_df, y_train)
    print("   [OK] Model fitting completed.")

    # 5. Save Models
    joblib.dump(classifier, os.path.join(models_dir, 'gold_xgb_classifier.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'gold_scaler.pkl'))
    joblib.dump(selected_features, os.path.join(models_dir, 'gold_features.pkl'))

    print(f"\n{'='*55}")
    print("Models and resources saved successfully!")
    print(f"{'='*55}")
    print("   models/gold_xgb_classifier.pkl")
    print("   models/gold_scaler.pkl")
    print(f"   models/gold_features.pkl  ({len(selected_features)} features)")

if __name__ == '__main__':
    train_model()
