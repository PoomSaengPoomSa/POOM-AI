import os
import sys
import pickle
import pymysql
import pandas as pd
import numpy as np
import uuid
from dotenv import load_dotenv, find_dotenv

def get_latest_actual_realestate_index():
    load_dotenv(find_dotenv())
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    DB_NAME = os.getenv('DB_NAME')
    
    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
        return None
        
    try:
        connection = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=int(DB_PORT),
            charset='utf8mb4'
        )
        try:
            with connection.cursor() as cursor:
                # Query the latest actual index (non-NULL value) from preprocessed table
                sql = "SELECT house_price_idx FROM ml_realestate_preprocessed WHERE house_price_idx IS NOT NULL ORDER BY date_ym DESC LIMIT 1"
                cursor.execute(sql)
                res = cursor.fetchone()
                if res:
                    return float(res[0])
        finally:
            connection.close()
    except Exception as e:
        print(f"[Error] Failed to fetch latest actual realestate index: {e}")
    return None

def predict_latest():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    load_dotenv(find_dotenv())

    # Load Model and selected features list
    model_path = os.path.join(models_dir, 'ensemble_model.pkl')
    features_path = os.path.join(models_dir, 'selected_features.pkl')
    
    if not (os.path.exists(model_path) and os.path.exists(features_path)):
        print(f"[Error] Missing model or selected features list. Run train.py first.")
        sys.exit(1)

    with open(model_path, 'rb') as f:
        ensemble = pickle.load(f)
    with open(features_path, 'rb') as f:
        selected_features = pickle.load(f)

    # DB Connection Setup
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    DB_NAME = os.getenv('DB_NAME')
    
    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
        raise ValueError("Missing database configuration in .env file.")
        
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
            # Query the latest preprocessed row
            sql = "SELECT * FROM ml_realestate_preprocessed ORDER BY date_ym DESC LIMIT 1"
            cursor.execute(sql)
            row = cursor.fetchone()
    finally:
        connection.close()

    if not row:
        print("[Error] No preprocessed data found in ml_realestate_preprocessed.")
        sys.exit(1)

    print(f"[PREDICT] Loaded latest preprocessed row for month: {row['date_ym']}")

    # Convert row to DataFrame and align with features list
    df_row = pd.DataFrame([row])
    X_latest = df_row[selected_features]

    # Convert all columns to numeric
    for col in X_latest.columns:
        X_latest[col] = pd.to_numeric(X_latest[col], errors='coerce')

    # Predict change rate
    latest_predicted_value = float(ensemble.predict(X_latest.values)[0])

    # Calculate predicted index based on latest actual index
    re_today = get_latest_actual_realestate_index()
    if re_today is not None:
        predicted_index = re_today * (1 + latest_predicted_value / 100)
        print(f"Calculated predicted_index: {predicted_index:.4f} using re_today: {re_today} and predicted_value: {latest_predicted_value}%")
    else:
        predicted_index = None
        print("[Warning] Could not calculate predicted_index because re_today is missing.")

    # Save to MySQL
    run_id = f"predict_realestate_{uuid.uuid4().hex[:16]}"
    
    connection = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT,
        charset='utf8mb4'
    )
    
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS realestate_predictions (
                run_id VARCHAR(50) NOT NULL,
                predicted_value DOUBLE NOT NULL,
                predicted_index DOUBLE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            sql = """
            INSERT INTO realestate_predictions (run_id, predicted_value, predicted_index)
            VALUES (%s, %s, %s)
            """
            cursor.execute(sql, (run_id, latest_predicted_value, predicted_index))
        connection.commit()
        print(f"[DB] Successfully saved real estate prediction (run_id: {run_id}) into MySQL.")
    finally:
        connection.close()

if __name__ == '__main__':
    predict_latest()
