import os
import sys
import joblib
import pymysql
import pandas as pd
import numpy as np
import uuid
from dotenv import load_dotenv, find_dotenv

def predict_latest():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    load_dotenv(find_dotenv())

    # Load Model, Scaler, and features list
    clf_path = os.path.join(models_dir, 'gold_xgb_classifier.pkl')
    scaler_path = os.path.join(models_dir, 'gold_scaler.pkl')
    feat_path = os.path.join(models_dir, 'gold_features.pkl')
    
    if not (os.path.exists(clf_path) and os.path.exists(scaler_path) and os.path.exists(feat_path)):
        print(f"[Error] Missing model/scaler/features list. Run train.py first.")
        sys.exit(1)

    classifier = joblib.load(clf_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(feat_path)

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
            sql = "SELECT * FROM ml_gold_preprocessed ORDER BY loaded_date DESC LIMIT 1"
            cursor.execute(sql)
            row = cursor.fetchone()
    finally:
        connection.close()

    if not row:
        print("[Error] No preprocessed data found in ml_gold_preprocessed.")
        sys.exit(1)

    print(f"[PREDICT] Loaded latest preprocessed row for date: {row['loaded_date']}")

    # Convert row to DataFrame and align with features list
    df_row = pd.DataFrame([row])
    X_latest = df_row[feature_names]

    # Convert all columns to numeric
    for col in X_latest.columns:
        X_latest[col] = pd.to_numeric(X_latest[col], errors='coerce')

    # Scale input features
    X_latest_scaled = scaler.transform(X_latest)
    X_latest_scaled_df = pd.DataFrame(X_latest_scaled, columns=feature_names)

    # Predict Probabilities
    latest_proba = classifier.predict_proba(X_latest_scaled_df)[0]
    prob_fall = float(latest_proba[0])
    prob_rise = float(latest_proba[1])

    print(f"Predictions - Fall: {prob_fall:.2%}, Rise: {prob_rise:.2%}")

    # Save to MySQL
    run_id = f"predict_gold_{uuid.uuid4().hex[:16]}"
    
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
            CREATE TABLE IF NOT EXISTS gold_predictions (
                run_id VARCHAR(50) NOT NULL,
                prob_rise DOUBLE NOT NULL,
                prob_fall DOUBLE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            sql = """
            INSERT INTO gold_predictions (run_id, prob_rise, prob_fall)
            VALUES (%s, %s, %s)
            """
            cursor.execute(sql, (run_id, prob_rise, prob_fall))
        connection.commit()
        print(f"[DB] Successfully saved gold prediction (run_id: {run_id}) into MySQL.")
    finally:
        connection.close()

if __name__ == '__main__':
    predict_latest()
