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

    # Load Model and Feature Names
    clf_path = os.path.join(models_dir, 'classifier.pkl')
    feat_path = os.path.join(models_dir, 'feature_names.pkl')
    
    if not (os.path.exists(clf_path) and os.path.exists(feat_path)):
        print(f"[Error] Missing model or features list. Run train.py first.")
        sys.exit(1)

    classifier = joblib.load(clf_path)
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
            # Query the latest preprocessed row (ideally 202606)
            sql = "SELECT * FROM ml_baserate_preprocessed ORDER BY date_ym DESC LIMIT 1"
            cursor.execute(sql)
            row = cursor.fetchone()
    finally:
        connection.close()

    if not row:
        print("[Error] No preprocessed data found in ml_baserate_preprocessed.")
        sys.exit(1)

    print(f"[PREDICT] Loaded latest preprocessed row for month: {row['date_ym']}")

    # Convert row to DataFrame and align with features list
    df_row = pd.DataFrame([row])
    X_latest = df_row[feature_names]

    # Convert all columns to numeric
    for col in X_latest.columns:
        X_latest[col] = pd.to_numeric(X_latest[col], errors='coerce')

    # Predict Probabilities
    latest_proba = classifier.predict_proba(X_latest)[0]
    prob_cut = float(latest_proba[0])
    prob_freeze = float(latest_proba[1])
    prob_hike = float(latest_proba[2])

    print(f"Predictions - Cut: {prob_cut:.2%}, Hold: {prob_freeze:.2%}, Hike: {prob_hike:.2%}")

    # Save to MySQL
    run_id = f"predict_baserate_{uuid.uuid4().hex[:16]}"
    
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
            CREATE TABLE IF NOT EXISTS baserate_predictions (
                run_id VARCHAR(50) NOT NULL,
                prob_hike DOUBLE NOT NULL,
                prob_freeze DOUBLE NOT NULL,
                prob_cut DOUBLE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            sql = """
            INSERT INTO baserate_predictions (run_id, prob_hike, prob_freeze, prob_cut)
            VALUES (%s, %s, %s, %s)
            """
            cursor.execute(sql, (run_id, prob_hike, prob_freeze, prob_cut))
        connection.commit()
        print(f"[DB] Successfully saved base rate prediction (run_id: {run_id}) into MySQL.")
    finally:
        connection.close()

if __name__ == '__main__':
    predict_latest()
