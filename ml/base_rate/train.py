import os
import pandas as pd
import numpy as np
import joblib
import pymysql
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv, find_dotenv
from model import InterestRateEnsembleModel
import warnings
 
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
 
 
def load_data_from_mysql():
    base_dir = os.path.dirname(os.path.abspath(__file__))
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
            sql = "SELECT * FROM ml_baserate_preprocessed ORDER BY date_ym ASC"
            cursor.execute(sql)
            rows = cursor.fetchall()
    finally:
        connection.close()
        
    df = pd.DataFrame(rows)
    for col in df.columns:
        if col not in ['date_ym', 'label']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df
 
 
def train_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(find_dotenv())

    # MLflow 설정
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', None))
    mlflow.set_experiment("base_rate")
 
    with mlflow.start_run():
 
        # -----------------------------------------
        # 1. Load Data & Split (MySQL Database)
        # -----------------------------------------
        df = load_data_from_mysql()
        df['date_ym'] = df['date_ym'].astype(str).str.strip()
        print(f"Data load completed from MySQL: {len(df)} rows")
 
        cfg = InterestRateEnsembleModel
 
        train_df = df[df['date_ym'] <= cfg.TRAIN_END].copy()
        test_df  = df[df['date_ym'] >= cfg.TEST_START].copy()
 
        drop_cols = [c for c in cfg.DROP_COLS if c in df.columns]
 
        X_train = train_df.drop(columns=drop_cols)
        X_test  = test_df.drop(columns=drop_cols)
 
        y_train_label  = train_df['label_encoded']
        y_test_label   = test_df['label_encoded']
 
        selected_features = list(X_train.columns)
 
        print(f"\n{'='*55}")
        print("Data Split Results")
        print(f"{'='*55}")
        print(f"   Train: {train_df['date_ym'].min()} ~ {train_df['date_ym'].max()}  ({len(X_train)} months)")
        print(f"   Test : {test_df['date_ym'].min()} ~ {test_df['date_ym'].max()}  ({len(X_test)} months)")
        print(f"   Total trained features: {X_train.shape[1]}")
        print(f"   Features list: {selected_features}")
 
        # MLflow - 데이터 정보 기록
        mlflow.log_param("train_start", train_df['date_ym'].min())
        mlflow.log_param("train_end", train_df['date_ym'].max())
        mlflow.log_param("test_start", test_df['date_ym'].min())
        mlflow.log_param("test_end", test_df['date_ym'].max())
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("num_features", X_train.shape[1])
 
        # Train Label Distribution
        print(f"\n   [Train Label Distribution]")
        for lbl_name, lbl_val in [('인하', 0), ('동결', 1), ('인상', 2)]:
            cnt = (y_train_label == lbl_val).sum()
            print(f"     {lbl_name}: {cnt} rows ({cnt/len(y_train_label)*100:.1f}%)")
            mlflow.log_param(f"train_label_{lbl_name}_count", int(cnt))
 
        # -----------------------------------------
        # 2. Model Training (Classification)
        # -----------------------------------------
        print(f"\n{'='*55}")
        print("Direction Classification (Cut/Hold/Hike) Model Training")
        print(f"{'='*55}")
 
        builder = InterestRateEnsembleModel(random_state=42)
 
        from sklearn.utils.class_weight import compute_sample_weight
        from sklearn.metrics import accuracy_score, f1_score
 
        # MLflow - 하이퍼파라미터 기록
        sample_weight_map = {0: 5.0, 1: 1.0, 2: 5.0}
        mlflow.log_param("random_state", 42)
        mlflow.log_param("weight_cut", sample_weight_map[0])
        mlflow.log_param("weight_hold", sample_weight_map[1])
        mlflow.log_param("weight_hike", sample_weight_map[2])
 
        sample_weights = compute_sample_weight(class_weight=sample_weight_map, y=y_train_label)
 
        classifier = builder.get_classifier()
        classifier.fit(X_train, y_train_label, sample_weight=sample_weights)
 
        # MLflow - 성능 지표 기록
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test_label, y_pred)
        f1 = f1_score(y_test_label, y_pred, average='weighted')
 
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score_weighted", f1)
        print(f"\n   [Test 성능]")
        print(f"     Accuracy : {accuracy:.4f}")
        print(f"     F1 Score : {f1:.4f}")
 
        # -----------------------------------------
        # 3. Save Models
        # -----------------------------------------
        models_dir  = os.path.join(base_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
 
        reg_path = os.path.join(models_dir, 'regressor.pkl')
        if os.path.exists(reg_path):
            os.remove(reg_path)
 
        joblib.dump(classifier, os.path.join(models_dir, 'classifier.pkl'))
        joblib.dump(selected_features, os.path.join(models_dir, 'feature_names.pkl'))
 
        # MLflow - 모델 저장 (MinIO artifact)
        mlflow.sklearn.log_model(classifier, "classifier")
 
        print(f"\n{'='*55}")
        print("Model saved successfully!")
        print(f"{'='*55}")
        print("   models/classifier.pkl")
        print(f"   models/feature_names.pkl  ({len(selected_features)} features)")
 
 
if __name__ == '__main__':
    train_model()