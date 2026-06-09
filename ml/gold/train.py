import os
import pandas as pd
import numpy as np
import joblib
import pymysql
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv, find_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from model import GoldModel
import warnings
 
warnings.filterwarnings('ignore')

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
 


def load_data_from_mysql():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'data', 'final_dataset.csv')
    
    load_dotenv(find_dotenv())
    
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





def save_performance_to_mysql(accuracy, precision, recall, f1_score, run_id=None):
    import uuid
    if not run_id:
        try:
            active_run = mlflow.active_run()
            run_id = active_run.info.run_id if active_run else uuid.uuid4().hex[:32]
        except Exception:
            run_id = uuid.uuid4().hex[:32]
            
    load_dotenv(find_dotenv())
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    DB_NAME = os.getenv('DB_NAME')
    
    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
        print("[Warning] Missing DB config. Skipping performance save.")
        return
        
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
                # 백엔드의 조회 쿼리에 매칭되는 테이블 구조 보장 (accuracy, precision, recall, f1_score)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS gold_performance (
                    run_id VARCHAR(50) NOT NULL PRIMARY KEY,
                    accuracy DOUBLE NOT NULL,
                    `precision` DOUBLE NOT NULL,
                    recall DOUBLE NOT NULL,
                    f1_score DOUBLE NOT NULL,
                    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                sql = """
                INSERT INTO gold_performance (run_id, accuracy, `precision`, recall, f1_score)
                VALUES (%s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (run_id, accuracy, precision, recall, f1_score))
            connection.commit()
            print("[DB] Successfully saved gold performance metrics into MySQL.")
        finally:
            connection.close()
    except Exception as e:
        print(f"[Error] Failed to save performance metrics to MySQL: {e}")


def train_model(valid_mode=False):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    load_dotenv(find_dotenv())
 
    # MLflow 설정
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI', None)
    skip_mlflow = False
    
    if tracking_uri:
        try:
            import requests
            resp = requests.get(tracking_uri, timeout=2)
            if resp.status_code == 404 and "TUNNEL NOT FOUND" in resp.text:
                print("[Warning] MLflow remote tunnel is offline. Skipping MLflow logging.")
                skip_mlflow = True
        except Exception as e:
            print(f"[Warning] Failed to connect to MLflow tracking server: {e}. Skipping MLflow logging.")
            skip_mlflow = True
    else:
        print("[Warning] MLFLOW_TRACKING_URI is not set. Skipping MLflow logging.")
        skip_mlflow = True

    if skip_mlflow:
        class DummyMLflow:
            def __getattr__(self, name): return self
            def __call__(self, *args, **kwargs): return self
            def __enter__(self): return self
            def __exit__(self, exc_type, exc_val, exc_tb): pass
            def active_run(self, *args, **kwargs): return None
        
        global mlflow
        mlflow = DummyMLflow()
    else:
        print(f"[MLflow] Using remote MLflow tracking: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("gold")
 
    with mlflow.start_run():
 
        # 1. Load Preprocessed Data
        print("Loading preprocessed gold data from MySQL...")
        df = load_data_from_mysql()
        print(f"   Loaded: {len(df)} rows")
 
        cfg = GoldModel
 
        # 2. Split Data Chronologically (Fixed Date Split)
        df['loaded_date'] = df['loaded_date'].astype(str).str.strip()
        if valid_mode:
            train_df = df[df['loaded_date'] <= cfg.TRAIN_END].copy()
            eval_df  = df[df['loaded_date'].between(cfg.VALID_START, cfg.VALID_END)].copy()
            eval_name = "Validation"
        else:
            train_df = df[df['loaded_date'] <= cfg.VALID_END].copy()
            eval_df  = df[df['loaded_date'].between(cfg.TEST_START, cfg.TEST_END)].copy()
            eval_name = "Test"
 
        drop_cols = [c for c in cfg.DROP_COLS if c in df.columns]
        X_train = train_df.drop(columns=drop_cols)
        X_test  = eval_df.drop(columns=drop_cols)
        
        y_train = train_df['target_tomorrow_gold_direction']
        y_test  = eval_df['target_tomorrow_gold_direction']
 
        selected_features = list(X_train.columns)
 
        print(f"\n{'='*55}")
        print(f"Gold ML Data Split Results ({eval_name} Mode)")
        print(f"{'='*55}")
        print(f"   Train: {train_df['loaded_date'].min()} ~ {train_df['loaded_date'].max()}  ({len(X_train)} trading days)")
        print(f"   Eval : {eval_df['loaded_date'].min()} ~ {eval_df['loaded_date'].max()}  ({len(X_test)} trading days)")
        print(f"   Total features: {X_train.shape[1]}")
 
        # MLflow - 데이터 정보 기록
        mlflow.log_param("train_start", train_df['loaded_date'].min())
        mlflow.log_param("train_end", train_df['loaded_date'].max())
        mlflow.log_param("eval_start", eval_df['loaded_date'].min())
        mlflow.log_param("eval_end", eval_df['loaded_date'].max())
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("eval_rows", len(X_test))
        mlflow.log_param("num_features", X_train.shape[1])
 
        # Label Distribution
        print(f"\n   [Train Label Distribution]")
        num_neg = (y_train == 0).sum()
        num_pos = (y_train == 1).sum()
        print(f"     하락/보합(0): {num_neg} rows ({num_neg/len(y_train)*100:.2f}%)")
        print(f"     상승(1): {num_pos} rows ({num_pos/len(y_train)*100:.2f}%)")
        mlflow.log_param("train_label_하락보합_count", int(num_neg))
        mlflow.log_param("train_label_상승_count", int(num_pos))
 
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
 
        scale_pos_val = float(num_neg) / float(num_pos)
        print(f"   Calculated scale_pos_weight: {scale_pos_val:.4f}")
 
        # MLflow - 하이퍼파라미터 기록
        mlflow.log_param("random_state", 42)
        mlflow.log_param("scale_pos_weight", round(scale_pos_val, 4))
 
        builder = GoldModel(random_state=42, scale_pos_weight=scale_pos_val)
        classifier = builder.get_classifier()
        
        classifier.fit(X_train_scaled_df, y_train)
        print("   [OK] Model fitting completed.")
 
        # MLflow - 성능 지표 기록
        y_pred = classifier.predict(X_test_scaled_df)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
 
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        print(f"\n   [Test 성능]")
        print(f"     Accuracy  : {accuracy:.4f}")
        print(f"     F1 Score  : {f1:.4f}")
        print(f"     Precision : {precision:.4f}")
        print(f"     Recall    : {recall:.4f}")

        import uuid
        try:
            active_run = mlflow.active_run()
            run_id_val = active_run.info.run_id if active_run else uuid.uuid4().hex[:32]
        except Exception:
            run_id_val = uuid.uuid4().hex[:32]

        # 꼬임 버그를 완벽하게 제거하기 위해 명시적 키워드 인자로 호출
        save_performance_to_mysql(accuracy=accuracy, precision=precision, recall=recall, f1_score=f1, run_id=run_id_val)
 
        # 5. Save Models
        joblib.dump(classifier, os.path.join(models_dir, 'gold_xgb_classifier.pkl'))
        joblib.dump(scaler, os.path.join(models_dir, 'gold_scaler.pkl'))
        joblib.dump(selected_features, os.path.join(models_dir, 'gold_features.pkl'))
 
        # MLflow - 모델 저장 (MinIO artifact)
        try:
            mlflow.sklearn.log_model(classifier, "classifier")
        except Exception as e:
            print(f"[Warning] Failed to log model to MLflow S3 artifact: {e}")
 
        print(f"\n{'='*55}")
        print("Models and resources saved successfully!")
        print(f"{'='*55}")
        print("   models/gold_xgb_classifier.pkl")
        print("   models/gold_scaler.pkl")
        print(f"   models/gold_features.pkl  ({len(selected_features)} features)")
 
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--valid', action='store_true', help='Validation mode: train on train, validate on valid, do not use test.')
    args = parser.parse_known_args()[0]
    
    train_model(valid_mode=args.valid)