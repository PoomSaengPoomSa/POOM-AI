import os
import sys
import pickle
import mlflow
import mlflow.sklearn
import numpy as np
import pymysql
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from model import RealEstateEnsembleRegressor






def save_performance_to_mysql(rmse, r2_score, mae, mse, run_id=None):
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
                # evaluated_at 자동 생성 기둥 보장
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS realestate_performance (
                    run_id VARCHAR(50) NOT NULL PRIMARY KEY,
                    rmse DOUBLE NOT NULL,
                    r2_score DOUBLE NOT NULL,
                    mae DOUBLE NOT NULL,
                    mse DOUBLE NOT NULL,
                    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                sql = """
                INSERT INTO realestate_performance (run_id, rmse, r2_score, mae, mse)
                VALUES (%s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (run_id, rmse, r2_score, mae, mse))
            connection.commit()
            print("[DB] Successfully saved real_estate performance metrics into MySQL.")
        finally:
            connection.close()
    except Exception as e:
        print(f"[Error] Failed to save performance metrics to MySQL: {e}")


def get_latest_actual_realestate_index():
    import pymysql
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
                sql = "SELECT house_price_idx FROM ml_realestate_preprocessed ORDER BY date_ym DESC LIMIT 1"
                cursor.execute(sql)
                res = cursor.fetchone()
                if res:
                    return float(res[0])
        finally:
            connection.close()
    except Exception as e:
        print(f"[Error] Failed to fetch latest actual realestate index: {e}")
    return None

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
            print(f"[CSV] Loaded preprocessed data successfully from local final_dataset.csv: {csv_path}")
            return df
        raise ValueError(f"No DB credentials and final CSV not found at: {csv_path}")
        
    try:
        connection = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=int(DB_PORT),
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        try:
            with connection.cursor() as cursor:
                sql = "SELECT * FROM ml_realestate_preprocessed ORDER BY date_ym ASC"
                cursor.execute(sql)
                rows = cursor.fetchall()
        finally:
            connection.close()
            
        df = pd.DataFrame(rows)
        for col in df.columns:
            if col not in ['date_ym']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        print("[DB] Loaded preprocessed data successfully from MySQL table 'ml_realestate_preprocessed'.")
        return df
    except Exception as e:
        print(f"[Warning] MySQL query failed ({e}). Falling back to local final_dataset.csv...")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"[CSV] Loaded preprocessed data successfully from local CSV: {csv_path}")
            return df
        raise RuntimeError(f"Database connection failed and local CSV not found at: {csv_path}")


def run_train(valid_mode=False):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')

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
        mlflow.set_experiment("real_estate")
 
    with mlflow.start_run():
 
        # Load Preprocessed Data
        df = load_data_from_mysql()
        
        # Load Selected Features list
        features_path = os.path.join(models_dir, 'selected_features.pkl')
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Selected features list not found at: {features_path}. Run preprocess.py first.")
            
        with open(features_path, 'rb') as f:
            selected_features = pickle.load(f)
 
        from model import RealEstateEnsembleRegressor as cfg
        df['date_ym'] = df['date_ym'].astype(str).str.strip()
        TARGET = "next_change_rate"

        if valid_mode:
            train_df = df[df['date_ym'] <= cfg.TRAIN_END].copy()
            test_df  = df[df['date_ym'].between(cfg.VALID_START, cfg.VALID_END)].copy()
            eval_name = "Validation"
        else:
            train_df = df[df['date_ym'] <= cfg.VALID_END].copy()
            test_df  = df[df['date_ym'].between(cfg.TEST_START, cfg.TEST_END)].copy()
            eval_name = "Test"
 
        train_clean = train_df.dropna(subset=[TARGET]).copy()
        test_clean  = test_df.dropna(subset=[TARGET]).copy()

        X_train_sc = train_clean[selected_features].values
        X_test_sc  = test_clean[selected_features].values
        y_train = train_clean[TARGET].values
        y_test  = test_clean[TARGET].values

        # MLflow - 전처리 파라미터 기록
        mlflow.log_param("valid_mode", str(valid_mode))
        mlflow.log_param("train_start", train_clean['date_ym'].min())
        mlflow.log_param("train_end", train_clean['date_ym'].max())
        mlflow.log_param("test_start", test_clean['date_ym'].min())
        mlflow.log_param("test_end", test_clean['date_ym'].max())
        mlflow.log_param("train_rows", len(X_train_sc))
        mlflow.log_param("test_rows", len(X_test_sc))
        mlflow.log_param("num_features", len(selected_features))
        mlflow.log_param("random_state", 42)
 
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
 
        # -----------------------------------------
        # Train final ensemble model
        # -----------------------------------------
        ensemble = RealEstateEnsembleRegressor(random_state=42)
        ensemble.fit(X_train_sc, y_train)
 
        # MLflow - Train set 성능 기록
        train_pred = ensemble.predict(X_train_sc)
        train_r2   = r2_score(y_train, train_pred)
        train_mae  = mean_absolute_error(y_train, train_pred)
        train_mse  = mean_squared_error(y_train, train_pred)
        train_rmse = np.sqrt(train_mse)
 
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("train_mse", train_mse)
        mlflow.log_metric("train_rmse", train_rmse)
 
        # MLflow - Test set 성능 기록
        test_pred = ensemble.predict(X_test_sc)
        test_r2   = r2_score(y_test, test_pred)
        test_mae  = mean_absolute_error(y_test, test_pred)
        test_mse  = mean_squared_error(y_test, test_pred)
        test_rmse = np.sqrt(test_mse)
 
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("test_rmse", test_rmse)
 
        print("\n" + "=" * 55)
        print(f"  Real Estate Our Model (Tuned XGBoost) - Train / {eval_name} Performance")
        print("=" * 55)
        print(f"   [Train]  R2: {train_r2:.4f} | MAE: {train_mae:.4f}% | RMSE: {train_rmse:.4f}")
        print(f"   [{eval_name} ]  R2: {test_r2:.4f} | MAE: {test_mae:.4f}% | RMSE: {test_rmse:.4f}")
        print("=" * 55 + "\n")

        import uuid
        try:
            active_run = mlflow.active_run()
            run_id_val = active_run.info.run_id if active_run else uuid.uuid4().hex[:32]
        except Exception:
            run_id_val = uuid.uuid4().hex[:32]
 
        # 테스트셋 성능을 DB에 저장
        save_performance_to_mysql(rmse=test_rmse, r2_score=test_r2, mae=test_mae, mse=test_mse, run_id=run_id_val)
 
        # Setup directories and save model
        model_path = os.path.join(models_dir, 'ensemble_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(ensemble, f)
 
        # MLflow - 모델 저장 (MinIO artifact)
        try:
            mlflow.sklearn.log_model(ensemble, "ensemble_model")
        except Exception as e:
            print(f"[Warning] Failed to log model to MLflow S3 artifact: {e}")
 
        print("=" * 55)
        print("Training Pipeline Completed Successfully!")
        print("=" * 55)
        print(f"  Saved Model   : {model_path}")
        print(f"  Saved Features: {features_path} and .txt")
        print(f"  Features size : {len(selected_features)}")
 
 
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--valid', action='store_true', help='Validation mode: train on train, validate on valid, do not use test.')
    args = parser.parse_known_args()[0]
    
    run_train(valid_mode=args.valid)