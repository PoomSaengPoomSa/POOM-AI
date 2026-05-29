import os
import pickle
import mlflow
import mlflow.sklearn
import numpy as np
import pymysql
from dotenv import load_dotenv, find_dotenv
from utils.preprocess import preprocess_data
from model import RealEstateEnsembleRegressor


def save_prediction_to_mysql(predicted_value, run_id):
    import pymysql
    load_dotenv(find_dotenv())
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    DB_NAME = os.getenv('DB_NAME')
    
    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
        print("[Warning] Missing DB config. Skipping prediction save.")
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
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS realestate_predictions (
                    run_id VARCHAR(50) NOT NULL,
                    predicted_value DOUBLE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                sql = """
                INSERT INTO realestate_predictions (run_id, predicted_value)
                VALUES (%s, %s)
                """
                cursor.execute(sql, (run_id, predicted_value))
            connection.commit()
            print("[DB] Successfully saved realestate_predictions (1 row) into MySQL.")
        finally:
            connection.close()
    except Exception as e:
        print(f"[Error] Failed to save realestate predictions to MySQL: {e}")


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


def run_train():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    load_dotenv(find_dotenv())
 
    # MLflow 설정
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', None))
    mlflow.set_experiment("real_estate")
 
    with mlflow.start_run():
 
        # Preprocess
        data = preprocess_data(test_months=24, vif_threshold=10.0)
        if data is None:
            print("[Error] Preprocessing failed.")
            return
 
        X_train_sc = data['X_train_sc']
        y_train = data['y_train']
        selected_features = data['features']
        scaler = data['scaler']
 
        # MLflow - 전처리 파라미터 기록
        mlflow.log_param("test_months", 24)
        mlflow.log_param("vif_threshold", 10.0)
        mlflow.log_param("train_rows", len(X_train_sc))
        mlflow.log_param("num_features", len(selected_features))
        mlflow.log_param("random_state", 42)
 
        # -----------------------------------------
        # TimeSeriesSplit Cross-Validation (5 Splits)
        # -----------------------------------------
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        from sklearn.linear_model import LinearRegression
 
        print("\n" + "=" * 55)
        print("  [TimeSeriesSplit Cross-Validation (5 Splits) on Train Set]")
        print("=" * 55)
 
        tscv = TimeSeriesSplit(n_splits=5)
        cv_metrics = {
            "rmse": [], "r2": [], "mae": [], "mse": []
        }
 
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_sc)):
            X_tr, X_val = X_train_sc[train_idx], X_train_sc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
 
            fold_model = LinearRegression().fit(X_tr, y_tr)
            fold_pred = fold_model.predict(X_val)
 
            fold_r2 = r2_score(y_val, fold_pred)
            fold_mae = mean_absolute_error(y_val, fold_pred)
            fold_mse = mean_squared_error(y_val, fold_pred)
            fold_rmse = np.sqrt(fold_mse)
 
            cv_metrics["r2"].append(fold_r2)
            cv_metrics["mae"].append(fold_mae)
            cv_metrics["mse"].append(fold_mse)
            cv_metrics["rmse"].append(fold_rmse)
 
            # MLflow - 폴드별 성능 기록
            mlflow.log_metric(f"fold_{fold+1}_r2", fold_r2)
            mlflow.log_metric(f"fold_{fold+1}_mae", fold_mae)
            mlflow.log_metric(f"fold_{fold+1}_mse", fold_mse)
            mlflow.log_metric(f"fold_{fold+1}_rmse", fold_rmse)
 
            print(f"    * Fold {fold+1} | Train: {len(X_tr)} months, Val: {len(X_val)} months | Val R2: {fold_r2:.4f} | Val MAE: {fold_mae:.4f}% | Val MSE: {fold_mse:.6f} | Val RMSE: {fold_rmse:.4f}")
 
        print("-" * 55)
        print("  --> Mean CV Metrics:")
        for k in cv_metrics.keys():
            mean_val = np.mean(cv_metrics[k])
            mlflow.log_metric(f"cv_mean_{k}", mean_val)
            print(f"      * {k:<10}: {mean_val:.4f}")
        print("=" * 55 + "\n")
 
        # Train final ensemble model
        ensemble = RealEstateEnsembleRegressor(random_state=42)
        ensemble.fit(X_train_sc, y_train)
 
        # MLflow - 최종 모델 성능 기록
        final_pred = ensemble.predict(X_train_sc)
        final_r2  = r2_score(y_train, final_pred)
        final_mae = mean_absolute_error(y_train, final_pred)
        final_mse = mean_squared_error(y_train, final_pred)
        final_rmse = np.sqrt(final_mse)
        
        mlflow.log_metric("train_r2", final_r2)
        mlflow.log_metric("train_mae", final_mae)
        mlflow.log_metric("train_mse", final_mse)
        mlflow.log_metric("train_rmse", final_rmse)

        # MySQL DB에 성능 지표 및 최신 예측 데이터 추가 적재 (하드코딩 없음, run_id 완벽 동기화)
        latest_predicted_value = float(ensemble.predict(X_train_sc[[-1]])[0])
        
        import uuid
        try:
            active_run = mlflow.active_run()
            run_id_val = active_run.info.run_id if active_run else uuid.uuid4().hex[:32]
        except Exception:
            run_id_val = uuid.uuid4().hex[:32]

        save_performance_to_mysql(final_rmse, final_r2, final_mae, final_mse, run_id=run_id_val)
        save_prediction_to_mysql(predicted_value=latest_predicted_value, run_id=run_id_val)
 
        # Setup directories and save
        models_dir = os.path.join(base_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
 
        model_path    = os.path.join(models_dir, 'ensemble_model.pkl')
        scaler_path   = os.path.join(models_dir, 'scaler.pkl')
        features_path = os.path.join(models_dir, 'selected_features.pkl')
 
        with open(model_path, 'wb') as f:
            pickle.dump(ensemble, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        with open(features_path, 'wb') as f:
            pickle.dump(selected_features, f)
 
        # Also save features as readable text
        txt_features_path = os.path.join(models_dir, 'selected_features.txt')
        with open(txt_features_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(selected_features))
 
        # MLflow - 모델 저장 (MinIO artifact)
        mlflow.sklearn.log_model(ensemble, "ensemble_model")
 
        print("=" * 55)
        print("Training Pipeline Completed Successfully!")
        print("=" * 55)
        print(f"  Saved Model   : {model_path}")
        print(f"  Saved Scaler  : {scaler_path}")
        print(f"  Saved Features: {features_path} and .txt")
        print(f"  Features size : {len(selected_features)}")
 
if __name__ == '__main__':
    run_train()