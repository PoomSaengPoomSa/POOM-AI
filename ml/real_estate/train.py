import os
import pickle
import mlflow
import mlflow.sklearn
import numpy as np
from dotenv import load_dotenv, find_dotenv
from utils.preprocess import preprocess_data
from model import RealEstateEnsembleRegressor
 
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
        from sklearn.metrics import r2_score, mean_absolute_error
        from sklearn.linear_model import LinearRegression
 
        print("\n" + "=" * 55)
        print("  [TimeSeriesSplit Cross-Validation (5 Splits) on Train Set]")
        print("=" * 55)
 
        tscv = TimeSeriesSplit(n_splits=5)
        cv_r2_scores = []
        cv_mae_scores = []
 
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_sc)):
            X_tr, X_val = X_train_sc[train_idx], X_train_sc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
 
            fold_model = LinearRegression().fit(X_tr, y_tr)
            fold_pred = fold_model.predict(X_val)
 
            fold_r2 = r2_score(y_val, fold_pred)
            fold_mae = mean_absolute_error(y_val, fold_pred)
 
            cv_r2_scores.append(fold_r2)
            cv_mae_scores.append(fold_mae)
 
            # MLflow - 폴드별 성능 기록
            mlflow.log_metric(f"fold_{fold+1}_r2", fold_r2)
            mlflow.log_metric(f"fold_{fold+1}_mae", fold_mae)
 
            print(f"    * Fold {fold+1} | Train: {len(X_tr)} months, Val: {len(X_val)} months | Val R2: {fold_r2:.4f} | Val MAE: {fold_mae:.4f}%")
 
        mean_r2  = np.mean(cv_r2_scores)
        mean_mae = np.mean(cv_mae_scores)
        print(f"  --> Mean CV R2  : {mean_r2:.4f}")
        print(f"  --> Mean CV MAE : {mean_mae:.4f}%")
        print("=" * 55 + "\n")
 
        # MLflow - CV 평균 성능 기록
        mlflow.log_metric("cv_mean_r2", mean_r2)
        mlflow.log_metric("cv_mean_mae", mean_mae)
 
        # Train final ensemble model
        ensemble = RealEstateEnsembleRegressor(random_state=42)
        ensemble.fit(X_train_sc, y_train)
 
        # MLflow - 최종 모델 성능 기록
        final_pred = ensemble.predict(X_train_sc)
        final_r2  = r2_score(y_train, final_pred)
        final_mae = mean_absolute_error(y_train, final_pred)
        mlflow.log_metric("train_r2", final_r2)
        mlflow.log_metric("train_mae", final_mae)
 
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