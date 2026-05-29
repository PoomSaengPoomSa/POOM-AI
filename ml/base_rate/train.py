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


def save_prediction_to_mysql(prob_hike, prob_freeze, prob_cut, run_id):
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
            print("[DB] Successfully saved baserate_predictions (1 row) into MySQL.")
        finally:
            connection.close()
    except Exception as e:
        print(f"[Error] Failed to save baserate predictions to MySQL: {e}")


def save_performance_to_mysql(precision, f1_score, accuracy, recall, run_id=None):
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
                INSERT INTO baserate_performance (run_id, accuracy, `precision`, recall, f1_score)
                VALUES (%s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (run_id, accuracy, precision, recall, f1_score))
            connection.commit()
            print("[DB] Successfully saved base_rate performance metrics into MySQL.")
        finally:
            connection.close()
    except Exception as e:
        print(f"[Error] Failed to save performance metrics to MySQL: {e}")


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
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
 
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
        precision = precision_score(y_test_label, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test_label, y_pred, average='weighted', zero_division=0)
 
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        print(f"\n   [Test 성능]")
        print(f"     Accuracy  : {accuracy:.4f}")
        print(f"     F1 Score  : {f1:.4f}")
        print(f"     Precision : {precision:.4f}")
        print(f"     Recall    : {recall:.4f}")

        # MySQL DB에 성능 지표 및 최신 예측 데이터 추가 적재 (하드코딩 없음, run_id 완벽 동기화)
        X_latest = df.drop(columns=drop_cols).iloc[[-1]]
        latest_proba = classifier.predict_proba(X_latest)[0]
        prob_cut = float(latest_proba[0])
        prob_freeze = float(latest_proba[1])
        prob_hike = float(latest_proba[2])
        
        import uuid
        try:
            active_run = mlflow.active_run()
            run_id_val = active_run.info.run_id if active_run else uuid.uuid4().hex[:32]
        except Exception:
            run_id_val = uuid.uuid4().hex[:32]

        save_performance_to_mysql(precision, f1, accuracy, recall, run_id=run_id_val)
        save_prediction_to_mysql(prob_hike=prob_hike, prob_freeze=prob_freeze, prob_cut=prob_cut, run_id=run_id_val)
 
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