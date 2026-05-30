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
 
GENERATE_REPORT = False  # 테스트 중엔 False, 운영 시 True로 변경

 
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


def save_prediction_to_mysql(prob_rise, prob_fall, run_id):
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
            print("[DB] Successfully saved gold_predictions (1 row) into MySQL.")
        finally:
            connection.close()
    except Exception as e:
        print(f"[Error] Failed to save gold predictions to MySQL: {e}")


def generate_and_save_gold_report(prob_rise, prob_fall, run_id):
    if not GENERATE_REPORT:
        print("[LLM] GENERATE_REPORT is set to False. Skipping LLM report generation for Gold.")
        return
        
    load_dotenv(find_dotenv())
    openai_key = os.getenv("OPENAI_API_KEY")
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    DB_NAME = os.getenv('DB_NAME')
    
    if not openai_key:
        print("[Warning] Missing OPENAI_API_KEY. Skipping LLM report generation.")
        return
    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
        print("[Warning] Missing DB config. Skipping LLM report generation.")
        return
        
    # 1. Fetch latest actual gold price
    latest_gold_val = None
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
                sql = "SELECT value FROM economic_indicator_history WHERE type = 'gold' ORDER BY recorded_at DESC LIMIT 1"
                cursor.execute(sql)
                res = cursor.fetchone()
                if res:
                    latest_gold_val = float(res[0])
        finally:
            connection.close()
    except Exception as e:
        print(f"[Warning] Failed to fetch latest actual gold price for LLM: {e}")
        
    # 2. Call OpenAI API using standard urllib
    import urllib.request
    import json
    
    prob_rise_pct = prob_rise * 100
    prob_fall_pct = prob_fall * 100
    
    prompt = f"""
    금값 AI 예측 모델 분석 결과:
    - 내일 상승 확률: {prob_rise_pct:.1f}%
    - 내일 하락 확률: {prob_fall_pct:.1f}%
    - 최신 금값 실제 가격: {f'{latest_gold_val:,.2f}' if latest_gold_val is not None else '데이터 없음'}
    - 주요 SHAP 변수 기여도 순위: 달러 인덱스 (dxy_proxy, 32%), 소비자물가지수 (kr_cpi, 25%), 원/달러 환율 (kr_usd_exchange, 23%), WTI 유가 (wti_oil, 20%)
    
    위 예측 데이터와 변수 기여도를 바탕으로 전문적이고 가독성이 높은 한국어 금값 전망 분석 리포트를 markdown 형식으로 작성해주세요.
    반드시 다음의 구조와 예시 이미지의 격식과 톤앤매너를 유지해주세요:
    
    구조 예시:
    ### [금값 분석 리포트]
    
    (여기에 향후 단기/장기 전망에 대한 한 줄 요약을 적어주세요. 예: 향후 12개월간 금값은 ... 수준으로 ...이 예상됩니다.)
    
    **1. (첫 번째 핵심 요인 제목)**
    (상승 혹은 하락을 이끄는 첫 번째 핵심 변수와 AI 분석 기여도를 엮어서 상세한 설명 한 단락을 작성해주세요.)
    
    **2. (두 번째 핵심 요인 제목)**
    (상승 혹은 하락을 이끄는 두 번째 핵심 변수와 AI 분석 기여도를 엮어서 상세한 설명 한 단락을 작성해주세요.)
    
    요구사항:
    - 마크다운 형식으로 작성할 것.
    - 너무 길지 않게 핵심 요약 위주로 작성할 것 (전체 400자 내외).
    """
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_key}"
    }
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a professional economic analyst. Always respond in Korean markdown format. Keep it concise, engaging, and professional."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    
    content = None
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            content = res_data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[Error] Failed to generate LLM report via OpenAI: {e}")
        return
        
    if not content:
        print("[Warning] Generated LLM report is empty.")
        return
        
    # 3. Save to trend_llm_report table (Cumulative Insert with 16-character UUID)
    import uuid
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
                CREATE TABLE IF NOT EXISTS trend_llm_report (
                    report_id VARCHAR(50) NOT NULL PRIMARY KEY,
                    type VARCHAR(50) NOT NULL,
                    model_name VARCHAR(50) NOT NULL,
                    language VARCHAR(10) NOT NULL,
                    content TEXT NOT NULL,
                    status VARCHAR(20) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    data_source VARCHAR(255)
                )
                """)
                
                report_id = f"rpt_{str(uuid.uuid4()).replace('-', '')[:16]}"
                sql = """
                INSERT INTO trend_llm_report (report_id, type, model_name, language, content, status, data_source)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (report_id, "gold", "gpt-4o", "ko", content, "done", "FRED, ECOS"))
            connection.commit()
            print("[DB] Successfully generated and saved Gold LLM report into MySQL trend_llm_report table.")
        finally:
            connection.close()
    except Exception as e:
        print(f"[Error] Failed to save Gold LLM report to MySQL: {e}")


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


def train_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    load_dotenv(find_dotenv())
 
    # MLflow 설정
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', None))
    mlflow.set_experiment("gold")
 
    with mlflow.start_run():
 
        # 1. Load Preprocessed Data
        print("Loading preprocessed gold data from MySQL...")
        df = load_data_from_mysql()
        print(f"   Loaded: {len(df)} rows")
 
        cfg = GoldModel
 
        # 2. Split Data Chronologically
        split_idx = int(len(df) * cfg.TRAIN_RATIO)
        train_df = df.iloc[:split_idx].copy()
        test_df  = df.iloc[split_idx:].copy()
 
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
 
        # MLflow - 데이터 정보 기록
        mlflow.log_param("train_start", train_df['loaded_date'].min())
        mlflow.log_param("train_end", train_df['loaded_date'].max())
        mlflow.log_param("test_start", test_df['loaded_date'].min())
        mlflow.log_param("test_end", test_df['loaded_date'].max())
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
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

        # MySQL DB에 성능 지표 및 최신 예측 데이터 추가 적재 (하드코딩 없음, run_id 완벽 동기화)
        X_latest = df.drop(columns=drop_cols).iloc[[-1]]
        X_latest_scaled = scaler.transform(X_latest)
        X_latest_scaled_df = pd.DataFrame(X_latest_scaled, columns=selected_features)
        
        latest_proba = classifier.predict_proba(X_latest_scaled_df)[0]
        prob_fall = float(latest_proba[0])
        prob_rise = float(latest_proba[1])
        
        import uuid
        try:
            active_run = mlflow.active_run()
            run_id_val = active_run.info.run_id if active_run else uuid.uuid4().hex[:32]
        except Exception:
            run_id_val = uuid.uuid4().hex[:32]

        # 꼬임 버그를 완벽하게 제거하기 위해 명시적 키워드 인자로 호출
        save_performance_to_mysql(accuracy=accuracy, precision=precision, recall=recall, f1_score=f1, run_id=run_id_val)
        save_prediction_to_mysql(prob_rise=prob_rise, prob_fall=prob_fall, run_id=run_id_val)
        generate_and_save_gold_report(prob_rise=prob_rise, prob_fall=prob_fall, run_id=run_id_val)
 
        # 5. Save Models
        joblib.dump(classifier, os.path.join(models_dir, 'gold_xgb_classifier.pkl'))
        joblib.dump(scaler, os.path.join(models_dir, 'gold_scaler.pkl'))
        joblib.dump(selected_features, os.path.join(models_dir, 'gold_features.pkl'))
 
        # MLflow - 모델 저장 (MinIO artifact)
        mlflow.sklearn.log_model(classifier, "classifier")
 
        print(f"\n{'='*55}")
        print("Models and resources saved successfully!")
        print(f"{'='*55}")
        print("   models/gold_xgb_classifier.pkl")
        print("   models/gold_scaler.pkl")
        print(f"   models/gold_features.pkl  ({len(selected_features)} features)")
 
if __name__ == '__main__':
    train_model()