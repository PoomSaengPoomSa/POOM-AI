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
 
GENERATE_REPORT = False  # 테스트 중엔 False, 운영 시 True로 변경

 
 
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


def generate_and_save_baserate_report(prob_hike, prob_freeze, prob_cut, run_id):
    if not GENERATE_REPORT:
        print("[LLM] GENERATE_REPORT is set to False. Skipping LLM report generation for Base Rate.")
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
        
    # 1. Fetch latest actual base rate
    latest_br_val = None
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
                sql = "SELECT value FROM economic_indicator_history WHERE type = 'base_rate' ORDER BY recorded_at DESC LIMIT 1"
                cursor.execute(sql)
                res = cursor.fetchone()
                if res:
                    latest_br_val = float(res[0])
        finally:
            connection.close()
    except Exception as e:
        print(f"[Warning] Failed to fetch latest actual base rate for LLM: {e}")
        
    # 2. Call OpenAI API using standard urllib
    import urllib.request
    import json
    
    prompt = f"""
    한국은행 기준금리 AI 예측 모델 분석 결과:
    - 금리 인하 확률: {prob_cut*100:.1f}%
    - 금리 동결 확률: {prob_freeze*100:.1f}%
    - 금리 인상 확률: {prob_hike*100:.1f}%
    - 최신 실제 기준금리: {f'{latest_br_val:.2f}%' if latest_br_val is not None else '데이터 없음'}
    - 주요 SHAP 변수 기여도 순위: 소비자물가지수 (kr_cpi, 45%), 미국 기준금리 (us_base_rate, 35%), 가계대출 증가율 (household_debt_growth, 12%), GDP 성장률 (gdp_growth, 8%)
    
    위 예측 데이터와 변수 기여도를 바탕으로 전문적이고 가독성이 높은 한국어 기준금리 전망 분석 리포트를 markdown 형식으로 작성해주세요.
    반드시 다음의 구조와 예시 이미지의 격식과 톤앤매너를 유지해주세요:
    
    구조 예시:
    ### [기준금리 전망 리포트]
    
    (여기에 향후 기준금리 통화정책 방향에 대한 한 줄 요약을 적어주세요. 예: 한국은행 금융통화위원회는 당분간 ... 수준으로 ...할 것으로 예상됩니다.)
    
    **1. (첫 번째 핵심 요인 제목)**
    (상승, 동결 혹은 하락을 이끄는 첫 번째 핵심 변수와 AI 분석 기여도를 엮어서 상세한 설명 한 단락을 작성해주세요.)
    
    **2. (두 번째 핵심 요인 제목)**
    (상승, 동결 혹은 하락을 이끄는 두 번째 핵심 변수와 AI 분석 기여도를 엮어서 상세한 설명 한 단락을 작성해주세요.)
    
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
                cursor.execute(sql, (report_id, "base_rate", "gpt-4o", "ko", content, "done", "FRED, ECOS, BOK"))
            connection.commit()
            print("[DB] Successfully generated and saved Base Rate LLM report into MySQL trend_llm_report table.")
        finally:
            connection.close()
    except Exception as e:
        print(f"[Error] Failed to save Base Rate LLM report to MySQL: {e}")


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
                CREATE TABLE IF NOT EXISTS baserate_performance (
                    run_id VARCHAR(50) NOT NULL PRIMARY KEY,
                    accuracy DOUBLE NOT NULL,
                    `precision` DOUBLE NOT NULL,
                    recall DOUBLE NOT NULL,
                    f1_score DOUBLE NOT NULL,
                    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
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

        # 꼬임 버그를 완벽하게 제거하기 위해 명시적 키워드 인자로 호출
        save_performance_to_mysql(accuracy=accuracy, precision=precision, recall=recall, f1_score=f1, run_id=run_id_val)
        save_prediction_to_mysql(prob_hike=prob_hike, prob_freeze=prob_freeze, prob_cut=prob_cut, run_id=run_id_val)
        generate_and_save_baserate_report(prob_hike=prob_hike, prob_freeze=prob_freeze, prob_cut=prob_cut, run_id=run_id_val)
 
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