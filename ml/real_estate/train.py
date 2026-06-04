import os
import sys
import pickle
import mlflow
import mlflow.sklearn
import numpy as np
import pymysql
from dotenv import load_dotenv, find_dotenv
from utils.preprocess import preprocess_data
from model import RealEstateEnsembleRegressor

# Windows cp949 환경에서 MLflow 이모지 출력 시 UnicodeEncodeError 방지
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


GENERATE_REPORT = True  # 테스트 중엔 False, 운영 시 True로 변경


def save_prediction_to_mysql(predicted_value, predicted_index, run_id):
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
                    predicted_index DOUBLE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                sql = """
                INSERT INTO realestate_predictions (run_id, predicted_value, predicted_index)
                VALUES (%s, %s, %s)
                """
                cursor.execute(sql, (run_id, predicted_value, predicted_index))
            connection.commit()
            print("[DB] Successfully saved realestate_predictions (1 row) into MySQL.")
        finally:
            connection.close()
    except Exception as e:
        print(f"[Error] Failed to save realestate predictions to MySQL: {e}")


def generate_and_save_realestate_report(predicted_value, predicted_index, run_id):
    if not GENERATE_REPORT:
        print("[LLM] GENERATE_REPORT is set to False. Skipping LLM report generation for Real Estate.")
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
        
    # 1. Fetch latest actual realestate index
    re_today = None
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
                sql = "SELECT house_price_idx FROM ml_realestate_raw ORDER BY date_ym DESC LIMIT 1"
                cursor.execute(sql)
                res = cursor.fetchone()
                if res:
                    re_today = float(res[0])
        finally:
            connection.close()
    except Exception as e:
        print(f"[Warning] Failed to fetch latest actual index for LLM: {e}")
        
    # 2. Call OpenAI API using standard urllib
    import urllib.request
    import json
    
    prompt = f"""
    부동산 가격지수 AI 예측 모델 분석 결과:
    - 이번달 실제 가격지수(re_today): {f'{re_today:.2f}' if re_today is not None else '데이터 없음'}
    - 다음달 예측 변동률: {predicted_value:.2f}%
    - 다음달 예측 환산 가격지수(predicted_index): {f'{predicted_index:.2f}' if predicted_index is not None else '데이터 없음'}
    - 주요 SHAP 변수 기여도 순위: 기준금리 (kr_base_rate, 40%), 소비자물가지수 (kr_cpi, 30%), 매수우위지수 (buyer_dominance, 20%), 주택담보대출금리 (kr_mortgage_rate, 10%)
    
    위 예측 데이터와 변수 기여도를 바탕으로 전문적이고 가독성이 높은 한국어 부동산 가격지수 전망 분석 리포트를 markdown 형식으로 작성해주세요.
    반드시 다음의 구조와 예시 이미지의 격식과 톤앤매너를 유지해주세요:
    
    구조 예시:
    ### [부동산 가격지수 분석 리포트]
    
    (여기에 부동산 시장 전망에 대한 한 줄 요약을 적어주세요. 예: 서울 아파트 시장의 회복세를 지지할 것으로 보입니다...)
    
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
                
                # 16-character UUID prefix as report ID
                report_id = f"rpt_{uuid.uuid4().hex[:16]}"
                sql = """
                INSERT INTO trend_llm_report (report_id, type, model_name, language, content, status, created_at, data_source)
                VALUES (%s, %s, %s, %s, %s, %s, NOW(), %s)
                """
                cursor.execute(sql, (report_id, "real_estate", "gpt-4o", "ko", content, "done", "ECOS, K-RealEstate"))
            connection.commit()
            print("[DB] Successfully generated and saved Real Estate LLM report into MySQL trend_llm_report table.")
        finally:
            connection.close()
    except Exception as e:
        print(f"[Error] Failed to save Real Estate LLM report to MySQL: {e}")


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


def run_train():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    load_dotenv(find_dotenv())
 
    # MLflow 설정
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', None))
    mlflow.set_experiment("real_estate")
 
    with mlflow.start_run():
 
        # Preprocess
        data = preprocess_data(vif_threshold=20.0)
        if data is None:
            print("[Error] Preprocessing failed.")
            return
 
        X_train_sc = data['X_train_sc']
        X_test_sc  = data['X_test_sc']
        y_train = data['y_train']
        y_test  = data['y_test']
        selected_features = data['features']
        scaler = data['scaler']
 
        # MLflow - 전처리 파라미터 기록
        train_df = data['train_df']
        test_df = data['test_df']
        mlflow.log_param("train_start", train_df['date_ym'].min())
        mlflow.log_param("train_end", train_df['date_ym'].max())
        mlflow.log_param("test_start", test_df['date_ym'].min())
        mlflow.log_param("test_end", test_df['date_ym'].max())
        mlflow.log_param("vif_threshold", 20.0)
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

        # MLflow - Test set 성능 기록 (hold-out, gold/base_rate와 동일한 방식)
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
        print("  Real Estate Ensemble - Train / Test Performance")
        print("=" * 55)
        print(f"   [Train]  R2: {train_r2:.4f} | MAE: {train_mae:.4f}% | RMSE: {train_rmse:.4f}")
        print(f"   [Test ]  R2: {test_r2:.4f} | MAE: {test_mae:.4f}% | RMSE: {test_rmse:.4f}")
        print("=" * 55 + "\n")

        # MySQL DB에 성능 지표 및 최신 예측 데이터 추가 적재 (하드코딩 없음, run_id 완벽 동기화)
        # 스케일러를 제거했으므로 원본 스케일 최신 피처 값을 그대로 사용하여 예측함
        X_latest = data['df'][selected_features].iloc[[-1]]
        latest_predicted_value = float(ensemble.predict(X_latest.values)[0])
        
        # 이번달 실제 가격지수 조회 및 실질 예측 지수 환산
        re_today = get_latest_actual_realestate_index()
        if re_today is not None:
            predicted_index = re_today * (1 + latest_predicted_value / 100)
            print(f"[Ensemble] Calculated predicted_index: {predicted_index:.4f} using re_today: {re_today} and predicted_value: {latest_predicted_value}%")
        else:
            predicted_index = None
            print("[Warning] Could not calculate predicted_index because re_today is missing.")
            
        import uuid
        try:
            active_run = mlflow.active_run()
            run_id_val = active_run.info.run_id if active_run else uuid.uuid4().hex[:32]
        except Exception:
            run_id_val = uuid.uuid4().hex[:32]

        # 테스트셋 성능을 DB에 저장 (gold/base_rate와 동일한 방식)
        save_performance_to_mysql(rmse=test_rmse, r2_score=test_r2, mae=test_mae, mse=test_mse, run_id=run_id_val)
        save_prediction_to_mysql(predicted_value=latest_predicted_value, predicted_index=predicted_index, run_id=run_id_val)
        generate_and_save_realestate_report(predicted_value=latest_predicted_value, predicted_index=predicted_index, run_id=run_id_val)
 
        # Setup directories and save
        models_dir = os.path.join(base_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
 
        model_path    = os.path.join(models_dir, 'ensemble_model.pkl')
        features_path = os.path.join(models_dir, 'selected_features.pkl')
 
        with open(model_path, 'wb') as f:
            pickle.dump(ensemble, f)
        with open(features_path, 'wb') as f:
            pickle.dump(selected_features, f)
 
        # Also save features as readable text
        txt_features_path = os.path.join(models_dir, 'selected_features.txt')
        with open(txt_features_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(selected_features))
 
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
    run_train()