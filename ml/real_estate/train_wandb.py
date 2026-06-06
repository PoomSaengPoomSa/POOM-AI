import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
import pymysql
import wandb
from dotenv import load_dotenv, find_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# model.py에서 모델 클래스 로드
from model import RealEstateEnsembleRegressor

# Windows cp949 인코딩 문제 방지
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# 글로벌 최적 모델 추적 변수
best_val_mse = float('inf')
best_weights = None
best_metrics = None

# 캐싱할 데이터 행렬 (글로벌 참조로 스윕 에이전트 내에서 초고속 사용)
val_pred_matrix = None
test_pred_matrix = None
y_val_arr = None
y_test_arr = None
model_names = []


def load_data_from_mysql():
    load_dotenv(find_dotenv())
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    DB_NAME = os.getenv('DB_NAME')
    
    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
        raise ValueError("Missing database credentials in .env file.")
        
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
    return df


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return rmse, r2, mae, mse


def sweep_train():
    global best_val_mse, best_weights, best_metrics
    global val_pred_matrix, test_pred_matrix, y_val_arr, y_test_arr, model_names
    
    # 1. WandB Run 초기화
    run = wandb.init()
    config = wandb.config
    
    # 2. 하이퍼파라미터 가중치 읽기 및 L1 정규화
    w = np.array([
        config.w_ridge,
        config.w_rf,
        config.w_cat,
        config.w_xgb,
        config.w_lgb
    ])
    
    w_sum = w.sum()
    if w_sum == 0:
        w_norm = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    else:
        w_norm = w / w_sum
        
    # 3. 캐싱된 개별 모델 예측값을 기반으로 가중합 예측 수행 (오직 Valid 셋 기준)
    y_val_pred = np.dot(w_norm, val_pred_matrix)
    val_rmse, val_r2, val_mae, val_mse = evaluate(y_val_arr, y_val_pred)
    
    # 4. 성능 로깅 (Test 메트릭은 튜닝 최적화에 절대 관여하지 않음)
    metrics_to_log = {
        "val_mse": val_mse,
        "val_rmse": val_rmse,
        "val_mae": val_mae,
        "val_r2": val_r2
    }
    
    # 참고를 위해 정규화된 각 가중치도 기록
    for name, weight in zip(model_names, w_norm):
        metrics_to_log[f"norm_w_{name}"] = weight
        
    wandb.log(metrics_to_log)
    
    # 5. 글로벌 최적 상태 업데이트 (Valid MSE 기준 최소화)
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_weights = w_norm.tolist()
        best_metrics = {
            "val_mse": val_mse,
            "val_rmse": val_rmse,
            "val_mae": val_mae,
            "val_r2": val_r2
        }
        
    run.finish()


def main():
    global val_pred_matrix, test_pred_matrix, y_val_arr, y_test_arr, model_names
    global best_val_mse, best_weights, best_metrics
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(find_dotenv())
    
    # 1. WandB 로그인
    wandb_key = os.getenv("WANDB_API_KEY")
    if not wandb_key:
        print("[Error] WANDB_API_KEY not found in .env file. Please check .env settings.")
        sys.exit(1)
        
    try:
        wandb.login(key=wandb_key)
        print("[WandB] Login successful!")
    except Exception as e:
        print(f"[Error] Failed to login to WandB: {e}")
        sys.exit(1)

    # 2. 데이터 로드 및 3분할 (Train / Valid / Test)
    df = load_data_from_mysql()
    df['date_ym'] = df['date_ym'].astype(str).str.strip()
    
    cfg = RealEstateEnsembleRegressor
    
    train_mask = df['date_ym'] <= cfg.TRAIN_END
    valid_mask = (df['date_ym'] >= cfg.VALID_START) & (df['date_ym'] <= cfg.VALID_END)
    test_mask  = (df['date_ym'] >= cfg.TEST_START) & (df['date_ym'] <= cfg.TEST_END)
    
    train_df = df[train_mask].copy()
    valid_df = df[valid_mask].copy()
    test_df  = df[test_mask].copy()
    
    drop_cols = [c for c in cfg.DROP_COLS if c in df.columns]
    X_train = train_df.drop(columns=drop_cols)
    X_valid = valid_df.drop(columns=drop_cols)
    X_test  = test_df.drop(columns=drop_cols)
    y_train = train_df['next_change_rate']
    y_valid = valid_df['next_change_rate']
    y_test  = test_df['next_change_rate']
    
    selected_features = list(X_train.columns)
    
    print("\n" + "=" * 65)
    print("부동산 가격지수 예측 데이터 3분할 완료")
    print("=" * 65)
    print(f"  * Train 기간 : {train_df['date_ym'].min()} ~ {train_df['date_ym'].max()} ({len(X_train)}개월)")
    print(f"  * Valid 기간 : {valid_df['date_ym'].min()} ~ {valid_df['date_ym'].max()} ({len(X_valid)}개월)")
    print(f"  * Test 기간  : {test_df['date_ym'].min()} ~ {test_df['date_ym'].max()} ({len(X_test)}개월)")
    print(f"  * 피처 개수  : {len(selected_features)}개")
    print("=" * 65 + "\n")
    
    # 3. 피처 스케일링 (Data Leakage 방지를 위해 Train 기준으로만 적합)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_valid_sc = scaler.transform(X_valid)
    X_test_sc  = scaler.transform(X_test)
    
    # 4. 5종 모델 최초 1회 선행 피팅
    ensemble = RealEstateEnsembleRegressor(random_state=42)
    ensemble.fit(X_train_sc, y_train)
    
    # 5. 개별 모델 예측값 획득 및 글로벌 변수 캐싱
    ind_preds_train = ensemble.get_individual_predictions(X_train_sc)
    ind_preds_valid = ensemble.get_individual_predictions(X_valid_sc)
    ind_preds_test  = ensemble.get_individual_predictions(X_test_sc)
    
    model_names = list(ensemble.models.keys())
    
    # 개별 모델 단독 성능 출력 및 WandB 로깅용 딕셔너리 준비
    print("\n" + "=" * 65)
    print("개별 모델 단독 Valid / Test 성능")
    print("=" * 65)
    
    single_model_results = {}
    for name in model_names:
        v_pred = ind_preds_valid[name]
        t_pred = ind_preds_test[name]
        
        v_rmse, v_r2, v_mae, v_mse = evaluate(y_valid, v_pred)
        t_rmse, t_r2, t_mae, t_mse = evaluate(y_test, t_pred)
        
        single_model_results[name] = {
            "val_rmse": v_rmse, "val_r2": v_r2, "val_mae": v_mae, "val_mse": v_mse,
            "test_rmse": t_rmse, "test_r2": t_r2, "test_mae": t_mae, "test_mse": t_mse
        }
        print(f"  * {name:<15} -> Valid MAE: {v_mae:.4f}% (MSE: {v_mse:.6f}) | Test MAE: {t_mae:.4f}% (MSE: {t_mse:.6f})")
    print("=" * 65 + "\n")
    
    # 스윕 에이전트 연산용 글로벌 캐싱 변수 매핑
    val_pred_matrix = np.array([ind_preds_valid[name] for name in model_names])  # (5, N_val)
    test_pred_matrix = np.array([ind_preds_test[name] for name in model_names])   # (5, N_test)
    y_val_arr = y_valid.values
    y_test_arr = y_test.values
    
    # 6. WandB Sweep 설정 (Bayesian Optimization)
    sweep_config = {
        "method": "bayes",
        "metric": {
            "name": "val_mse",
            "goal": "minimize"
        },
        "parameters": {
            "w_ridge": {"min": 0.0, "max": 1.0},
            "w_rf": {"min": 0.0, "max": 1.0},
            "w_cat": {"min": 0.0, "max": 1.0},
            "w_xgb": {"min": 0.0, "max": 1.0},
            "w_lgb": {"min": 0.0, "max": 1.0}
        }
    }
    
    # Sweep 생성
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project="real_estate_house_price"
    )
    
    # Sweep Agent 실행 (50회 반복 탐색)
    print("WandB Sweep Bayesian Optimization 시작...")
    wandb.agent(
        sweep_id=sweep_id,
        function=sweep_train,
        count=50
    )
    
    # 7. 최적 앙상블 블렌드 가중치 검증 결과 요약
    if best_weights is not None:
        best_weight_dict = {name: round(w, 4) for name, w in zip(model_names, best_weights)}
        
        # 최적 가중치로 Test 세트 최종 1회 평가 수행
        best_w_arr = np.array(best_weights)
        y_test_pred = np.dot(best_w_arr, test_pred_matrix)
        t_rmse, t_r2, t_mae, t_mse = evaluate(y_test_arr, y_test_pred)
        
        print("\n" + "=" * 65)
        print("🏆 WandB Sweep Bayes 최적 앙상블 블렌드 결과")
        print("=" * 65)
        for name, weight in best_weight_dict.items():
            print(f"  - {name:<15} 가중치: {weight:.4f}")
        print("-" * 65)
        print(f"  * [Validation] MSE: {best_val_mse:.6f} | RMSE: {best_metrics['val_rmse']:.4f} | MAE: {best_metrics['val_mae']:.4f}%")
        print(f"  * [Test (최종)] MSE: {t_mse:.6f} | RMSE: {t_rmse:.4f} | R2: {t_r2:.4f} | MAE: {t_mae:.4f}%")
        print("=" * 65 + "\n")
        
        # 8. 최종 결과 JSON 저장
        models_dir = os.path.join(base_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        weights_save_path = os.path.join(models_dir, 'best_ensemble_weights.json')
        
        with open(weights_save_path, 'w', encoding='utf-8') as f:
            json.dump(best_weight_dict, f, indent=4, ensure_ascii=False)
            
        print(f"[OK] 최적 앙상블 가중치를 파일로 저장하였습니다: {weights_save_path}")
        
    else:
        print("[Warning] Sweep did not yield any valid weights.")


if __name__ == '__main__':
    main()
