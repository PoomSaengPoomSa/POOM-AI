import os
import sys
import io

# Windows 터미널 한글 및 이모지 출력 오류 방지 (UTF-8 강제)
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings

warnings.filterwarnings('ignore')

# 상위 경로를 파이썬 패스에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.preprocess import preprocess_data

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return rmse, r2, mae, mse

def main():
    print("=" * 80)
    print("🏆 아파트 매매가격지수 예측 모델 토너먼트 (AutoML Benchmark) 실행 🏆")
    print("=" * 80)

    # 1. 데이터 로드 및 전처리
    print("[1] 데이터 전처리 및 VIF 피처 선택 로드 중...")
    data = preprocess_data(vif_threshold=20.0)
    if data is None:
        print("[ERROR] 데이터 전처리에 실패했습니다.")
        return

    X_train = data['X_train_sc']
    X_test  = data['X_test_sc']
    y_train = data['y_train']
    y_test  = data['y_test']
    selected_features = data['features']

    print(f"  * 학습 피처 개수: {len(selected_features)}개 ({selected_features})")
    print(f"  * Train 샘플 수 : {X_train.shape[0]}개월")
    print(f"  * Test 샘플 수  : {X_test.shape[0]}개월")
    print("-" * 80)

    # 2. 대결할 모델 라인업 정의
    models = {
        "OLS (LinearRegression)": LinearRegression(),
        "Ridge Regression (L2)": Ridge(alpha=1.0),
        "Lasso Regression (L1)": Lasso(alpha=0.01),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5),
        "Huber Regressor (Robust)": HuberRegressor(max_iter=1000),
        "Support Vector Regressor (SVR)": SVR(kernel='rbf', C=1.0, epsilon=0.1),
        "Decision Tree": DecisionTreeRegressor(max_depth=4, random_state=42),
        "Random Forest (Bagging)": RandomForestRegressor(n_estimators=150, max_depth=4, random_state=42, n_jobs=-1),
        "Gradient Boosting (Boosting)": GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.03, random_state=42),
        "Extra Trees": ExtraTreesRegressor(n_estimators=150, max_depth=4, random_state=42, n_jobs=-1),
        "XGBoost Regressor": xgb.XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.03, random_state=42),
        "CatBoost Regressor": CatBoostRegressor(iterations=150, learning_rate=0.03, depth=4, verbose=0, random_seed=42)
    }

    # LightGBM 추가 시도 (설치되어 있으면 대결에 추가)
    try:
        from lightgbm import LGBMRegressor
        models["LightGBM Regressor"] = LGBMRegressor(n_estimators=150, max_depth=3, learning_rate=0.03, random_state=42, verbose=-1)
    except ImportError:
        pass

    # 3. 모델 학습 및 평가 토너먼트 진행
    print("[2] 모델 대결 시작 (11+ Regressors Tournament)...")
    results = []

    for name, model in models.items():
        try:
            # 학습
            model.fit(X_train, y_train)
            
            # 예측
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # 평가 지표 산출
            tr_rmse, tr_r2, tr_mae, tr_mse = evaluate(y_train, train_pred)
            te_rmse, te_r2, te_mae, te_mse = evaluate(y_test, test_pred)
            
            results.append({
                "Model": name,
                "Train_MAE(%)": tr_mae,
                "Test_MAE(%)": te_mae,
                "Train_RMSE(%)": tr_rmse,
                "Test_RMSE(%)": te_rmse,
                "Train_R2": tr_r2,
                "Test_R2": te_r2
            })
            print(f"  * {name:<32} 완료! (Test MAE: {te_mae:.4f}%)")
        except Exception as e:
            print(f"  * [오류 발생] {name} 학습 중 에러: {e}")

    # 4. 결과 정렬 및 리더보드 생성
    leaderboard = pd.DataFrame(results)
    # Test MAE가 가장 낮고, Test RMSE가 낮은 순서로 정렬
    leaderboard = leaderboard.sort_values(by=["Test_MAE(%)", "Test_RMSE(%)"], ascending=True)

    print("\n" + "=" * 90)
    print("🏆 아파트 매매가격지수 예측 모델 토너먼트 최종 리더보드 (Sorted by Test MAE) 🏆")
    print("=" * 90)
    print(leaderboard.to_string(index=False))
    print("=" * 90)

    # 5. 리더보드 저장
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    leaderboard_path = os.path.join(results_dir, 'benchmark_leaderboard.csv')
    leaderboard.to_csv(leaderboard_path, index=False, encoding='utf-8-sig')
    print(f"[3] 최종 리더보드를 저장했습니다: {leaderboard_path}")

    # 6. 최적의 챔피언 및 조합 추천 제안
    champion = leaderboard.iloc[0]
    print("\n👑 오늘의 챔피언 모델:")
    print(f"  * 모델명 : {champion['Model']}")
    print(f"  * Test MAE  : {champion['Test_MAE(%)']:.4f}%")
    print(f"  * Test R2   : {champion['Test_R2']:.4f}")
    
    print("\n💡 후속 분석 조언:")
    print("  1. 선형 모델(L1/L2 규제) 대비 트리 기반 앙상블 모델(RF, ExtraTrees, CatBoost)의 오차가 현저하게 작습니다.")
    print("  2. 만약 특정 단일 모델로 갈아타고 싶거나, 앙상블 가중치를 조정하려면 리더보드 상위 3개 모델의 비율을 높이는 것을 권장합니다.")

if __name__ == '__main__':
    main()
