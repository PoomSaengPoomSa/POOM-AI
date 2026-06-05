"""
CatBoost 하이퍼파라미터 Optuna 튜닝 스크립트
- TimeSeriesSplit 교차검증으로 시계열 데이터 누수 방지
- 최적 파라미터를 찾아 model.py에 적용할 수 있도록 출력
"""
import sys
import os
import numpy as np
import optuna
import warnings
warnings.filterwarnings('ignore')

# Optuna 로그 레벨 낮추기
optuna.logging.set_verbosity(optuna.logging.WARNING)

from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
from utils.preprocess import preprocess_data

def objective(trial, X_train, y_train):
    params = {
        'iterations':     trial.suggest_int('iterations', 100, 800),
        'learning_rate':  trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'depth':          trial.suggest_int('depth', 3, 6),
        'l2_leaf_reg':    trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'subsample':      trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        'random_seed': 42,
        'verbose': 0
    }

    tscv = TimeSeriesSplit(n_splits=5)
    rmse_scores = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        # 최소 학습 샘플 보장
        if len(X_tr) < 10:
            continue

        model = CatBoostRegressor(**params)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        rmse_scores.append(rmse)

    return np.mean(rmse_scores) if rmse_scores else float('inf')


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, base_dir)

    print("=" * 55)
    print("  CatBoost Optuna 하이퍼파라미터 튜닝 시작")
    print("=" * 55)

    # 전처리 (valid_mode=False → train+valid 전부 학습 데이터로 사용)
    print("\n[1] 데이터 전처리 중...")
    data = preprocess_data(vif_threshold=20.0, valid_mode=False)
    if data is None:
        print("[ERROR] 전처리 실패. 종료합니다.")
        return

    X_train = data['X_train_sc']
    y_train = data['y_train']
    X_test  = data['X_test_sc']
    y_test  = data['y_test']

    print(f"   Train: {X_train.shape[0]}개 샘플, {X_train.shape[1]}개 피처")
    print(f"   Test : {X_test.shape[0]}개 샘플")

    # Optuna 스터디 실행
    n_trials = 80
    print(f"\n[2] Optuna 최적화 시작 ({n_trials} trials, TimeSeriesSplit 5-fold)...")
    study = optuna.create_study(direction='minimize', study_name='catboost_realestate')
    study.optimize(lambda trial: objective(trial, X_train, y_train),
                   n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_cv_rmse = study.best_value

    print(f"\n[3] 최적 파라미터 (CV RMSE: {best_cv_rmse:.4f})")
    for k, v in best_params.items():
        print(f"   {k}: {v}")

    # 최적 파라미터로 최종 평가
    print("\n[4] 최적 파라미터로 Test 세트 최종 평가...")
    final_cat = CatBoostRegressor(**best_params, random_seed=42, verbose=0)
    final_cat.fit(X_train, y_train)
    test_pred = final_cat.predict(X_test)
    test_r2   = r2_score(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

    print(f"\n{'=' * 55}")
    print(f"  [최적 CatBoost] Test R²: {test_r2:.4f}  RMSE: {test_rmse:.4f}")
    print(f"  [기존 CatBoost] Test R²: 0.1922         RMSE: 0.1047")
    print(f"  [기존 Ensemble] Test R²: 0.5232         RMSE: 0.0805")
    print(f"{'=' * 55}")

    print("\n[5] model.py에 적용할 CatBoostRegressor 파라미터:")
    print(f"""
    self.cat = CatBoostRegressor(
        iterations={best_params['iterations']},
        learning_rate={best_params['learning_rate']:.5f},
        depth={best_params['depth']},
        l2_leaf_reg={best_params['l2_leaf_reg']:.4f},
        subsample={best_params['subsample']:.4f},
        colsample_bylevel={best_params['colsample_bylevel']:.4f},
        random_seed=random_state,
        verbose=0
    )
    """)


if __name__ == '__main__':
    main()
