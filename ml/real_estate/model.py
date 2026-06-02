import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

class RealEstateEnsembleRegressor(BaseEstimator, RegressorMixin):
    """
    부동산 매매가격지수 변동률 예측을 위한 가중 앙상블 회귀 모델
    """

    # 학습에서 제외할 컬럼
    DROP_COLS = [
        'date_ym', 'next_change_rate', 'next_house_price_idx',
        'house_price_idx', 'kr_cpi', 'kr_unemployment', 
        'kr_base_rate', 'kr_mortgage_rate', 'kospi200', 
        'apt_trade_count', 'kr_m2', 'buyer_dominance'
    ]

    # 시계열 분할 기준
    TRAIN_END  = '202504'   # Train: ~2025.04
    TEST_START = '202505'   # Test:  2025.05~

    def __init__(self, random_state=42):
        self.random_state = random_state
        
        # 1. Tuned Linear Model (L2 Regularized Ridge)
        self.ridge = Ridge(
            alpha=1.0
        )
        
        # 2. Robust Bagging Model
        self.rf = RandomForestRegressor(
            n_estimators=150,
            max_depth=4,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=random_state,
            n_jobs=-1
        )
        
        # 3. Robust Symmetric Boosting Model
        self.cat = CatBoostRegressor(
            iterations=150,
            learning_rate=0.03,
            depth=4,
            l2_leaf_reg=4.0,
            random_seed=random_state,
            verbose=0
        )
        
        self.models = {
            'RidgeRegressor': self.ridge,
            'RandomForest': self.rf,
            'CatBoost': self.cat
        }
        
    def fit(self, X, y):
        print("\n  [Production-Grade 3-Model Ensemble Training]")
        for name, model in self.models.items():
            print(f"    - Training {name}...")
            model.fit(X, y)
        print("  Ensemble training complete!")
        return self
        
    def predict(self, X):
        # Weighted average of the Top 3 Champions:
        # RidgeRegressor (60%), RandomForest (20%), CatBoost (20%)
        # Sum of weights = 1.0.
        preds = []
        weights = [0.60, 0.20, 0.20]
        
        preds.append(self.ridge.predict(X) * weights[0])
        preds.append(self.rf.predict(X) * weights[1])
        preds.append(self.cat.predict(X) * weights[2])
        
        return np.sum(preds, axis=0)
        
    def get_individual_predictions(self, X):
        preds = {}
        for name, model in self.models.items():
            preds[name] = model.predict(X)
        return preds


if __name__ == '__main__':
    builder = RealEstateEnsembleRegressor()
    print("=" * 50)
    print("✅ 부동산 앙상블 회귀 모델 생성 완료!")
    print("=" * 50)
    print(f"   구성 모델: Ridge, RandomForest, CatBoost")
    print(f"\n   Train 기간: ~{builder.TRAIN_END}")
    print(f"   Test 기간 : {builder.TEST_START}~")
