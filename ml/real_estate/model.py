import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from xgboost import XGBRegressor

class RealEstateEnsembleRegressor(BaseEstimator, RegressorMixin):
    # 시계열 분할 기준 (특정 시점 고정 분할)
    TRAIN_END    = '202403'
    VALID_START  = '202404'
    VALID_END    = '202503'
    TEST_START   = '202504'
    TEST_END     = '202603'

    def __init__(self, random_state=42):
        self.random_state = random_state
        
        # 1. Optimal Regularized Tuned XGBoost Model
        self.xgb = XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.01,
            subsample=1.0,
            random_state=random_state,
            n_jobs=-1
        )
        
        # Keep models dictionary to prevent downstream scripts from breaking
        self.models = {
            'XGBoost': self.xgb
        }
        
    def fit(self, X, y):
        print("\n  [Production-Grade Tuned XGBoost Regressor Training]")
        for name, model in self.models.items():
            print(f"    - Training {name}...")
            model.fit(X, y)
        print("  Model training complete!")
        return self
        
    def predict(self, X):
        return self.xgb.predict(X)
        
    def get_individual_predictions(self, X):
        preds = {}
        for name, model in self.models.items():
            preds[name] = model.predict(X)
        return preds


