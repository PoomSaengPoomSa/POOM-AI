import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from catboost import CatBoostRegressor

class RealEstateEnsembleRegressor(BaseEstimator, RegressorMixin):
    # 시계열 분할 기준 (특정 시점 고정 분할)
    TRAIN_END  = '202504'
    TEST_START = '202505'

    def __init__(self, random_state=42):
        self.random_state = random_state
        
        # We select only the Top 3 Champion Models from the tournament to maximize performance:
        # 1. Supreme Bagging Model (ExtraTrees)
        self.et = ExtraTreesRegressor(
            n_estimators=150,
            max_depth=4,
            random_state=random_state,
            n_jobs=-1
        )
        
        # 2. Robust Bagging Model (RandomForest)
        self.rf = RandomForestRegressor(
            n_estimators=150,
            max_depth=4,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=random_state,
            n_jobs=-1
        )
        
        # 3. Robust Symmetric Boosting Model (CatBoost)
        self.cat = CatBoostRegressor(
            iterations=150,
            learning_rate=0.03,
            depth=4,
            l2_leaf_reg=4.0,
            random_seed=random_state,
            verbose=0
        )
        
        self.models = {
            'ExtraTrees': self.et,
            'RandomForest': self.rf,
            'CatBoost': self.cat
        }
        
    def fit(self, X, y):
        print("\n  [Production-Grade 3-Model Ensemble Training (Tree Dream Team)]")
        for name, model in self.models.items():
            print(f"    - Training {name}...")
            model.fit(X, y)
        print("  Ensemble training complete!")
        return self
        
    def predict(self, X):
        # Weighted average of the Top 3 Champions:
        # ExtraTrees (40%), RandomForest (40%), CatBoost (20%)
        # Sum of weights = 1.0.
        preds = []
        weights = [0.40, 0.40, 0.20]
        
        preds.append(self.et.predict(X) * weights[0])
        preds.append(self.rf.predict(X) * weights[1])
        preds.append(self.cat.predict(X) * weights[2])
        
        return np.sum(preds, axis=0)
        
    def get_individual_predictions(self, X):
        preds = {}
        for name, model in self.models.items():
            preds[name] = model.predict(X)
        return preds
