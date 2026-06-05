import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from catboost import CatBoostClassifier

class GoldModel:
    """
    금값 상승/하락 방향 예측을 위한 앙상블(XGBoost + RandomForest + CatBoost) 이진 분류 모델
    - 클래스: 하락/보합(0), 상승(1)
    """

    # 학습에서 제외할 컬럼
    DROP_COLS = ['loaded_date', 'target_tomorrow_gold_change_rate', 'target_tomorrow_gold_direction']

    # 시계열 분할 기준 (특정 시점 고정 분할)
    TRAIN_END    = '2024-03-31'
    VALID_START  = '2024-04-01'
    VALID_END    = '2025-03-31'
    TEST_START   = '2025-04-01'
    TEST_END     = '2026-03-31'

    def __init__(self, random_state=42, scale_pos_weight=1.0):
        self.random_state = random_state
        self.scale_pos_weight = scale_pos_weight
        self.classifier = self._build_classifier()

    def _build_classifier(self):
        """XGBoost, RandomForest, CatBoost로 구성된 Soft Voting 앙상블 분류기 빌드"""
        xgb_clf = xgb.XGBClassifier(
            n_estimators=180,
            learning_rate=0.005,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=self.scale_pos_weight,
            reg_alpha=0.1,
            reg_lambda=0.5,
            random_state=self.random_state,
            eval_metric='logloss'
        )

        class_weight = {0: 1.0, 1: self.scale_pos_weight}
        rf_clf = RandomForestClassifier(
            n_estimators=150,
            max_depth=4,
            class_weight=class_weight,
            random_state=self.random_state
        )

        cat_clf = CatBoostClassifier(
            iterations=150,
            learning_rate=0.01,
            depth=4,
            scale_pos_weight=self.scale_pos_weight,
            random_state=self.random_state,
            verbose=0
        )

        ensemble = VotingClassifier(
            estimators=[
                ('xgb', xgb_clf),
                ('rf', rf_clf),
                ('cat', cat_clf)
            ],
            voting='soft'
        )
        return ensemble

    def get_classifier(self):
        return self.classifier
