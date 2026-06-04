import xgboost as xgb

class GoldModel:
    """
    금값 상승/하락 방향 예측을 위한 XGBoost 이진 분류 모델
    - 클래스: 하락/보합(0), 상승(1)
    """

    # 학습에서 제외할 컬럼
    DROP_COLS = ['loaded_date', 'target_tomorrow_gold_change_rate', 'target_tomorrow_gold_direction']

    # 시계열 분할 기준 (특정 시점 고정 분할)
    TRAIN_END  = '2025-04-30'
    TEST_START = '2025-05-01'

    def __init__(self, random_state=42, scale_pos_weight=1.0):
        self.random_state = random_state
        self.scale_pos_weight = scale_pos_weight
        self.classifier = self._build_classifier()

    def _build_classifier(self):
        """그리드 서치로 튜닝된 최적 하이퍼파라미터 적용 XGBoost 모델 생성"""
        return xgb.XGBClassifier(
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

    def get_classifier(self):
        return self.classifier
