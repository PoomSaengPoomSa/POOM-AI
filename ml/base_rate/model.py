import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    GradientBoostingClassifier,
)

class InterestRateEnsembleModel:
    """
    한국 기준금리 변동 예측을 위한 단일 3클래스 분류 모델
    - 클래스: 인하(0), 동결(1), 인상(2)
    """

    # 학습에서 제외할 컬럼
    DROP_COLS = ['date_ym', 'kr_base_rate_change', 'label', 'label_encoded']

    # 시계열 분할 기준
    TRAIN_END  = '202504'   # Train: ~2024.04
    TEST_START = '202505'   # Test:  2025.05~

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.classifier = self._build_classifier()

    def _build_classifier(self):
        """3-class 분류 앙상블: 금리 방향 예측"""

        # 클래스 불균형 대응: 동결이 압도적이므로 소수 클래스에 극단적 가중치 부여
        # 인하=0, 동결=1, 인상=2
        sample_weight_map = {0: 5.0, 1: 1.0, 2: 5.0}

        xgb_clf = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            n_estimators=200,
            learning_rate=0.005,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=2.0,
            scale_pos_weight=1,
            random_state=self.random_state,
            eval_metric='mlogloss',
        )

        cat_clf = CatBoostClassifier(
            iterations=200,
            learning_rate=0.05,
            depth=4,
            loss_function='MultiClass',
            verbose=False,
            random_state=self.random_state
        )

        """lgb_clf = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=3,
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=self.random_state,
            verbose=-1,
        )

        rf_clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=2,
            class_weight=sample_weight_map,
            random_state=self.random_state,
            n_jobs=-1,
        )"""

        ensemble_clf = VotingClassifier(
            estimators=[
                #('cat', cat_clf),
                ('xgb', xgb_clf),
                #('lgb', lgb_clf),
                #('rf', rf_clf),
            ],
            voting='soft',   # 확률 기반 투표
        )
        return ensemble_clf

    def get_classifier(self):
        return self.classifier

if __name__ == '__main__':
    builder = InterestRateEnsembleModel()
    print("=" * 50)
    print("✅ 단일 3클래스 분류 모델 생성 완료!")
    print("=" * 50)
    print(f"   분류 모델: VotingClassifier (XGB, LGB, GB, RF)")
    print(f"\n   Train 기간: ~{builder.TRAIN_END}")
    print(f"   Test 기간 : {builder.TEST_START}~")