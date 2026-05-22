import os

import pandas as pd

import numpy as np

import joblib

import xgboost as xgb



from model import InterestRateEnsembleModel



import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

warnings.filterwarnings('ignore', category=UserWarning)





# 선택할 피처 수

N_FEATURES = 5





def train_model():

    base_dir = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(base_dir, 'data', 'final_dataset.csv')



    # ═══════════════════════════════════════════════

    # 1. 데이터 로드 & 분할

    # ═══════════════════════════════════════════════

    df = pd.read_csv(data_path, encoding='utf-8-sig')

    df['date_ym'] = df['date_ym'].astype(str).str.strip()

    print(f"📂 데이터 로드 완료: {len(df)}건")



    cfg = InterestRateEnsembleModel



    train_df = df[df['date_ym'] <= cfg.TRAIN_END].copy()

    test_df  = df[df['date_ym'] >= cfg.TEST_START].copy()



    drop_cols = [c for c in cfg.DROP_COLS if c in df.columns]



    X_train_all = train_df.drop(columns=drop_cols)

    X_test_all  = test_df.drop(columns=drop_cols)



    y_train_label  = train_df['label_encoded']

    y_test_label   = test_df['label_encoded']



    print(f"\n{'='*55}")

    print(f"📊 데이터 분할 결과")

    print(f"{'='*55}")

    print(f"   Train: {train_df['date_ym'].min()} ~ {train_df['date_ym'].max()}  ({len(X_train_all)}개월)")

    print(f"   Test : {test_df['date_ym'].min()} ~ {test_df['date_ym'].max()}  ({len(X_test_all)}개월)")

    print(f"   전체 피처 수: {X_train_all.shape[1]}개")



    # Train 라벨 분포

    print(f"\n   [Train 라벨 분포]")

    for lbl_name, lbl_val in [('인하', 0), ('동결', 1), ('인상', 2)]:

        cnt = (y_train_label == lbl_val).sum()

        print(f"     {lbl_name}: {cnt}건 ({cnt/len(y_train_label)*100:.1f}%)")



    # ═══════════════════════════════════════════════

    # 1.5. VIF 기반 다중공선성 제거 (차원 축소)

    # ═══════════════════════════════════════════════

    print(f"\n{'='*55}")

    print(f"📉 VIF 기반 다중공선성 제거 진행 중...")

    print(f"{'='*55}")

    from dimensionality_reduction import calculate_vif_iteratively

   

    # 다중공선성이 높은 변수들을 반복적으로 제거 (단, 목표 N_FEATURES개까지만)

    removed_features, _ = calculate_vif_iteratively(X_train_all, n_features=N_FEATURES, threshold=10.0)

   

    X_train_vif = X_train_all.drop(columns=removed_features)

    X_test_vif  = X_test_all.drop(columns=removed_features)

    print(f"\n   VIF 제거 완료! 남은 피처 수: {X_train_vif.shape[1]}개")



    # ═══════════════════════════════════════════════

    # 2. Feature Selection (20개 Random Forest 기반 중요도 평균)

    # ═══════════════════════════════════════════════

    print(f"\n{'='*55}")

    print(f"🔍 Feature Selection: 20개 Random Forest 앙상블로 상위 {N_FEATURES}개 피처 선택 중...")

    print(f"{'='*55}")



    from sklearn.ensemble import RandomForestClassifier



    importances_list = []

    # 20개의 Random Forest 모델 학습 (VIF로 정제된 데이터 사용)

    for i in range(20):

        rf = RandomForestClassifier(

            n_estimators=100,

            random_state=i,

            max_depth=5,

            class_weight='balanced',

            n_jobs=-1

        )

        rf.fit(X_train_vif, y_train_label)

        importances_list.append(rf.feature_importances_)

       

    # 20개 RF의 피처 중요도 평균 산출

    avg_importances = np.mean(importances_list, axis=0)



    # DataFrame으로 정리 후 정렬

    feat_imp = pd.DataFrame({

        'feature': X_train_vif.columns,

        'importance': avg_importances

    }).sort_values('importance', ascending=False)



    selected_features = feat_imp.head(N_FEATURES)['feature'].tolist()



    print(f"\n   [선택된 {N_FEATURES}개 피처]")

    for i, row in feat_imp.head(N_FEATURES).iterrows():

        print(f"     {row['feature']}: {row['importance']:.4f}")



    # 선택된 피처만으로 데이터 축소

    X_train = X_train_vif[selected_features]

    X_test  = X_test_vif[selected_features]



    print(f"\n   피처 수: {X_train_vif.shape[1]}개 → {X_train.shape[1]}개")



    # ═══════════════════════════════════════════════

    # 3. 방향 분류 학습 (선택된 피처로 최종 모델)

    # ═══════════════════════════════════════════════

    print(f"\n{'='*55}")

    print("🎯 방향 분류 (인하/동결/인상) 모델 학습")

    print(f"{'='*55}")



    builder = InterestRateEnsembleModel(random_state=42)



    from sklearn.utils.class_weight import compute_sample_weight

   

    # 가중치 맵핑 (인하:5, 동결:1, 인상:5)

    sample_weight_map = {0: 5.0, 1: 1.0, 2: 5.0}

    sample_weights = compute_sample_weight(class_weight=sample_weight_map, y=y_train_label)



    classifier = builder.get_classifier()

    # 앙상블 학습 시 fit 단계에서 가중치 부여 (모든 하위 모델에 일괄 적용됨)

    classifier.fit(X_train, y_train_label, sample_weight=sample_weights)



    # ═══════════════════════════════════════════════

    # 4. 모델 저장

    # ═══════════════════════════════════════════════

    models_dir  = os.path.join(base_dir, 'models')

    os.makedirs(models_dir, exist_ok=True)



    # 기존 regressor 모형 삭제(있을 경우)

    reg_path = os.path.join(models_dir, 'regressor.pkl')

    if os.path.exists(reg_path):

        os.remove(reg_path)



    # 모델 저장

    joblib.dump(classifier, os.path.join(models_dir, 'classifier.pkl'))

    joblib.dump(selected_features, os.path.join(models_dir, 'feature_names.pkl'))



    print(f"\n{'='*55}")

    print("💾 모델 저장 완료!")

    print(f"{'='*55}")

    print(f"   models/classifier.pkl")

    print(f"   models/feature_names.pkl  ({N_FEATURES}개 피처)")





if __name__ == '__main__':

    train_model()

