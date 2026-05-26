import os
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
import pymysql
from dotenv import load_dotenv

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from model import InterestRateEnsembleModel


k=5

# ── 한글 폰트 설정 ──
font_path = 'C:/Windows/Fonts/malgun.ttf'
if os.path.exists(font_path):
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False


def load_data_from_mysql():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(dotenv_path=os.path.join(base_dir, '.env'))
    
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    DB_NAME = os.getenv('DB_NAME')
    
    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
        raise ValueError("Missing database credentials in .env file.")
        
    DB_PORT = int(DB_PORT)
    
    connection = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    
    try:
        with connection.cursor() as cursor:
            sql = "SELECT * FROM ml_baserate_preprocessed ORDER BY date_ym ASC"
            cursor.execute(sql)
            rows = cursor.fetchall()
    finally:
        connection.close()
        
    df = pd.DataFrame(rows)
    # Convert MySQL decimal objects/others to standard float64/int64 numeric types
    for col in df.columns:
        if col not in ['date_ym', 'label']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def explain_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir  = os.path.join(base_dir, 'models')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # ═══════════════════════════════════════════════
    # 1. 모델 & 데이터 로드 (MySQL Database)
    # ═══════════════════════════════════════════════
    print("[XAI] 분류 모델 로드 중...")
    classifier = joblib.load(os.path.join(models_dir, 'classifier.pkl'))

    df = load_data_from_mysql()
    df['date_ym'] = df['date_ym'].astype(str).str.strip()



    cfg = InterestRateEnsembleModel

    feature_names = joblib.load(os.path.join(models_dir, 'feature_names.pkl'))



    test_df  = df[df['date_ym'] >= cfg.TEST_START].copy()

    X_test  = test_df[feature_names]



    meta_path = os.path.join(base_dir, 'data', 'metadata.csv')

    col_name_map = {}

    if os.path.exists(meta_path):

        meta_df = pd.read_csv(meta_path, encoding='utf-8-sig')

        col_name_map = dict(zip(meta_df['컬럼영문명'], meta_df['컬럼한글명']))



    def get_korean_name(eng_name):

        if eng_name in col_name_map: return col_name_map[eng_name]

        for base, kr in col_name_map.items():

            if eng_name.startswith(base):

                suffix = eng_name[len(base):]

                suffix_map = {

                    '_change': ' 변화율', '_yoy': ' YoY',

                    '_ma3': ' 3개월평균', '_ma6': ' 6개월평균',

                    '_mom3': ' 3개월모멘텀', '_mom6': ' 6개월모멘텀',

                    '_lag1': ' 1개월전', '_lag2': ' 2개월전', '_lag3': ' 3개월전',

                }

                for suf, kr_suf in suffix_map.items():

                    if suffix == suf: return kr + kr_suf

        return eng_name



    # ═══════════════════════════════════════════════
    # 2. SHAP 분석 (CatBoost 단일 모델 기준)
    # ═══════════════════════════════════════════════
    print(f"\n{'='*55}")
    print("[XAI] CatBoost 모델 SHAP 분석 중...")
    print(f"{'='*55}")

    import warnings
    warnings.filterwarnings('ignore') # Shap 경고 메시지 무시
    
    # 1. VotingClassifier에서 CatBoost 모델만 단독으로 추출 ('cat'은 model.py에서 지정한 이름)
    try:
        cat_model = classifier.named_estimators_['xgb']
    except KeyError:
        raise ValueError("앙상블 모델 내에 'cat'이라는 이름의 CatBoost 모델이 존재하지 않습니다.")

    # 2. CatBoost 모델에 대한 TreeExplainer 생성 및 SHAP 값 도출
    explainer = shap.TreeExplainer(cat_model)
    sv = explainer.shap_values(X_test)
    
    # 3. 모델별 SHAP 반환 형태 통일 (하단 시각화 코드를 위해 리스트 형태로 변환)
    # CatBoost의 다중 클래스 출력은 보통 (n_samples, n_features, n_classes) 형태의 3차원 배열입니다.
    # SHAP 반환 형태를 클래별(인하, 동결, 인상) 리스트로 통일
    if isinstance(sv, np.ndarray) and sv.ndim == 3:
        shap_by_class = [sv[:, :, 0], sv[:, :, 1], sv[:, :, 2]]
    elif isinstance(sv, list) and len(sv) == 3:
        shap_by_class = [sv[0], sv[1], sv[2]]
    else:
        raise ValueError(f"예상치 못한 SHAP 값 형태입니다: {type(sv)}")

    # 글로벌 중요도 산출 (단일 모델이므로 평균낼 필요 없이 바로 계산)
    mean_abs_inha = np.abs(shap_by_class[0]).mean(axis=0)
    mean_abs_dong = np.abs(shap_by_class[1]).mean(axis=0)
    mean_abs_insang = np.abs(shap_by_class[2]).mean(axis=0)
    
    mean_abs = (mean_abs_inha + mean_abs_dong + mean_abs_insang) / 3

    importance_df = pd.DataFrame({
        'feature': X_test.columns,
        'feature_kr': [get_korean_name(c) for c in X_test.columns],
        'importance': mean_abs,
        'importance_인하': mean_abs_inha,
        'importance_동결': mean_abs_dong,
        'importance_인상': mean_abs_insang,
    }).sort_values('importance', ascending=False)
    
    misclass_records = []

    top_k = importance_df.head(k)



    # 전체 중요도 (Bar)

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.barh(range(len(top_k)-1, -1, -1), top_k['importance'].values, color='#4C72B0', edgecolor='#2d4a7c')

    ax.set_yticks(range(len(top_k)-1, -1, -1))

    ax.set_yticklabels(top_k['feature'].values, fontsize=11)

    ax.set_xlabel('평균 |SHAP value|', fontsize=12)

    ax.set_title('금리 방향(3클래스) 예측 — 피처 중요도 Top k', fontsize=14, fontweight='bold')

    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()

    clf_shap_path = os.path.join(results_dir, 'shap_classifier_top_k.png')

    plt.savefig(clf_shap_path, dpi=150, bbox_inches='tight')

    plt.close()



    # 다중 클래스 요약 (Stacked Bar)

    class_names = ['인하', '동결', '인상']

    kr_feature_names = [get_korean_name(c) for c in X_test.columns]

   

    shap.summary_plot(
        shap_by_class, X_test,
        feature_names=list(X_test.columns),
        class_names=class_names,
        max_display=15, show=False,
        plot_type="bar"
    )

    plt.legend(fontsize=10) # 숫자를 줄이면 더 작아집니다 (예: 8, 9, 10)

    plt.title('클래스별 전체 피처 기여도', fontsize=14, fontweight='bold')

    plt.tight_layout()

    summary_path = os.path.join(results_dir, 'shap_summary_bar.png')

    plt.savefig(summary_path, dpi=150, bbox_inches='tight')

    plt.close()



    # ── SHAP 대표 시각화: Beeswarm Plot (클래스별) ──

    # beeswarm CSV 저장을 위한 데이터 추출 (top 15 feature 대상)
    top_15_features_raw = importance_df['feature'].head(15).values
    top_15_features_kr = importance_df['feature_kr'].head(15).values
    
    beeswarm_records = []

    for cls_idx, cls_name in enumerate(class_names):

        plt.figure(figsize=(12, 8))

        # plot_type="dot" 이 디폴트 (벌떼 차트)

        shap.summary_plot(
            shap_by_class[cls_idx], X_test,
            feature_names=list(X_test.columns),
            max_display=15, show=False
        )

        plt.title(f'[{cls_name}] 예측 기여도 (SHAP Beeswarm Plot)', fontsize=14, fontweight='bold')

        plt.tight_layout()

        fname = f'shap_beeswarm_{cls_name}.png'

        plt.savefig(os.path.join(results_dir, fname), dpi=150, bbox_inches='tight')

        plt.close()

        # Beeswarm 데이터를 CSV로 저장하기 위해 데이터 수집
        sv_cls = shap_by_class[cls_idx]
        for f_raw, f_kr in zip(top_15_features_raw, top_15_features_kr):
            col_idx = X_test.columns.get_loc(f_raw)
            for j, val in enumerate(X_test.iloc[:, col_idx]):
                shap_val = sv_cls[j, col_idx]
                beeswarm_records.append({
                    'class': cls_name,
                    'feature_kr': f_kr,
                    'feature_value': val,
                    'shap_value': shap_val
                })

    # 추출한 Beeswarm 데이터를 CSV로 저장
    beeswarm_df = pd.DataFrame(beeswarm_records)
    beeswarm_csv_path = os.path.join(results_dir, 'shap_beeswarm.csv')
    beeswarm_df.to_csv(beeswarm_csv_path, index=False, encoding='utf-8-sig')
    print(f"      [CSV] Beeswarm 데이터 CSV 저장 완료: {beeswarm_csv_path}")



    importance_df.to_csv(os.path.join(results_dir, 'feature_importance_classifier.csv'), index=False, encoding='utf-8-sig')

    # 3. 오분류 분석

    # ═══════════════════════════════════════════════

    print(f"\n{'='*55}")

    print("[ANALYSIS] 오분류 케이스 심층 분석")

    print(f"{'='*55}")



    cls_preds_test = classifier.predict(X_test)

    cls_proba_test = classifier.predict_proba(X_test)

    y_test_label = test_df['label_encoded'].values



    wrong_mask = cls_preds_test != y_test_label

    wrong_indices = np.where(wrong_mask)[0]



    if len(wrong_indices) == 0:

        print("   오분류 케이스 없음! [OK]")

    else:

        for idx in wrong_indices:
            date = test_df.iloc[idx]['date_ym']
            actual = class_names[int(y_test_label[idx])]
            pred = class_names[int(cls_preds_test[idx])]
            proba = cls_proba_test[idx]

            print(f"\n   [CASE] {date} (실제: {actual} → 예측: {pred})")
            print(f"      확률: 인하 {proba[0]:.1%}  동결 {proba[1]:.1%}  인상 {proba[2]:.1%}")

            actual_cls = int(y_test_label[idx])
            # sv는 해당 케이스(idx)에 대한 실제 클래스(actual_cls)의 SHAP 값 (numpy 배열, n_features 크기)
            sv = shap_by_class[actual_cls][idx] if actual_cls < len(shap_by_class) else shap_by_class[0][idx]

            # =========================================================================
            # [추가 코드] 예시 이미지와 같은 SHAP 워터폴 플롯 생성 (오분류 케이스 대상)
            # =========================================================================
            try:
                # shap.plots.waterfall은 shap.Explanation 객체를 필요로 하므로 수동으로 생성
                # 다중 클래스 TreeExplainer의 expected_value는 (n_classes,) 형태의 배열
                actual_base_value = explainer.expected_value[actual_cls]
                
                exp = shap.Explanation(
                    values=sv, # SHAP 값 (n_features)
                    base_values=actual_base_value, # 기본값 (스칼라)
                    data=X_test.iloc[idx].values, # 피처 값 (n_features)
                    feature_names=list(X_test.columns) # 피처 이름 리스트 (n_features)
                )

                # 플롯 생성
                plt.figure(figsize=(10, 8)) # 예시 이미지와 유사한 비율 (필요시 조정 가능)
                
                # 워터폴 플롯 그리기
                shap.plots.waterfall(exp, max_display=10, show=False) 
                
                # -------------------------------------------------------------
                # [글씨 깨짐 해결 코드 추가] 
                # SHAP이 생성한 특수 유니코드 마이너스(\u2212)를 일반 마이너스(-)로 치환
                fig = plt.gcf()
                for ax_obj in fig.axes:
                    # 1. 그래프 내부 텍스트 (화살표 안의 숫자 등) 치환
                    for text_obj in ax_obj.texts:
                        current_text = text_obj.get_text()
                        if '\u2212' in current_text:
                            text_obj.set_text(current_text.replace('\u2212', '-'))
                    
                    # 2. y축 틱 라벨 (왼쪽 피처명 및 수치) 치환
                    new_labels = []
                    for label in ax_obj.get_yticklabels():
                        new_labels.append(label.get_text().replace('\u2212', '-'))
                    ax_obj.set_yticklabels(new_labels)
                # -------------------------------------------------------------
                
                # 제목 설정 (오분류된 케이스임을 명시)
                plt.title(f'오분류 분석 (실제:{actual} → 예측:{pred})\n{date} | 실제 클래스({actual})에 대한 SHAP 워터폴', fontsize=14, fontweight='bold')
                
                # 파일명 생성 (날짜와 예측 정보 포함)
                # 파일명에 공백이나 특수 문자가 들어가지 않도록 처리
                safe_date = date.replace(':', '').replace(' ', '_')
                waterfall_filename = f'waterfall_mistake_{safe_date}_{actual}_to_{pred}.png'
                waterfall_path = os.path.join(results_dir, waterfall_filename)
                
                plt.savefig(waterfall_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"      [PLOT] 워터폴 플롯 저장 완료: {waterfall_filename}")

            except Exception as e:
                # 워터폴 플롯 생성 실패 시 메시지 출력 후 다음 케이스로 진행
                print(f"      [WARNING] 워터폴 플롯 생성 중 오류 발생: {e}")
                import traceback
                traceback.print_exc() # 상세 오류 정보 출력 (선택 사항)
            # =========================================================================
            
            top_factors = pd.DataFrame({
                'feature': X_test.columns,
                'feature_kr': kr_feature_names,
                'shap_value': sv,
                'feature_value': X_test.iloc[idx].values,
            })

            top_factors['abs_shap'] = top_factors['shap_value'].abs()
            factors_str = []

            for _, r in top_factors.nlargest(5, 'abs_shap').iterrows():
                direction = '↑' if r['shap_value'] > 0 else '↓'
                factors_str.append(f"{direction} {r['feature_kr']}: 값={r['feature_value']:.4f}, SHAP={r['shap_value']:+.4f}")
                print(f"        {direction} {r['feature_kr']}: 값={r['feature_value']:.4f}, SHAP={r['shap_value']:+.4f}")

            # CSV 저장을 위한 오분류 딕셔너리 기록
            misclass_records.append({
                'date_ym': date,
                'actual': actual,
                'predict': pred,
                'proba_인하': proba[0],
                'proba_동결': proba[1],
                'proba_인상': proba[2],
                'top5_factors': " | ".join(factors_str)
            })

    # 오분류 결과 CSV 저장
    if len(misclass_records) > 0:
        misclass_df = pd.DataFrame(misclass_records)
        misclass_csv_path = os.path.join(results_dir, 'misclassification_analysis.csv')
        misclass_df.to_csv(misclass_csv_path, index=False, encoding='utf-8-sig')
        print(f"      [CSV] 오분류 상세 내역 저장 완료: {misclass_csv_path}")

    print(f"\n[OK] SHAP XAI 분석 완료!")



if __name__ == '__main__':

    explain_model()