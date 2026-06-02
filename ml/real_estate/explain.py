import os
import pickle
import numpy as np
import pandas as pd
import pymysql
import shap
import matplotlib
from dotenv import load_dotenv, find_dotenv

matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from model import RealEstateEnsembleRegressor

# ── 한글 폰트 설정 ──
font_path = 'C:/Windows/Fonts/malgun.ttf'
if os.path.exists(font_path):
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False


def load_data_from_mysql():
    load_dotenv(find_dotenv())
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
            sql = "SELECT * FROM ml_realestate_preprocessed ORDER BY date_ym ASC"
            cursor.execute(sql)
            rows = cursor.fetchall()
    finally:
        connection.close()
        
    df = pd.DataFrame(rows)
    for col in df.columns:
        if col not in ['date_ym']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def run_explain():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Load model and features
    model_path = os.path.join(models_dir, 'regressor.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    features_path = os.path.join(models_dir, 'selected_features.pkl')
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path)):
        print("[Error] Models not found. Run train.py first.")
        return
        
    with open(model_path, 'rb') as f:
        ensemble = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(features_path, 'rb') as f:
        selected_features = pickle.load(f)
        
    # Get preprocessed data from MySQL
    df = load_data_from_mysql()
    df['date_ym'] = df['date_ym'].astype(str).str.strip()
    cfg = RealEstateEnsembleRegressor
    
    test_mask = df['date_ym'] >= cfg.TEST_START
    test_df = df[test_mask].copy()
    
    X_test = test_df[selected_features]
    X_test_sc = scaler.transform(X_test)
    X_train = df[df['date_ym'] <= cfg.TRAIN_END][selected_features]
    X_train_sc = scaler.transform(X_train)
    
    X_test_df = pd.DataFrame(X_test_sc, columns=selected_features)
    X_train_df = pd.DataFrame(X_train_sc, columns=selected_features)
    
    # Load Metadata for Korean Feature Names mapping
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

    korean_feature_names = [get_korean_name(f) for f in selected_features]

    print("\n" + "=" * 55)
    print("Computing SHAP Values (Explainable AI)")
    print("=" * 55)
    
    # -----------------------------------------
    # Compute SHAP for each model and average them using weights (Ridge 60%, RF 20%, CatBoost 20%)
    # -----------------------------------------
    shap_values_list = []
    base_values_list = []
    weights = {'RidgeRegressor': 0.60, 'RandomForest': 0.20, 'CatBoost': 0.20}
    
    for name, model in ensemble.models.items():
        print(f"  * Explaining {name}...")
        w = weights.get(name, 0.33)
        try:
            explainer = shap.TreeExplainer(model, data=X_train_df)
            sv = explainer(X_test_df).values
            expected_val = explainer.expected_value
        except Exception as e:
            print(f"    - Explainer fallback for {name} due to: {e}")
            explainer = shap.Explainer(model, X_train_df)
            sv = explainer(X_test_df).values
            expected_val = explainer.expected_value
            
        if isinstance(expected_val, np.ndarray) and len(expected_val) > 0:
            expected_val = expected_val[0]
            
        shap_values_list.append(sv * w)
        base_values_list.append(expected_val * w)
            
    # Weighted SHAP values for the ensemble
    ensemble_shap_values = np.sum(shap_values_list, axis=0)
    ensemble_base_value = np.sum(base_values_list)
    
    # Create Explanation object using raw (unscaled) X_test values for beautiful axis labels
    X_test_raw_df = X_test.reset_index(drop=True)
    ensemble_explanation = shap.Explanation(
        values=ensemble_shap_values,
        base_values=np.repeat(ensemble_base_value, len(X_test_raw_df)),
        data=X_test_raw_df.values,
        feature_names=korean_feature_names
    )
    
    # -----------------------------------------
    # 1. Save Beeswarm Plot (Global Feature Impact)
    # -----------------------------------------
    print("  Generating SHAP Beeswarm Plot...")
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(ensemble_explanation, show=False)
    
    # Font Unicode minus sign fix
    fig = plt.gcf()
    for ax_obj in fig.axes:
        for text_obj in ax_obj.texts:
            text_obj.set_text(text_obj.get_text().replace('\u2212', '-'))
        new_labels = []
        for label in ax_obj.get_yticklabels():
            new_labels.append(label.get_text().replace('\u2212', '-'))
        ax_obj.set_yticklabels(new_labels)
        new_xlabels = []
        for label in ax_obj.get_xticklabels():
            new_xlabels.append(label.get_text().replace('\u2212', '-'))
        ax_obj.set_xticklabels(new_xlabels)
        
    plt.title('부동산 매매가격지수 예측 변수 기여도 (Beeswarm Plot)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    beeswarm_path = os.path.join(results_dir, 'shap_beeswarm.png')
    plt.savefig(beeswarm_path, dpi=150)
    plt.close()
    print(f"    - Saved beeswarm plot to: {beeswarm_path}")
    
    # -----------------------------------------
    # 2. Save Feature Importance Bar Chart
    # -----------------------------------------
    print("  Generating SHAP Feature Importance Plot...")
    plt.figure(figsize=(10, 6))
    shap.plots.bar(ensemble_explanation, show=False)
    
    fig = plt.gcf()
    for ax_obj in fig.axes:
        for text_obj in ax_obj.texts:
            text_obj.set_text(text_obj.get_text().replace('\u2212', '-'))
        new_labels = []
        for label in ax_obj.get_yticklabels():
            new_labels.append(label.get_text().replace('\u2212', '-'))
        ax_obj.set_yticklabels(new_labels)
        
    plt.title('전체 피처별 평균 기여도 순위 (Feature Importance)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    bar_path = os.path.join(results_dir, 'shap_importance.png')
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"    - Saved feature importance to: {bar_path}")
    
    # -----------------------------------------
    # 3. Save Local Waterfall Plots for High-Error Months
    # -----------------------------------------
    predictions_path = os.path.join(results_dir, 'predictions.csv')
    if os.path.exists(predictions_path):
        pred_df = pd.read_csv(predictions_path)
        max_err_idx = pred_df['abs_error'].idxmax()
        max_err_row = pred_df.loc[max_err_idx]
        
        print(f"  Generating Waterfall Plot for Max Error Month: {max_err_row['date_ym']}...")
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(ensemble_explanation[max_err_idx], show=False)
        
        fig = plt.gcf()
        for ax_obj in fig.axes:
            for text_obj in ax_obj.texts:
                text_obj.set_text(text_obj.get_text().replace('\u2212', '-'))
            new_labels = []
            for label in ax_obj.get_yticklabels():
                new_labels.append(label.get_text().replace('\u2212', '-'))
            ax_obj.set_yticklabels(new_labels)
            
        plt.title(f"최대 오차 시점 예측 분석 (실제:{max_err_row['actual_change_rate']:.2f}% → 예측:{max_err_row['pred_change_rate']:.2f}%)\n{max_err_row['date_ym']} | SHAP Waterfall Plot", fontsize=14, fontweight='bold')
        plt.tight_layout()
        waterfall_path = os.path.join(results_dir, f"shap_waterfall_{max_err_row['date_ym']}.png")
        plt.savefig(waterfall_path, dpi=150)
        plt.close()
        print(f"    - Saved waterfall plot to: {waterfall_path}")
        
    # -----------------------------------------
    # 4. Save CSV Data Results
    # -----------------------------------------
    # 4-1) feature_importance_regressor.csv
    mean_abs_shap = np.abs(ensemble_shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': selected_features,
        'feature_kr': korean_feature_names,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=False)
    importance_df.to_csv(os.path.join(results_dir, 'feature_importance_regressor.csv'), index=False, encoding='utf-8-sig')

    # 4-2) shap_beeswarm.csv (Beeswarm 데이터화하여 CSV로 저장)
    beeswarm_records = []
    for col_idx, (f_raw, f_kr) in enumerate(zip(selected_features, korean_feature_names)):
        for j, val in enumerate(X_test_raw_df.iloc[:, col_idx]):
            shap_val = ensemble_shap_values[j, col_idx]
            beeswarm_records.append({
                'class': 'real_estate',
                'feature_kr': f_kr,
                'feature_value': val,
                'shap_value': shap_val
            })
    beeswarm_df = pd.DataFrame(beeswarm_records)
    beeswarm_df.to_csv(os.path.join(results_dir, 'shap_beeswarm.csv'), index=False, encoding='utf-8-sig')

    # 4-3) shap_values.csv (Compatibility with interpret_xai)
    shap_df = pd.DataFrame(ensemble_shap_values, columns=[f"shap_{f}" for f in selected_features])
    shap_df['date_ym'] = test_df['date_ym'].values
    shap_df.to_csv(os.path.join(results_dir, 'shap_values.csv'), index=False, encoding='utf-8-sig')
    
    print("\nSHAP XAI Analysis Completed Successfully!")


if __name__ == '__main__':
    run_explain()
