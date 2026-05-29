import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

import os
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import pymysql
from dotenv import load_dotenv
from model import GoldModel
import warnings

warnings.filterwarnings('ignore')

# ── Korean Font Setup ──
font_path = 'C:/Windows/Fonts/malgun.ttf'
if os.path.exists(font_path):
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

def load_data_from_mysql():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'data', 'final_dataset.csv')
    
    load_dotenv(find_dotenv())
    
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    DB_NAME = os.getenv('DB_NAME')
    
    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
        print("[Warning] Missing DB configuration. Falling back to local final_dataset.csv...")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"[CSV] Loaded data successfully from local final_dataset.csv: {csv_path}")
            return df
        raise ValueError(f"No DB credentials and final CSV not found at: {csv_path}")
        
    try:
        DB_PORT = int(DB_PORT)
        connection = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=5
        )
        
        try:
            with connection.cursor() as cursor:
                sql = "SELECT * FROM ml_gold_preprocessed ORDER BY loaded_date ASC"
                cursor.execute(sql)
                rows = cursor.fetchall()
        finally:
            connection.close()
            
        df = pd.DataFrame(rows)
        for col in df.columns:
            if col not in ['loaded_date']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        print("[DB] Loaded preprocessed data successfully from MySQL table 'ml_gold_preprocessed'.")
        return df
    except Exception as e:
        print(f"[Warning] MySQL connection/query failed ({e}). Falling back to local final_dataset.csv...")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"[CSV] Loaded preprocessed data successfully from local CSV: {csv_path}")
            return df
        raise RuntimeError(f"Database connection failed and local CSV not found at: {csv_path}")

def explain_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load Model, Scaler, Features and Preprocessed Data
    print("[XAI] Loading trained models and resources...")
    classifier = joblib.load(os.path.join(models_dir, 'gold_xgb_classifier.pkl'))
    scaler = joblib.load(os.path.join(models_dir, 'gold_scaler.pkl'))
    feature_names = joblib.load(os.path.join(models_dir, 'gold_features.pkl'))
    
    df = load_data_from_mysql()
    cfg = GoldModel

    split_idx = int(len(df) * cfg.TRAIN_RATIO)
    test_df = df.iloc[split_idx:].copy()

    X_test = test_df[feature_names]
    y_test = test_df['target_tomorrow_gold_direction'].values
    dates_test = test_df['loaded_date']

    # Scale test set
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)

    # Load Column metadata translation map
    meta_path = os.path.join(base_dir, 'data', 'metadata.csv')
    col_name_map = {}
    if os.path.exists(meta_path):
        meta_df = pd.read_csv(meta_path, encoding='utf-8-sig')
        col_name_map = dict(zip(meta_df['컬럼영문명'], meta_df['컬럼한글명']))

    def get_korean_name(eng_name):
        # 1. Direct match in metadata
        if eng_name in col_name_map:
            return col_name_map[eng_name]
            
        # 2. Match with derived/engineered features
        # Lags
        if '_lag_' in eng_name:
            parts = eng_name.split('_lag_')
            base_col = parts[0]
            lag_val = parts[1]
            return get_korean_name(base_col) + f" {lag_val}일전"
            
        # Change rates
        if eng_name.endswith('_change_rate'):
            base = eng_name.replace('_change_rate', '')
            return get_korean_name(base) + " 변화율"
            
        # Technical indicators
        tech_map = {
            'gold_rsi_14': '금값 RSI(14일)',
            'gold_macd': '금값 MACD',
            'gold_macd_signal': '금값 MACD 시그널',
            'gold_change_rate_ema_5': '금값 변화율 EMA(5일)',
            'gold_change_rate_ema_20': '금값 변화율 EMA(20일)',
            'sp500_kospi200_spread': 'S&P500-KOSPI200 스프레드',
            'gold_dxy_interaction': '금값-달러인덱스 상호작용',
            'gold_change_rate_sma_5': '금값 변화율 SMA(5일이동평균)',
            'gold_change_rate_sma_20': '금값 변화율 SMA(20일이동평균)',
            'gold_change_rate_std_5': '금값 변동성(5일표준편차)',
            'gold_change_rate_std_20': '금값 변동성(20일표준편차)',
        }
        if eng_name in tech_map:
            return tech_map[eng_name]

        return eng_name

    # 2. SHAP Analysis
    print(f"\n{'='*55}")
    print("[XAI] Running TreeExplainer on Tuned XGBoost Model...")
    print(f"{'='*55}")

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_test_scaled_df)

    # 3. Global Importance Tabular Dumps
    # SHAP returns a single array for binary classification (prob of class 1)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        'feature': X_test.columns,
        'feature_kr': [get_korean_name(c) for c in X_test.columns],
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=False)

    importance_csv_path = os.path.join(results_dir, 'feature_importance_classifier.csv')
    importance_df.to_csv(importance_csv_path, index=False, encoding='utf-8-sig')
    print(f"   [CSV] Saved feature importance CSV to: {importance_csv_path}")

    # 4. Generate SHAP Plots
    # A. Feature Importance Bar Chart
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_scaled_df, plot_type="bar", show=False)
    plt.title("금값 상승/하락 예측 모델 - 피처 중요도 Top 15 (SHAP)", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    bar_path = os.path.join(results_dir, 'gold_shap_importance.png')
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   [PLOT] Saved global importance bar chart: {bar_path}")

    # B. Beeswarm Plot
    plt.figure(figsize=(12, 9))
    shap.summary_plot(shap_values, X_test_scaled_df, show=False)
    plt.title("금값 변동 예측 기여도 분포 (SHAP Beeswarm Plot)", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    beeswarm_path = os.path.join(results_dir, 'gold_shap_beeswarm.png')
    plt.savefig(beeswarm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   [PLOT] Saved beeswarm distribution plot: {beeswarm_path}")

    # C. Dump Beeswarm data to CSV for downstream LLM analysis
    top_15_features = importance_df['feature'].head(15).values
    top_15_features_kr = importance_df['feature_kr'].head(15).values
    
    beeswarm_records = []
    for f_raw, f_kr in zip(top_15_features, top_15_features_kr):
        col_idx = X_test.columns.get_loc(f_raw)
        for j, val in enumerate(X_test.iloc[:, col_idx]):
            shap_val = shap_values[j, col_idx]
            beeswarm_records.append({
                'feature_kr': f_kr,
                'feature_value': val,
                'shap_value': shap_val
            })

    beeswarm_df = pd.DataFrame(beeswarm_records)
    beeswarm_csv_path = os.path.join(results_dir, 'shap_beeswarm.csv')
    beeswarm_df.to_csv(beeswarm_csv_path, index=False, encoding='utf-8-sig')
    print(f"   [CSV] Saved Beeswarm summary CSV to: {beeswarm_csv_path}")

    # 5. Misclassification Waterfall Analysis
    preds_test = classifier.predict(X_test_scaled_df)
    wrong_mask = preds_test != y_test
    wrong_indices = np.where(wrong_mask)[0]

    if len(wrong_indices) == 0:
        print("   No misclassification found in the test set. [OK]")
    else:
        misclass_records = []
        
        # We perform analysis on all misclassified cases and dump to CSV,
        # but generate a waterfall plot specifically for the first misclassified case.
        print(f"\n   Found {len(wrong_indices)} misclassified trading days. Logging cases...")
        
        # Setup first waterfall plot
        rep_idx = wrong_indices[0]
        date_ymd = dates_test.iloc[rep_idx]
        actual_lbl = '상승' if y_test[rep_idx] == 1 else '하락/보합'
        pred_lbl = '상승' if preds_test[rep_idx] == 1 else '하락/보합'
        
        try:
            # Recreate Explanation object for single waterfall instance
            exp = shap.Explanation(
                values=shap_values[rep_idx],
                base_values=explainer.expected_value,
                data=X_test_scaled_df.iloc[rep_idx].values,
                feature_names=X_test.columns.tolist()
            )
            
            plt.figure(figsize=(10, 8))
            shap.plots.waterfall(exp, max_display=10, show=False)
            plt.title(f"오분류 워터폴 분석 ({date_ymd} | 실제: {actual_lbl} → 예측: {pred_lbl})", fontsize=13, fontweight='bold', pad=15)
            
            # Clean up matplotlib unicode characters to prevent overlaps/font crashes
            fig = plt.gcf()
            for ax_obj in fig.axes:
                for text_obj in ax_obj.texts:
                    txt = text_obj.get_text()
                    if '\u2212' in txt:
                        text_obj.set_text(txt.replace('\u2212', '-'))
                new_lbls = [label.get_text().replace('\u2212', '-') for label in ax_obj.get_yticklabels()]
                ax_obj.set_yticklabels(new_lbls)
                
            plt.tight_layout()
            waterfall_path = os.path.join(results_dir, 'gold_shap_waterfall_mistake.png')
            plt.savefig(waterfall_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   [PLOT] Saved first misclassification waterfall plot to: {waterfall_path}")
            
        except Exception as e:
            print(f"   [WARNING] Waterfall plot failed: {e}")

        # Logging all wrong predictions for GPT analysis
        proba_test = classifier.predict_proba(X_test_scaled_df)
        for idx in wrong_indices:
            date = dates_test.iloc[idx]
            actual = '상승' if y_test[idx] == 1 else '하락/보합'
            pred = '상승' if preds_test[idx] == 1 else '하락/보합'
            pb = proba_test[idx]
            
            # Get top 5 factors contributing to this prediction (in terms of SHAP magnitude)
            sv = shap_values[idx]
            top_factors = pd.DataFrame({
                'feature_kr': [get_korean_name(c) for c in X_test.columns],
                'shap_value': sv,
                'feature_value': X_test.iloc[idx].values,
            })
            top_factors['abs_shap'] = top_factors['shap_value'].abs()
            top_5 = top_factors.nlargest(5, 'abs_shap')
            
            factors_str = []
            for _, r in top_5.iterrows():
                direction = '↑' if r['shap_value'] > 0 else '↓'
                factors_str.append(f"{direction} {r['feature_kr']}: 값={r['feature_value']:.4f}, SHAP={r['shap_value']:+.4f}")

            misclass_records.append({
                'date_ymd': date,
                'actual': actual,
                'predict': pred,
                'proba_down': pb[0],
                'proba_up': pb[1],
                'top5_factors': " | ".join(factors_str)
            })
            
        misclass_df = pd.DataFrame(misclass_records)
        misclass_csv_path = os.path.join(results_dir, 'misclassification_analysis.csv')
        misclass_df.to_csv(misclass_csv_path, index=False, encoding='utf-8-sig')
        print(f"   [CSV] Saved comprehensive misclassification CSV to: {misclass_csv_path}")

    print(f"\n{'='*55}")
    print("🎉 [SUCCESS] SHAP XAI Analysis completed successfully!")
    print(f"{'='*55}\n")

if __name__ == '__main__':
    explain_model()
