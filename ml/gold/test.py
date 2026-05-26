import os
import pandas as pd
import numpy as np
import joblib
import pymysql
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
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
    
    env_path = os.path.abspath(os.path.join(base_dir, '../../.env'))
    load_dotenv(dotenv_path=env_path)
    
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

def test_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load Model, Scaler and Features
    print("[TEST] Loading trained models and resources...")
    classifier = joblib.load(os.path.join(models_dir, 'gold_xgb_classifier.pkl'))
    scaler = joblib.load(os.path.join(models_dir, 'gold_scaler.pkl'))
    feature_names = joblib.load(os.path.join(models_dir, 'gold_features.pkl'))
    
    df = load_data_from_mysql()
    cfg = GoldModel

    X_all = df[feature_names]
    y_all = df['target_tomorrow_gold_direction']

    # Train/Test Split Masks
    split_idx = int(len(df) * cfg.TRAIN_RATIO)
    train_mask = df.index < split_idx
    test_mask  = df.index >= split_idx
    
    print(f"   Total: {len(df)} days | Train: {train_mask.sum()} days | Test: {test_mask.sum()} days")

    # 2. Standardization
    X_all_scaled = scaler.transform(X_all)
    X_all_scaled_df = pd.DataFrame(X_all_scaled, columns=feature_names)

    # 3. Model Inference
    label_names = ['하락/보합', '상승']
    preds = classifier.predict(X_all_scaled_df)
    proba = classifier.predict_proba(X_all_scaled_df)

    tr_acc = accuracy_score(y_all[train_mask], preds[train_mask])
    te_acc = accuracy_score(y_all[test_mask], preds[test_mask])

    print(f"\n{'='*55}")
    print("[TEST] Gold Prediction Test Performance Summary")
    print(f"{'='*55}")
    print(f"   Train Hit Rate: {tr_acc*100:.2f}%")
    print(f"   Test Hit Rate:  {te_acc*100:.2f}%")
    
    gap = (tr_acc - te_acc) * 100
    if gap > 15:
        print(f"   [WARNING] Overfitting suspected (Gap: {gap:.2f}%p)")
    else:
        print(f"   [OK] No severe overfitting (Gap: {gap:.2f}%p)")
        
    print(f"\n[REPORT] Detailed Test Classification Report")
    print(classification_report(
        y_all[test_mask], preds[test_mask],
        labels=[0, 1], target_names=label_names, zero_division=0
    ))

    # 4. Save Prediction Results
    result_df = pd.DataFrame({
        'loaded_date':       df['loaded_date'].values,
        'split':             np.where(train_mask, 'train', 'test'),
        'actual_direction':  df['target_tomorrow_gold_direction'].values,
        'pred_direction':    preds,
        'pred_proba_down':   proba[:, 0],
        'pred_proba_up':     proba[:, 1],
        'match':             y_all.values == preds,
    })
    
    test_result = result_df[result_df['split'] == 'test']
    save_path = os.path.join(results_dir, 'test_result.csv')
    test_result.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"   Saved test prediction results to: {save_path}")

    # Calculate exhaustive metrics
    tr_f1_macro = f1_score(y_all[train_mask], preds[train_mask], average='macro', zero_division=0)
    te_f1_macro = f1_score(y_all[test_mask], preds[test_mask], average='macro', zero_division=0)
    tr_f1_weighted = f1_score(y_all[train_mask], preds[train_mask], average='weighted', zero_division=0)
    te_f1_weighted = f1_score(y_all[test_mask], preds[test_mask], average='weighted', zero_division=0)

    tr_prec_macro = precision_score(y_all[train_mask], preds[train_mask], average='macro', zero_division=0)
    te_prec_macro = precision_score(y_all[test_mask], preds[test_mask], average='macro', zero_division=0)
    tr_rec_macro = recall_score(y_all[train_mask], preds[train_mask], average='macro', zero_division=0)
    te_rec_macro = recall_score(y_all[test_mask], preds[test_mask], average='macro', zero_division=0)

    try:
        tr_auc = roc_auc_score(y_all[train_mask], proba[train_mask, 1])
        te_auc = roc_auc_score(y_all[test_mask], proba[test_mask, 1])
    except Exception:
        tr_auc = np.nan
        te_auc = np.nan

    metrics = pd.DataFrame([
        {
            '구분': 'Train',
            'Accuracy(%)': round(tr_acc*100, 2),
            'F1_Macro': round(tr_f1_macro, 4),
            'F1_Weighted': round(tr_f1_weighted, 4),
            'Precision_Macro': round(tr_prec_macro, 4),
            'Recall_Macro': round(tr_rec_macro, 4),
            'AUC_ROC': round(tr_auc, 4) if not np.isnan(tr_auc) else 'N/A'
        },
        {
            '구분': 'Test',
            'Accuracy(%)': round(te_acc*100, 2),
            'F1_Macro': round(te_f1_macro, 4),
            'F1_Weighted': round(te_f1_weighted, 4),
            'Precision_Macro': round(te_prec_macro, 4),
            'Recall_Macro': round(te_rec_macro, 4),
            'AUC_ROC': round(te_auc, 4) if not np.isnan(te_auc) else 'N/A'
        }
    ])
    
    metrics_path = os.path.join(results_dir, 'test_metrics.csv')
    metrics.to_csv(metrics_path, index=False, encoding='utf-8-sig')
    print(f"   Saved exhaustive metrics to: {metrics_path}")

    # 5. Generate Premium Dashboard Visualizations
    print("[TEST] Plotting premium evaluation dashboard...")
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

    # ── Panel 1: Train vs Test Accuracy ──
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(['Train', 'Test'], [tr_acc*100, te_acc*100], width=0.45,
                   color=['#3A6073', '#FF5F6D'], edgecolor='white', linewidth=1)
    ax1.set_ylabel('예측 정확도 (Hit Rate, %)', fontsize=11)
    ax1.set_title('Train vs Test 성능 비교', fontsize=13, fontweight='bold', pad=15)
    ax1.set_ylim(0, 105)
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f'{bar.get_height():.2f}%', ha='center', fontsize=11, fontweight='bold',
                 color='#333333')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('#cccccc')
    ax1.spines['bottom'].set_color('#cccccc')

    # ── Panel 2: Confusion Matrix Heatmap ──
    ax2 = fig.add_subplot(gs[0, 1])
    cm = confusion_matrix(y_all[test_mask], preds[test_mask])
    im = ax2.imshow(cm, cmap='Blues', aspect='auto', interpolation='nearest')
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['하락/보합', '상승'], fontsize=11)
    ax2.set_yticklabels(['하락/보합', '상승'], fontsize=11)
    ax2.set_xlabel('예측 방향', fontsize=11, labelpad=10)
    ax2.set_ylabel('실제 방향', fontsize=11, labelpad=10)
    ax2.set_title('혼동 행렬 (Confusion Matrix - Test)', fontsize=13, fontweight='bold', pad=15)
    
    # Write values in heatmap
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = 'white' if val > thresh else 'black'
            ax2.text(j, i, f"{val}\n({val/cm.sum()*100:.1f}%)", ha='center', va='center',
                     fontsize=14, fontweight='bold', color=color)
    ax2.grid(False)

    # ── Panel 3: Performance Cards ──
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    report = classification_report(y_all[test_mask], preds[test_mask], output_dict=True)
    correct_count = (y_all[test_mask].values == preds[test_mask]).sum()
    total_count = test_mask.sum()
    
    info_lines = [
        ('최종 테스트 정확도', f"{te_acc*100:.2f}%"),
        ('', ''),
        ('상승(1) F1-Score', f"{report['1']['f1-score']:.4f}"),
        ('하락(0) F1-Score', f"{report['0']['f1-score']:.4f}"),
        ('Macro F1-Score', f"{te_f1_macro:.4f}"),
        ('Test AUC-ROC 스코어', f"{te_auc:.4f}" if not np.isnan(te_auc) else 'N/A'),
        ('', ''),
        ('정답 개수', f"{correct_count} / {total_count} 거래일"),
    ]
    
    y_start = 0.88
    ax3.text(0.5, 0.96, '금값 예측 테스트 핵심 지표', transform=ax3.transAxes,
             fontsize=14, fontweight='bold', ha='center', va='top')
             
    for i, (label, value) in enumerate(info_lines):
        y = y_start - i * 0.1
        if label == '': continue
        ax3.text(0.12, y, label, transform=ax3.transAxes, fontsize=11, ha='left', va='top', color='#333333')
        ax3.text(0.88, y, value, transform=ax3.transAxes, fontsize=12, ha='right', va='top', fontweight='bold', color='#1e3d59')
        
    from matplotlib.patches import FancyBboxPatch
    bg = FancyBboxPatch((0.03, 0.05), 0.94, 0.92, transform=ax3.transAxes,
                        boxstyle='round,pad=0.01', facecolor='#f7f9fa',
                        edgecolor='#1e3d59', linewidth=1.5, zorder=0)
    ax3.add_patch(bg)

    # ── Panel 4: Daily Actual vs Predicted Direction Timeline (Last 60 trading days) ──
    ax4 = fig.add_subplot(gs[1, :])
    
    # Filter for the last 60 days
    last_n = 60
    timeline_df = test_result.tail(last_n)
    
    dates = timeline_df['loaded_date'].values
    actual = timeline_df['actual_direction'].values
    predicted = timeline_df['pred_direction'].values
    matches = timeline_df['match'].values
    
    x_pos = np.arange(len(dates))
    
    # Background: Green for hit, Light Red for mistake
    for i, is_match in enumerate(matches):
        bg_color = '#eafaf1' if is_match else '#fdf2f2'
        ax4.axvspan(i - 0.45, i + 0.45, facecolor=bg_color, alpha=0.9, zorder=0)
        
    # Plot predicted indicators
    # We plot actual direction as bars (+1 for 상승, -1 for 하락/보합)
    actual_scores = np.where(actual == 1, 1, -1)
    pred_scores = np.where(predicted == 1, 1, -1)
    
    colors = ['#FF5F6D' if s == -1 else '#3A6073' for s in actual_scores]
    ax4.bar(x_pos, actual_scores, 0.6, label='실제 금리 방향', color=colors,
            edgecolor='white', linewidth=0.5, zorder=3)
            
    # Add text indicators
    for i, (a_score, p_score) in enumerate(zip(actual_scores, pred_scores)):
        y_text = a_score + 0.08 if a_score > 0 else a_score - 0.18
        va = 'bottom' if a_score > 0 else 'top'
        color = '#2e7d32' if a_score == p_score else '#c62828'
        lbl_str = '↑상승' if p_score == 1 else '↓하락'
        ax4.text(i, y_text, lbl_str, ha='center', va=va, fontsize=8, fontweight='bold', color=color)
        
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(dates, rotation=45, ha='right', fontsize=9)
    ax4.set_yticks([-1, 1])
    ax4.set_yticklabels(['하락/보합', '상승'], fontsize=11)
    ax4.set_ylabel('금값 변동 기조', fontsize=11)
    ax4.set_title(f'최근 {last_n} 거래일 실제 변동 기조 + 예측 라벨 (배경: 초록=적중, 빨강=오분류)', fontsize=13, fontweight='bold', pad=15)
    ax4.axhline(y=0, color='#666666', linewidth=0.8, zorder=2)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['left'].set_color('#cccccc')
    ax4.spines['bottom'].set_color('#cccccc')
    ax4.set_ylim(-1.5, 1.5)
    ax4.grid(axis='y', alpha=0.2, linestyle='--')

    fig_path = os.path.join(results_dir, 'test_dashboard.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n[OK] Evaluation completed. Dashboard saved at: {fig_path}")

if __name__ == '__main__':
    test_model()
