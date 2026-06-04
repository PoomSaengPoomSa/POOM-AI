import os
import pandas as pd
import numpy as np
import joblib
import pymysql
from dotenv import load_dotenv, find_dotenv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from model import InterestRateEnsembleModel

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

def test_model(valid_mode=False):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    results_dir = os.path.join(base_dir, 'results')

    # ═══════════════════════════════════════════════
    # 1. 모델 & 데이터 로드 (MySQL Database)
    # ═══════════════════════════════════════════════
    print("[TEST] 저장된 분류 모델 로드 중...")
    classifier = joblib.load(os.path.join(models_dir, 'classifier.pkl'))
    feature_names = joblib.load(os.path.join(models_dir, 'feature_names.pkl'))
    df = load_data_from_mysql()
    df['date_ym'] = df['date_ym'].astype(str).str.strip()
    cfg = InterestRateEnsembleModel
    X_all = df[feature_names]
    y_label  = df['label_encoded']
    
    # Train/Validation/Test 분리 조건식
    if valid_mode:
        train_mask = df['date_ym'] <= cfg.TRAIN_END
        eval_mask  = df['date_ym'].between(cfg.VALID_START, cfg.VALID_END)
        eval_name  = "Validation"
        print(f"   전체: {len(df)}건  |  Train: {train_mask.sum()}건  |  Valid (Eval): {eval_mask.sum()}건")
    else:
        train_mask = df['date_ym'] <= cfg.VALID_END
        eval_mask  = df['date_ym'].between(cfg.TEST_START, cfg.TEST_END)
        eval_name  = "Test"
        print(f"   전체: {len(df)}건  |  Train+Valid: {train_mask.sum()}건  |  Test (Eval): {eval_mask.sum()}건")

    # ═══════════════════════════════════════════════
    # 2. 통계치 및 분류 수행
    # ═══════════════════════════════════════════════
    label_names = ['인하', '동결', '인상']
    cls_preds = classifier.predict(X_all)
    cls_proba = classifier.predict_proba(X_all)
    tr_acc = accuracy_score(y_label[train_mask], cls_preds[train_mask])
    eval_acc = accuracy_score(y_label[eval_mask], cls_preds[eval_mask])
    
    print(f"\n{'='*55}")
    print(f"[TEST] {eval_name} 성능 점검")
    print(f"{'='*55}")
    print(f"   Train 분류정확도: {tr_acc*100:.1f}%")
    print(f"   {eval_name} 분류정확도:  {eval_acc*100:.1f}%")
    gap = (tr_acc - eval_acc) * 100
    if gap > 20:
        print(f"   [WARNING] 과적합 의심 (차이: {gap:.1f}%p)")
    else:
        print(f"   [OK] 과적합 없음 (차이: {gap:.1f}%p)")
    print(f"\n[REPORT] {eval_name} 상세 리포트")
    print(classification_report(
        y_label[eval_mask], cls_preds[eval_mask],
        labels=[0, 1, 2], target_names=label_names, zero_division=0,
    ))

    # ═══════════════════════════════════════════════
    # 3. 전수 결과 저장
    # ═══════════════════════════════════════════════
    result_df = pd.DataFrame({
        'date_ym':           df['date_ym'].values,
        'split':             np.where(train_mask, 'train', 'eval'),
        'actual_label':      df['label'].values,
        'pred_direction':    [label_names[int(p)] for p in cls_preds],
        'pred_proba_인하':   cls_proba[:, 0],
        'pred_proba_동결':   cls_proba[:, 1],
        'pred_proba_인상':   cls_proba[:, 2],
        'match':             y_label.values == cls_preds,
    })
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, f"{eval_name.lower()}_result.csv")
    eval_result = result_df[result_df['split'] == 'eval']
    eval_result.to_csv(save_path, index=False, encoding='utf-8-sig')
    total = len(eval_result)
    correct = eval_result['match'].sum()
    
    tr_f1_macro = f1_score(y_label[train_mask], cls_preds[train_mask], average='macro', zero_division=0)
    eval_f1_macro = f1_score(y_label[eval_mask], cls_preds[eval_mask], average='macro', zero_division=0)
    
    tr_f1_weighted = f1_score(y_label[train_mask], cls_preds[train_mask], average='weighted', zero_division=0)
    eval_f1_weighted = f1_score(y_label[eval_mask], cls_preds[eval_mask], average='weighted', zero_division=0)

    tr_prec_macro = precision_score(y_label[train_mask], cls_preds[train_mask], average='macro', zero_division=0)
    eval_prec_macro = precision_score(y_label[eval_mask], cls_preds[eval_mask], average='macro', zero_division=0)
    
    tr_prec_weighted = precision_score(y_label[train_mask], cls_preds[train_mask], average='weighted', zero_division=0)
    eval_prec_weighted = precision_score(y_label[eval_mask], cls_preds[eval_mask], average='weighted', zero_division=0)

    tr_rec_macro = recall_score(y_label[train_mask], cls_preds[train_mask], average='macro', zero_division=0)
    eval_rec_macro = recall_score(y_label[eval_mask], cls_preds[eval_mask], average='macro', zero_division=0)
    
    tr_rec_weighted = recall_score(y_label[train_mask], cls_preds[train_mask], average='weighted', zero_division=0)
    eval_rec_weighted = recall_score(y_label[eval_mask], cls_preds[eval_mask], average='weighted', zero_division=0)

    try:
        tr_auc_macro = roc_auc_score(y_label[train_mask], cls_proba[train_mask], multi_class='ovr', average='macro', labels=[0, 1, 2])
        tr_auc_weighted = roc_auc_score(y_label[train_mask], cls_proba[train_mask], multi_class='ovr', average='weighted', labels=[0, 1, 2])
    except Exception:
        tr_auc_macro = np.nan
        tr_auc_weighted = np.nan

    try:
        eval_auc_macro = roc_auc_score(y_label[eval_mask], cls_proba[eval_mask], multi_class='ovr', average='macro', labels=[0, 1, 2])
        eval_auc_weighted = roc_auc_score(y_label[eval_mask], cls_proba[eval_mask], multi_class='ovr', average='weighted', labels=[0, 1, 2])
    except Exception:
        eval_auc_macro = np.nan
        eval_auc_weighted = np.nan

    train_label_str = "Train" if valid_mode else "Train+Valid"
    metrics = pd.DataFrame([
        {
            '구분': train_label_str,
            'Accuracy(%)': round(tr_acc*100, 2),
            'F1_Macro': round(tr_f1_macro, 4),
            'F1_Weighted': round(tr_f1_weighted, 4),
            'Precision_Macro': round(tr_prec_macro, 4),
            'Precision_Weighted': round(tr_prec_weighted, 4),
            'Recall_Macro': round(tr_rec_macro, 4),
            'Recall_Weighted': round(tr_rec_weighted, 4),
            'AUC_Macro': round(tr_auc_macro, 4) if not np.isnan(tr_auc_macro) else 'N/A',
            'AUC_Weighted': round(tr_auc_weighted, 4) if not np.isnan(tr_auc_weighted) else 'N/A'
        },
        {
            '구분': eval_name,
            'Accuracy(%)': round(eval_acc*100, 2),
            'F1_Macro': round(eval_f1_macro, 4),
            'F1_Weighted': round(eval_f1_weighted, 4),
            'Precision_Macro': round(eval_prec_macro, 4),
            'Precision_Weighted': round(eval_prec_weighted, 4),
            'Recall_Macro': round(eval_rec_macro, 4),
            'Recall_Weighted': round(eval_rec_weighted, 4),
            'AUC_Macro': round(eval_auc_macro, 4) if not np.isnan(eval_auc_macro) else 'N/A',
            'AUC_Weighted': round(eval_auc_weighted, 4) if not np.isnan(eval_auc_weighted) else 'N/A'
        }
    ])
    metrics_path = os.path.join(results_dir, 'test_metrics.csv')
    metrics.to_csv(metrics_path, index=False, encoding='utf-8-sig')

    # ═══════════════════════════════════════════════
    # 4. 대시보드 시각화 저장
    # ═══════════════════════════════════════════════
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

    # ── (1) 정확도 비교 ──
    ax1 = fig.add_subplot(gs[0, 0])
    bar_labels = [train_label_str, eval_name]
    bars = ax1.bar(bar_labels, [tr_acc*100, eval_acc*100], width=0.4,
                   color=['#4C72B0', '#DD8452'], edgecolor='white')
    ax1.set_ylabel('분류 정확도 (%)', fontsize=11)
    ax1.set_title('데이터 분할별 성능 비교', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 105)
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{bar.get_height():.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ── (2) 혼동 행렬 히트맵 ──
    ax2 = fig.add_subplot(gs[0, 1])
    cm_display = confusion_matrix(y_label[eval_mask], cls_preds[eval_mask], labels=[0, 1, 2])
    im = ax2.imshow(cm_display, cmap='Blues', aspect='auto')
    ax2.set_xticks([0, 1, 2])
    ax2.set_yticks([0, 1, 2])
    ax2.set_xticklabels(['인하', '동결', '인상'], fontsize=11)
    ax2.set_yticklabels(['인하', '동결', '인상'], fontsize=11)
    ax2.set_xlabel('예측', fontsize=11)
    ax2.set_ylabel('실제', fontsize=11)
    ax2.set_title(f'혼동 행렬 ({eval_name})', fontsize=13, fontweight='bold')
    for i in range(3):
        for j in range(3):
            val = cm_display[i, j]
            color = 'white' if val > cm_display.max() / 2 else 'black'
            ax2.text(j, i, str(val), ha='center', va='center',
                     fontsize=16, fontweight='bold', color=color)

    # ── (3) 성능 요약 카드 ──
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    report = classification_report(y_label[eval_mask], cls_preds[eval_mask], labels=[0, 1, 2], target_names=label_names, zero_division=0, output_dict=True)
    info_lines = [
        (f'{eval_name} 분류 정확도', f"{eval_acc*100:.1f}%"),
        ('', ''),
        ('인상 재현율 (Recall)', f"{report['인상']['recall']*100:.1f}%"),
        ('인하 재현율 (Recall)', f"{report['인하']['recall']*100:.1f}%"),
        ('동결 재현율 (Recall)', f"{report['동결']['recall']*100:.1f}%"),
        ('', ''),
        ('정답 개수', f"{int(correct)} / {total} 개월"),
    ]
    y_start = 0.88
    ax3.text(0.5, 0.97, f'{eval_name} 기간 핵심 지표', transform=ax3.transAxes,
             fontsize=13, fontweight='bold', ha='center', va='top')
    for i, (label, value) in enumerate(info_lines):
        y = y_start - i * 0.1
        if label == '': continue
        ax3.text(0.15, y, label, transform=ax3.transAxes, fontsize=12, ha='left', va='top', color='#333')
        ax3.text(0.85, y, value, transform=ax3.transAxes, fontsize=12, ha='right', va='top', fontweight='bold', color='#4C72B0')
    from matplotlib.patches import FancyBboxPatch
    bg = FancyBboxPatch((0.05, 0.05), 0.9, 0.92, transform=ax3.transAxes,
                        boxstyle='round,pad=0.02', facecolor='#f0f4ff',
                        edgecolor='#4C72B0', linewidth=1.5, zorder=0)
    ax3.add_patch(bg)

    # ── (4) 월별 방향성 예측 타임라인 ──
    ax4 = fig.add_subplot(gs[1, :])
    dates = df[eval_mask]['date_ym'].values
    actual_labels = y_label[eval_mask].values
    pred_labels = cls_preds[eval_mask]
    x_pos = np.arange(len(dates))
    actual_scores = np.where(actual_labels == 0, -1, np.where(actual_labels == 2, 1, 0))
    pred_scores  = np.where(pred_labels == 0, -1, np.where(pred_labels == 2, 1, 0))
    for i, is_match in enumerate(actual_labels == pred_labels):
        bg_color = '#e8f5e9' if is_match else '#ffebee'
        ax4.axvspan(i - 0.45, i + 0.45, facecolor=bg_color, alpha=0.7, zorder=0)
    colors = ['#e53935' if s == -1 else '#1e88e5' if s == 1 else '#bdbdbd' for s in actual_scores]
    ax4.bar(x_pos, actual_scores, 0.6, label='실제 금리 방향', color=colors,
            edgecolor='white', linewidth=0.5, zorder=3)
    for i, (a_score, p_score) in enumerate(zip(actual_scores, pred_scores)):
        if a_score < 0:
            y_text = a_score - 0.15
            va = 'top'
        else:
            y_text = a_score + 0.1
            va = 'bottom'
        color = '#2e7d32' if a_score == p_score else '#c62828'
        lbl_str = { -1: '↓인하', 0: '―동결', 1: '↑인상' }[p_score]
        ax4.text(i, y_text, lbl_str, ha='center', va=va, fontsize=9, fontweight='bold', color=color)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(dates, rotation=45, ha='right', fontsize=9)
    ax4.set_yticks([-1, 0, 1])
    ax4.set_yticklabels(['인하', '동결', '인상'], fontsize=11)
    ax4.set_ylabel('금리 방향 기조', fontsize=11)
    ax4.set_title(f'{eval_name} 기간 월별 실제 기조 + 예측 라벨  (배경: 초록=정답, 빨강=오답)', fontsize=13, fontweight='bold')
    ax4.axhline(y=0, color='black', linewidth=1, zorder=2)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.set_ylim(-1.5, 1.5)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    fig_path = os.path.join(results_dir, f'{eval_name.lower()}_dashboard.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n[OK] 테스트 완료: {int(correct)}/{total} 맞춤")
    print(f"[PLOT] 대시보드 저장: {fig_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--valid', action='store_true', help='Validation mode')
    args = parser.parse_known_args()[0]
    test_model(valid_mode=args.valid)