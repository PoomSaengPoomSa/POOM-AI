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
)
from model import InterestRateEnsembleModel

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

def test_model():
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
    # Train/Test 분리
    train_mask = df['date_ym'] <= cfg.TRAIN_END
    test_mask  = df['date_ym'] >= cfg.TEST_START
    print(f"   전체: {len(df)}건  |  Train: {train_mask.sum()}건  |  Test: {test_mask.sum()}건")

    # ═══════════════════════════════════════════════
    # 2. 통계치 및 분류 수행

    # ═══════════════════════════════════════════════
    label_names = ['인하', '동결', '인상']
    cls_preds = classifier.predict(X_all)
    cls_proba = classifier.predict_proba(X_all)
    tr_acc = accuracy_score(y_label[train_mask], cls_preds[train_mask])
    te_acc = accuracy_score(y_label[test_mask], cls_preds[test_mask])
    print(f"\n{'='*55}")
    print("[TEST] Test 성능 점검")
    print(f"{'='*55}")
    print(f"   Train 분류정확도: {tr_acc*100:.1f}%")
    print(f"   Test 분류정확도:  {te_acc*100:.1f}%")
    gap = (tr_acc - te_acc) * 100
    if gap > 20:
        print(f"   [WARNING] 과적합 의심 (차이: {gap:.1f}%p)")
    else:
        print(f"   [OK] 과적합 없음 (차이: {gap:.1f}%p)")
    print(f"\n[REPORT] Test 상세 리포트")
    print(classification_report(
        y_label[test_mask], cls_preds[test_mask],
        labels=[0, 1, 2], target_names=label_names, zero_division=0,
    ))

    # ═══════════════════════════════════════════════
    # 3. 전수 결과 저장

    # ═══════════════════════════════════════════════
    result_df = pd.DataFrame({
        'date_ym':           df['date_ym'].values,
        'split':             np.where(train_mask, 'train', 'test'),
        'actual_label':      df['label'].values,
        'pred_direction':    [label_names[int(p)] for p in cls_preds],
        'pred_proba_인하':   cls_proba[:, 0],
        'pred_proba_동결':   cls_proba[:, 1],
        'pred_proba_인상':   cls_proba[:, 2],
        'match':             y_label.values == cls_preds,
    })
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, 'test_result.csv')
    test_result = result_df[result_df['split'] == 'test']
    test_result.to_csv(save_path, index=False, encoding='utf-8-sig')
    total = len(test_result)
    correct = test_result['match'].sum()
    metrics = pd.DataFrame([
        {'구분': 'Train', '분류정확도(%)': round(tr_acc*100, 1)},
        {'구분': 'Test', '분류정확도(%)': round(te_acc*100, 1)}
    ])
    metrics_path = os.path.join(results_dir, 'test_metrics.csv')
    metrics.to_csv(metrics_path, index=False, encoding='utf-8-sig')

    # ═══════════════════════════════════════════════
    # 4. 대시보드 시각화 저장

    # ═══════════════════════════════════════════════
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

    # ── (1) Train vs Test 정확도 비교 ──
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(['Train', 'Test'], [tr_acc*100, te_acc*100], width=0.5,
                   color=['#4C72B0', '#DD8452'], edgecolor='white')
    ax1.set_ylabel('분류 정확도 (%)', fontsize=11)
    ax1.set_title('Train vs Test 성능 비교', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 105)
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{bar.get_height():.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ── (2) 혼동 행렬 히트맵 ──
    ax2 = fig.add_subplot(gs[0, 1])
    cm_display = confusion_matrix(y_label[test_mask], cls_preds[test_mask], labels=[0, 1, 2])
    im = ax2.imshow(cm_display, cmap='Blues', aspect='auto')
    ax2.set_xticks([0, 1, 2])
    ax2.set_yticks([0, 1, 2])
    ax2.set_xticklabels(['인하', '동결', '인상'], fontsize=11)
    ax2.set_yticklabels(['인하', '동결', '인상'], fontsize=11)
    ax2.set_xlabel('예측', fontsize=11)
    ax2.set_ylabel('실제', fontsize=11)
    ax2.set_title('혼동 행렬 (Test)', fontsize=13, fontweight='bold')
    for i in range(3):
        for j in range(3):
            val = cm_display[i, j]
            color = 'white' if val > cm_display.max() / 2 else 'black'
            ax2.text(j, i, str(val), ha='center', va='center',
                     fontsize=16, fontweight='bold', color=color)

    # ── (3) 성능 요약 카드 ──
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    # Classification Report 파싱
    report = classification_report(y_label[test_mask], cls_preds[test_mask], labels=[0, 1, 2], target_names=label_names, zero_division=0, output_dict=True)
    info_lines = [
        ('Test 분류 정확도', f"{te_acc*100:.1f}%"),
        ('', ''),
        ('인상 재현율 (Recall)', f"{report['인상']['recall']*100:.1f}%"),
        ('인하 재현율 (Recall)', f"{report['인하']['recall']*100:.1f}%"),
        ('동결 재현율 (Recall)', f"{report['동결']['recall']*100:.1f}%"),
        ('', ''),
        ('정답 개수', f"{int(correct)} / {total} 개월"),
    ]
    y_start = 0.88
    ax3.text(0.5, 0.97, 'Test 기간 핵심 지표', transform=ax3.transAxes,
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
    dates = df[test_mask]['date_ym'].values
    actual_labels = y_label[test_mask].values
    pred_labels = cls_preds[test_mask]
    x_pos = np.arange(len(dates))
    # 방향성 스코어 매핑 (시각화용): 인하 -1, 동결 0, 인상 1
    actual_scores = np.where(actual_labels == 0, -1, np.where(actual_labels == 2, 1, 0))
    pred_scores  = np.where(pred_labels == 0, -1, np.where(pred_labels == 2, 1, 0))
    # 배경: 일치하면 연녹색, 틀리면 연빨간색
    for i, is_match in enumerate(actual_labels == pred_labels):
        bg_color = '#e8f5e9' if is_match else '#ffebee'
        ax4.axvspan(i - 0.45, i + 0.45, facecolor=bg_color, alpha=0.7, zorder=0)
    # 실제 라벨 바
    colors = ['#e53935' if s == -1 else '#1e88e5' if s == 1 else '#bdbdbd' for s in actual_scores]
    ax4.bar(x_pos, actual_scores, 0.6, label='실제 금리 방향', color=colors,
            edgecolor='white', linewidth=0.5, zorder=3)
    # 예측 텍스트 달기
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
    ax4.set_title('Test 기간 월별 실제 기조 + 예측 라벨  (배경: 초록=정답, 빨강=오답)', fontsize=13, fontweight='bold')
    ax4.axhline(y=0, color='black', linewidth=1, zorder=2)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.set_ylim(-1.5, 1.5)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    fig_path = os.path.join(results_dir, 'test_dashboard.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n[OK] 테스트 완료: {int(correct)}/{total} 맞춤")
    print(f"[PLOT] 대시보드 저장: {fig_path}")
if __name__ == '__main__':
    test_model()