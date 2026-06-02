import os
import pickle
import numpy as np
import pandas as pd
import pymysql
from dotenv import load_dotenv, find_dotenv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
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


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "mae": round(mae, 4),
        "mse": round(mse, 6)
    }


def run_test():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Check if models exist
    model_path = os.path.join(models_dir, 'regressor.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    features_path = os.path.join(models_dir, 'selected_features.pkl')
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path)):
        print("[Error] Trained models/scalers not found. Run train.py first.")
        return
        
    with open(model_path, 'rb') as f:
        ensemble = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(features_path, 'rb') as f:
        selected_features = pickle.load(f)
        
    # Get data from MySQL
    df = load_data_from_mysql()
    df['date_ym'] = df['date_ym'].astype(str).str.strip()
    cfg = RealEstateEnsembleRegressor
    
    X_all = df[selected_features]
    y_target = df['next_change_rate']
    
    # Train/Test masks
    train_mask = df['date_ym'] <= cfg.TRAIN_END
    test_mask  = df['date_ym'] >= cfg.TEST_START
    print(f"   전체: {len(df)}건  |  Train: {train_mask.sum()}건  |  Test: {test_mask.sum()}건")

    X_train_sc = scaler.transform(X_all[train_mask])
    X_test_sc  = scaler.transform(X_all[test_mask])
    X_all_sc   = scaler.transform(X_all)
    
    # -----------------------------------------
    # Predictions
    # -----------------------------------------
    # Baseline: Simple Linear Regression on the same features
    lr = LinearRegression().fit(X_train_sc, y_target[train_mask])
    lr_pred = lr.predict(X_test_sc)
    
    # 2. Tuned Ridge Regression
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=1.0).fit(X_train_sc, y_target[train_mask])
    ridge_pred = ridge.predict(X_test_sc)
    
    # Individual models in Ensemble on Test set
    ind_preds_test = ensemble.get_individual_predictions(X_test_sc)
    
    # Ensemble prediction
    ensemble_pred_test = ensemble.predict(X_test_sc)
    ensemble_pred_all  = ensemble.predict(X_all_sc)
    
    # -----------------------------------------
    # Evaluation
    # -----------------------------------------
    y_test = y_target[test_mask]
    
    results = {}
    results["LinearRegression (OLS)"] = evaluate(y_test, lr_pred)
    results["Ridge Regression (alpha=1.0)"] = evaluate(y_test, ridge_pred)
    
    for name, pred in ind_preds_test.items():
        results[f"Individual {name}"] = evaluate(y_test, pred)
        
    results["Ensemble (Weighted ML Blend)"] = evaluate(y_test, ensemble_pred_test)
    
    # Print results
    print("\n" + "=" * 75)
    print("Model Comparison and Evaluation (Last 24 Months Test Set)")
    print("=" * 75)
    print(f"{'Model Name':<30} {'RMSE':>10} {'R2':>10} {'MAE':>10} {'MSE':>12}")
    print("-" * 75)
    for model_name, metrics in results.items():
        print(f"{model_name:<30} {metrics['rmse']:>10.4f} {metrics['r2']:>10.4f} {metrics['mae']:>10.4f} {metrics['mse']:>12.6f}")
    print("=" * 75)
    
    # Save evaluation metrics
    metrics_df = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model'})
    metrics_path = os.path.join(results_dir, 'evaluation_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
    print(f"Saved evaluation metrics to: {metrics_path}")
    
    # Save predictions alongside actual values
    result_df = pd.DataFrame({
        'date_ym':           df['date_ym'].values,
        'split':             np.where(train_mask, 'train', 'test'),
        'actual_change_rate': df['next_change_rate'].values,
        'pred_change_rate':   ensemble_pred_all,
        'error':             df['next_change_rate'].values - ensemble_pred_all,
        'abs_error':         np.abs(df['next_change_rate'].values - ensemble_pred_all),
        'match':             np.abs(df['next_change_rate'].values - ensemble_pred_all) < 0.1 # tolerance
    })
    
    save_path = os.path.join(results_dir, 'test_result.csv')
    legacy_pred_path = os.path.join(results_dir, 'predictions.csv') # compatibility
    
    test_result = result_df[result_df['split'] == 'test']
    test_result.to_csv(save_path, index=False, encoding='utf-8-sig')
    test_result.to_csv(legacy_pred_path, index=False, encoding='utf-8-sig')
    print(f"Saved predictions comparison to: {save_path}")

    # Metrics for train vs test RMSE comparison
    tr_rmse = np.sqrt(mean_squared_error(y_target[train_mask], ensemble_pred_all[train_mask]))
    te_rmse = results["Ensemble (Weighted ML Blend)"]["rmse"]
    tr_r2   = r2_score(y_target[train_mask], ensemble_pred_all[train_mask])
    te_r2   = results["Ensemble (Weighted ML Blend)"]["r2"]
    tr_mae  = mean_absolute_error(y_target[train_mask], ensemble_pred_all[train_mask])
    te_mae  = results["Ensemble (Weighted ML Blend)"]["mae"]
    
    test_metrics = pd.DataFrame([
        {
            '구분': 'Train',
            'RMSE': round(tr_rmse, 4),
            'R2': round(tr_r2, 4),
            'MAE': round(tr_mae, 4),
        },
        {
            '구분': 'Test',
            'RMSE': round(te_rmse, 4),
            'R2': round(te_r2, 4),
            'MAE': round(te_mae, 4),
        }
    ])
    test_metrics_path = os.path.join(results_dir, 'test_metrics.csv')
    test_metrics.to_csv(test_metrics_path, index=False, encoding='utf-8-sig')

    # ═══════════════════════════════════════════════
    # Visual Dashboard Generation (test_dashboard.png)
    # ═══════════════════════════════════════════════
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

    # ── (1) Train vs Test RMSE Comparison ──
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(['Train', 'Test'], [tr_rmse, te_rmse], width=0.5,
                   color=['#4C72B0', '#DD8452'], edgecolor='white')
    ax1.set_ylabel('RMSE', fontsize=11)
    ax1.set_title('Train vs Test RMSE 비교', fontsize=13, fontweight='bold')
    max_val = max(tr_rmse, te_rmse)
    ax1.set_ylim(0, max_val * 1.25)
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max_val * 0.02),
                 f'{bar.get_height():.4f}', ha='center', fontsize=11, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ── (2) Actual vs Predicted Scatter Plot (Test Set) ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(y_test, ensemble_pred_test, color='#4C72B0', alpha=0.8, edgecolors='white', s=70, label='예측 데이터')
    
    # 45-degree reference line
    min_pt = min(y_test.min(), ensemble_pred_test.min())
    max_pt = max(y_test.max(), ensemble_pred_test.max())
    ax2.plot([min_pt, max_pt], [min_pt, max_pt], color='#E24A33', linestyle='--', linewidth=2, label='정답 기준선')
    
    ax2.set_xlabel('실제 변동률 (%)', fontsize=11)
    ax2.set_ylabel('예측 변동률 (%)', fontsize=11)
    ax2.set_title('실제값 vs 예측값 분포 (Test)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # ── (3) Summary Scorecard Panel ──
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    info_lines = [
        ('Test RMSE', f"{te_rmse:.4f}"),
        ('Test MAE', f"{te_mae:.4f}%"),
        ('Test R2 Score', f"{te_r2:.4f}"),
        ('', ''),
        ('Train RMSE', f"{tr_rmse:.4f}"),
        ('Train R2 Score', f"{tr_r2:.4f}"),
        ('', ''),
        ('평가 기간 (개월)', f"{len(y_test)} 개월"),
    ]
    y_start = 0.88
    ax3.text(0.5, 0.97, '부동산 모델 핵심 평가 지표', transform=ax3.transAxes,
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

    # ── (4) Monthly Actual vs Predicted Change Rate Timeline ──
    ax4 = fig.add_subplot(gs[1, :])
    dates = df[test_mask]['date_ym'].values
    x_pos = np.arange(len(dates))
    
    ax4.plot(x_pos, y_test, marker='o', linewidth=2.5, color='#4C72B0', label='실제 변동률')
    ax4.plot(x_pos, ensemble_pred_test, marker='s', linewidth=2.5, color='#55A868', linestyle='--', label='예측 변동률')
    
    # Fill discrepancy region
    ax4.fill_between(x_pos, y_test, ensemble_pred_test, color='#C4AD66', alpha=0.2, label='예측 오차')
    
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(dates, rotation=45, ha='right', fontsize=9)
    ax4.set_ylabel('지수 변동률 (%)', fontsize=11)
    ax4.set_title('Test 기간 월별 실제 지수 변동률 vs 모델 예측치 추이', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=11, loc='upper left')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    fig_path = os.path.join(results_dir, 'test_dashboard.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n[OK] 테스트 완료!")
    print(f"[PLOT] 대시보드 저장: {fig_path}")
    
    # Find months with highest errors for local XAI waterfall analysis
    top_errors = test_result.sort_values(by='abs_error', ascending=False).head(3)
    print("\n[Outlier / Highest Error Months for local XAI]")
    for idx, row in top_errors.iterrows():
        print(f"  * Date: {row['date_ym']}, Actual Change Rate: {row['actual_change_rate']:.4f}%, Predicted: {row['pred_change_rate']:.4f}% (Error: {row['error']:.4f}%)")


if __name__ == '__main__':
    run_test()
