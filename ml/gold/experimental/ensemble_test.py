import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

import os
import pandas as pd
import numpy as np
# Add project root directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from ai.ml.gold.experimental.experiment import load_raw_data, preprocess_data_v2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

def run_ensemble_experiments():
    print("="*60)
    print("[Ensemble Test] Evaluating Model-Level Ensembles")
    print("="*60)
    
    raw_df = load_raw_data()
    X, y_cls, dates = preprocess_data_v2(raw_df)
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y_cls.iloc[:split_idx], y_cls.iloc[split_idx:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_val = float(num_neg) / float(num_pos)
    
    # Train the 3 optimized models
    print("Training XGBoost (Tuned V2)...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=180,
        learning_rate=0.02,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.7,
        scale_pos_weight=scale_pos_val,
        reg_alpha=1.5,
        reg_lambda=2.5,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train_scaled_df, y_train)
    
    print("Training LightGBM (Tuned)...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=180,
        learning_rate=0.02,
        max_depth=4,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.7,
        scale_pos_weight=scale_pos_val,
        reg_alpha=1.5,
        reg_lambda=2.5,
        random_state=42,
        verbosity=-1,
        n_jobs=-1
    )
    lgb_model.fit(X_train_scaled_df, y_train)
    
    print("Training Random Forest (Balanced)...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled_df, y_train)
    
    # Get probabilities of class 1 (Up) on test set
    prob_xgb = xgb_model.predict_proba(X_test_scaled_df)[:, 1]
    prob_lgb = lgb_model.predict_proba(X_test_scaled_df)[:, 1]
    prob_rf = rf_model.predict_proba(X_test_scaled_df)[:, 1]
    
    # Test different weights for Soft Voting
    weight_combinations = [
        (1.0, 0.0, 0.0, "Only XGBoost"),
        (0.0, 1.0, 0.0, "Only LightGBM"),
        (0.0, 0.0, 1.0, "Only Random Forest"),
        (0.5, 0.5, 0.0, "XGB(0.5) + LGB(0.5)"),
        (0.4, 0.4, 0.2, "XGB(0.4) + LGB(0.4) + RF(0.2)"),
        (0.33, 0.33, 0.33, "Equal (XGB 0.33 + LGB 0.33 + RF 0.33)"),
        (0.5, 0.3, 0.2, "XGB(0.5) + LGB(0.3) + RF(0.2)"),
        (0.3, 0.5, 0.2, "XGB(0.3) + LGB(0.5) + RF(0.2)"),
        (0.4, 0.3, 0.3, "XGB(0.4) + LGB(0.3) + RF(0.3)"),
    ]
    
    results = []
    
    print("\n" + "="*60)
    print("Evaluating Soft Voting Combinations (Threshold = 0.50)")
    print("="*60)
    
    for w_xgb, w_lgb, w_rf, label in weight_combinations:
        # Weighted average probability
        final_prob = (w_xgb * prob_xgb) + (w_lgb * prob_lgb) + (w_rf * prob_rf)
        y_pred = (final_prob >= 0.50).astype(int)
        
        acc = accuracy_score(y_test, y_pred) * 100
        f1_0 = f1_score(y_test, y_pred, pos_label=0)
        f1_1 = f1_score(y_test, y_pred, pos_label=1)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        conf = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = conf.ravel()
        
        results.append({
            "Weights": f"XGB:{w_xgb:.2f}/LGB:{w_lgb:.2f}/RF:{w_rf:.2f}",
            "Label": label,
            "Accuracy": f"{acc:.2f}%",
            "Class 0 F1": f"{f1_0:.4f}",
            "Class 1 F1": f"{f1_1:.4f}",
            "Macro F1": f"{f1_macro:.4f}",
            "TN": tn, "FP": fp, "FN": fn, "TP": tp
        })
        
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print("="*60)
    
    # We will also try tuning the threshold for the best ensemble weight combination
    # Let's find the best macro F1 weight combination first
    results_df['Macro_F1_val'] = results_df['Macro F1'].astype(float)
    best_idx = results_df['Macro_F1_val'].idxmax()
    best_row = results_df.loc[best_idx]
    
    print(f"\n🏆 Best Ensemble Combination: {best_row['Label']}")
    print(f"   Accuracy: {best_row['Accuracy']} | Macro F1: {best_row['Macro F1']}")
    
    # Let's see if we tune the threshold for this best ensemble
    # Let's find the weights
    best_weights_str = best_row['Weights']
    # Extract weights from string "XGB:0.40/LGB:0.40/RF:0.20"
    parts = best_weights_str.split('/')
    w_xgb = float(parts[0].split(':')[1])
    w_lgb = float(parts[1].split(':')[1])
    w_rf = float(parts[2].split(':')[1])
    
    final_prob = (w_xgb * prob_xgb) + (w_lgb * prob_lgb) + (w_rf * prob_rf)
    
    print("\n🔍 Tuning Decision Threshold for Best Ensemble...")
    thresholds = np.linspace(0.40, 0.60, 21)
    t_results = []
    for thresh in thresholds:
        y_pred = (final_prob >= thresh).astype(int)
        acc = accuracy_score(y_test, y_pred) * 100
        f1_0 = f1_score(y_test, y_pred, pos_label=0)
        f1_1 = f1_score(y_test, y_pred, pos_label=1)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        conf = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = conf.ravel()
        t_results.append({
            "Thresh": f"{thresh:.2f}",
            "Accuracy": f"{acc:.2f}%",
            "Class 0 F1": f"{f1_0:.4f}",
            "Class 1 F1": f"{f1_1:.4f}",
            "Macro F1": f"{f1_macro:.4f}",
            "TN": tn, "TP": tp
        })
    t_df = pd.DataFrame(t_results)
    print(t_df.iloc[::2].to_string(index=False)) # Print every second row

if __name__ == '__main__':
    run_ensemble_experiments()
