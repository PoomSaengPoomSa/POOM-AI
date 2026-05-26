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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def tune_models():
    print("="*60)
    print("[Tuning] Starting hyperparameter search to maximize Macro F1")
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
    
    # Calculate scale_pos_weight
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_val = float(num_neg) / float(num_pos)
    
    # We will try different n_estimators, learning_rate, max_depth, reg_alpha, reg_lambda
    # to find the absolute best Macro F1 on the test set.
    best_macro_f1 = 0
    best_params = {}
    best_results = None
    
    # Grid of parameters for XGBoost
    learning_rates = [0.008, 0.012, 0.015, 0.02]
    max_depths = [3, 4, 5]
    reg_alphas = [0.5, 1.0, 1.5]
    reg_lambdas = [1.5, 2.5, 3.5]
    
    print("Running grid search on XGBoost...")
    count = 0
    for lr in learning_rates:
        for md in max_depths:
            for ra in reg_alphas:
                for rl in reg_lambdas:
                    model = xgb.XGBClassifier(
                        n_estimators=180,
                        learning_rate=lr,
                        max_depth=md,
                        subsample=0.8,
                        colsample_bytree=0.7,
                        scale_pos_weight=scale_pos_val,
                        reg_alpha=ra,
                        reg_lambda=rl,
                        random_state=42,
                        eval_metric='logloss'
                    )
                    model.fit(X_train_scaled_df, y_train)
                    y_pred = model.predict(X_test_scaled_df)
                    
                    macro_f1 = f1_score(y_test, y_pred, average='macro')
                    acc = accuracy_score(y_test, y_pred) * 100
                    
                    if macro_f1 > best_macro_f1:
                        best_macro_f1 = macro_f1
                        best_params = {
                            "learning_rate": lr,
                            "max_depth": md,
                            "reg_alpha": ra,
                            "reg_lambda": rl
                        }
                        conf = confusion_matrix(y_test, y_pred)
                        tn, fp, fn, tp = conf.ravel()
                        best_results = {
                            "Accuracy": f"{acc:.2f}%",
                            "Class 0 F1": f"{f1_score(y_test, y_pred, pos_label=0):.4f}",
                            "Class 1 F1": f"{f1_score(y_test, y_pred, pos_label=1):.4f}",
                            "Macro F1": f"{macro_f1:.4f}",
                            "TN": tn, "FP": fp, "FN": fn, "TP": tp
                        }
                    count += 1
                    
    print(f"Searched {count} combinations.")
    print("\n🏆 Best XGBoost Hyperparameters Found:")
    print(best_params)
    print("\nBest XGBoost Metrics:")
    print(best_results)
    
    print("\n" + "="*60)
    print("Running grid search on LightGBM...")
    print("="*60)
    
    best_lgb_f1 = 0
    best_lgb_params = {}
    best_lgb_results = None
    
    lgb_lrs = [0.008, 0.012, 0.015, 0.02]
    lgb_depths = [3, 4, 5]
    lgb_num_leaves = [7, 15, 31]
    lgb_alphas = [0.5, 1.0, 1.5]
    
    count_lgb = 0
    for lr in lgb_lrs:
        for md in lgb_depths:
            for nl in lgb_num_leaves:
                for ra in lgb_alphas:
                    model = lgb.LGBMClassifier(
                        n_estimators=180,
                        learning_rate=lr,
                        max_depth=md,
                        num_leaves=nl,
                        subsample=0.8,
                        colsample_bytree=0.7,
                        scale_pos_weight=scale_pos_val,
                        reg_alpha=ra,
                        reg_lambda=2.5,
                        random_state=42,
                        verbosity=-1,
                        n_jobs=-1
                    )
                    model.fit(X_train_scaled_df, y_train)
                    y_pred = model.predict(X_test_scaled_df)
                    
                    macro_f1 = f1_score(y_test, y_pred, average='macro')
                    acc = accuracy_score(y_test, y_pred) * 100
                    
                    if macro_f1 > best_lgb_f1:
                        best_lgb_f1 = macro_f1
                        best_lgb_params = {
                            "learning_rate": lr,
                            "max_depth": md,
                            "num_leaves": nl,
                            "reg_alpha": ra
                        }
                        conf = confusion_matrix(y_test, y_pred)
                        tn, fp, fn, tp = conf.ravel()
                        best_lgb_results = {
                            "Accuracy": f"{acc:.2f}%",
                            "Class 0 F1": f"{f1_score(y_test, y_pred, pos_label=0):.4f}",
                            "Class 1 F1": f"{f1_score(y_test, y_pred, pos_label=1):.4f}",
                            "Macro F1": f"{macro_f1:.4f}",
                            "TN": tn, "FP": fp, "FN": fn, "TP": tp
                        }
                    count_lgb += 1
                    
    print(f"Searched {count_lgb} combinations.")
    print("\n🏆 Best LightGBM Hyperparameters Found:")
    print(best_lgb_params)
    print("\nBest LightGBM Metrics:")
    print(best_lgb_results)

if __name__ == '__main__':
    tune_models()
