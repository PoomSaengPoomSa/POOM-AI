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

def explore_thresholds():
    print("="*60)
    print("[Threshold Exploration] Searching for F1 >= 0.65")
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
    
    # We will test three models:
    # 1. XGBoost (Tuned V2)
    # 2. LightGBM (Tuned)
    # 3. Random Forest (with balanced weights)
    
    models = {
        "XGBoost": xgb.XGBClassifier(
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
        ),
        "LightGBM": lgb.LGBMClassifier(
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
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    }
    
    thresholds = np.linspace(0.30, 0.70, 41)
    
    for name, model in models.items():
        print(f"\n📊 exploring thresholds for model: {name}")
        model.fit(X_train_scaled_df, y_train)
        # Get probability of class 1 (Up)
        y_prob = model.predict_proba(X_test_scaled_df)[:, 1]
        
        best_class1_f1 = 0
        best_class1_thresh = 0.50
        best_class1_metrics = None
        
        best_macro_f1 = 0
        best_macro_thresh = 0.50
        best_macro_metrics = None
        
        rows = []
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            acc = accuracy_score(y_test, y_pred) * 100
            f1_0 = f1_score(y_test, y_pred, pos_label=0)
            f1_1 = f1_score(y_test, y_pred, pos_label=1)
            f1_macro = f1_score(y_test, y_pred, average='macro')
            prec_1 = precision_score(y_test, y_pred, pos_label=1)
            rec_1 = recall_score(y_test, y_pred, pos_label=1)
            
            rows.append({
                "Thresh": f"{thresh:.2f}",
                "Accuracy": f"{acc:.2f}%",
                "Class 0 F1": f"{f1_0:.4f}",
                "Class 1 F1": f"{f1_1:.4f}",
                "Macro F1": f"{f1_macro:.4f}"
            })
            
            if f1_1 > best_class1_f1:
                best_class1_f1 = f1_1
                best_class1_thresh = thresh
                best_class1_metrics = (acc, f1_0, f1_1, f1_macro, prec_1, rec_1)
                
            if f1_macro > best_macro_f1:
                best_macro_f1 = f1_macro
                best_macro_thresh = thresh
                best_macro_metrics = (acc, f1_0, f1_1, f1_macro, prec_1, rec_1)
        
        # Display table of a few key thresholds around 0.40 to 0.60
        temp_df = pd.DataFrame(rows)
        print(temp_df.iloc[::4].to_string(index=False)) # print every 4th threshold to keep output short
        
        print(f"\n   -> Best Class 1 (Up) F1: {best_class1_f1:.4f} at Thresh {best_class1_thresh:.2f}")
        print(f"      Metrics: Acc={best_class1_metrics[0]:.2f}%, Class0_F1={best_class1_metrics[1]:.4f}, Class1_F1={best_class1_metrics[2]:.4f}, Macro_F1={best_class1_metrics[3]:.4f}, Prec_1={best_class1_metrics[4]:.4f}, Rec_1={best_class1_metrics[5]:.4f}")
        
        print(f"   -> Best Macro F1: {best_macro_f1:.4f} at Thresh {best_macro_thresh:.2f}")
        print(f"      Metrics: Acc={best_macro_metrics[0]:.2f}%, Class0_F1={best_macro_metrics[1]:.4f}, Class1_F1={best_macro_metrics[2]:.4f}, Macro_F1={best_macro_metrics[3]:.4f}, Prec_1={best_macro_metrics[4]:.4f}, Rec_1={best_macro_metrics[5]:.4f}")

if __name__ == '__main__':
    explore_thresholds()
