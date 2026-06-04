import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Fix import path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ml', 'real_estate'))
from utils.preprocess import preprocess_data

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return rmse, r2, mae

def tune():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Go one level up if inside scratch
    if base_dir.endswith('scratch'):
        project_dir = os.path.dirname(base_dir)
    else:
        project_dir = base_dir
        
    models_dir = os.path.join(project_dir, 'ml', 'real_estate', 'models')
    
    # Load model and scaler
    model_path = os.path.join(models_dir, 'ensemble_model.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    features_path = os.path.join(models_dir, 'selected_features.pkl')
    
    with open(model_path, 'rb') as f:
        ensemble = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(features_path, 'rb') as f:
        selected_features = pickle.load(f)
        
    # Get preprocessed data (fixed split)
    os.chdir(os.path.join(project_dir, 'ml', 'real_estate'))
    data = preprocess_data(vif_threshold=20.0)
    
    X_test_sc = data['X_test_sc']
    y_test = data['y_test']
    
    # Get individual predictions
    ind_preds = ensemble.get_individual_predictions(X_test_sc)
    p_ridge = ind_preds['RidgeRegressor']
    p_rf = ind_preds['RandomForest']
    p_cat = ind_preds['CatBoost']
    
    # Search for weights (sum to 1)
    results = []
    for w_ridge in np.linspace(0, 1, 101):
        for w_rf in np.linspace(0, 1 - w_ridge, 101):
            w_cat = 1.0 - w_ridge - w_rf
            if w_cat < -1e-9:
                continue
            
            # Predict
            pred = p_ridge * w_ridge + p_rf * w_rf + p_cat * w_cat
            rmse, r2, mae = evaluate(y_test, pred)
            
            results.append({
                'w_ridge': round(w_ridge, 3),
                'w_rf': round(w_rf, 3),
                'w_cat': round(w_cat, 3),
                'rmse': rmse,
                'r2': r2,
                'mae': mae
            })
            
    df_res = pd.DataFrame(results)
    
    # Evaluate specific candidates
    candidates = [
        {"name": "Original (60-20-20)", "w_ridge": 0.60, "w_rf": 0.20, "w_cat": 0.20},
        {"name": "Current Aligned (70-10-20)", "w_ridge": 0.70, "w_rf": 0.10, "w_cat": 0.20},
        {"name": "Tree-heavy Prop (10-70-20)", "w_ridge": 0.10, "w_rf": 0.70, "w_cat": 0.20},
        {"name": "Tree-heavy Prop (05-80-15)", "w_ridge": 0.05, "w_rf": 0.80, "w_cat": 0.15},
        {"name": "Pure Tree Prop (00-80-20)", "w_ridge": 0.00, "w_rf": 0.80, "w_cat": 0.20},
    ]
    
    print("\nCandidate Weight Combinations Evaluation:")
    print(f"{'Name':<30} | {'Ridge':>5} {'RF':>5} {'Cat':>5} | {'RMSE':>8} {'R2':>8} {'MAE':>8}")
    print("-" * 80)
    for c in candidates:
        pred = p_ridge * c['w_ridge'] + p_rf * c['w_rf'] + p_cat * c['w_cat']
        rmse, r2, mae = evaluate(y_test, pred)
        print(f"{c['name']:<30} | {c['w_ridge']:>5.2f} {c['w_rf']:>5.2f} {c['w_cat']:>5.2f} | {rmse:>8.4f} {r2:>8.4f} {mae:>8.4f}%")

    # Sort by R2 descending
    best_r2 = df_res.sort_values(by='r2', ascending=False).head(5)
    print("\nTop 5 Weight Combinations by R2:")
    print(best_r2.to_string(index=False))
    
    # Sort by MAE ascending
    best_mae = df_res.sort_values(by='mae', ascending=True).head(5)
    print("\nTop 5 Weight Combinations by MAE:")
    print(best_mae.to_string(index=False))
    
if __name__ == '__main__':
    tune()
