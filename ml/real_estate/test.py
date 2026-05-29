import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from utils.preprocess import preprocess_data

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
    
    # Check if models exist
    model_path = os.path.join(models_dir, 'ensemble_model.pkl')
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
        
    # Get preprocessed data
    data = preprocess_data(test_months=24, vif_threshold=10.0)
    if data is None:
        print("[Error] Preprocessing failed.")
        return
        
    X_train_sc = data['X_train_sc']
    X_test_sc = data['X_test_sc']
    y_train = data['y_train']
    y_test = data['y_test']
    test_df = data['test_df']
    
    # -----------------------------------------
    # Predictions
    # -----------------------------------------
    # Baseline: Simple Linear Regression on the same features
    lr = LinearRegression().fit(X_train_sc, y_train)
    lr_pred = lr.predict(X_test_sc)
    
    # 2. Tuned Ridge Regression (Supreme Champion)
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=1.0).fit(X_train_sc, y_train)
    ridge_pred = ridge.predict(X_test_sc)
    
    # Individual models in Ensemble
    ind_preds = ensemble.get_individual_predictions(X_test_sc)
    
    # Ensemble prediction
    ensemble_pred = ensemble.predict(X_test_sc)
    
    # -----------------------------------------
    # Evaluation
    # -----------------------------------------
    n_samples = len(y_test)
    n_features = len(selected_features)
    
    results = {}
    results["LinearRegression (OLS)"] = evaluate(y_test, lr_pred)
    results["Ridge Regression (alpha=1.0)"] = evaluate(y_test, ridge_pred)
    
    for name, pred in ind_preds.items():
        results[f"Individual {name}"] = evaluate(y_test, pred)
        
    results["Ensemble (Weighted ML Blend)"] = evaluate(y_test, ensemble_pred)
    
    # Print results
    print("\n" + "=" * 75)
    print("Model Comparison and Evaluation (Last 24 Months Test Set)")
    print("=" * 75)
    print(f"{'Model Name':<30} {'RMSE':>10} {'R2':>10} {'MAE':>10} {'MSE':>12}")
    print("-" * 75)
    for model_name, metrics in results.items():
        print(f"{model_name:<30} {metrics['rmse']:>10.4f} {metrics['r2']:>10.4f} {metrics['mae']:>10.4f} {metrics['mse']:>12.6f}")
    print("=" * 75)
    
    # Save results
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    metrics_df = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model'})
    metrics_path = os.path.join(results_dir, 'evaluation_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
    print(f"Saved evaluation metrics to: {metrics_path}")
    
    # Save predictions alongside actual values for visualization/reporting
    pred_df = test_df[['date_ym', 'house_price_idx', 'next_change_rate']].copy()
    pred_df['pred_baseline'] = lr_pred
    pred_df['pred_ensemble'] = ensemble_pred
    pred_df['error_ensemble'] = pred_df['next_change_rate'] - pred_df['pred_ensemble']
    pred_df['abs_error_ensemble'] = pred_df['error_ensemble'].abs()
    
    pred_path = os.path.join(results_dir, 'predictions.csv')
    pred_df.to_csv(pred_path, index=False, encoding='utf-8-sig')
    print(f"Saved predictions comparison to: {pred_path}")
    
    # Find months with highest errors for local XAI waterfall analysis
    top_errors = pred_df.sort_values(by='abs_error_ensemble', ascending=False).head(3)
    print("\n[Outlier / Highest Error Months for local XAI]")
    for idx, row in top_errors.iterrows():
        print(f"  * Date: {row['date_ym']}, Actual Change Rate: {row['next_change_rate']:.4f}%, Predicted: {row['pred_ensemble']:.4f}% (Error: {row['error_ensemble']:.4f}%)")

if __name__ == '__main__':
    run_test()
