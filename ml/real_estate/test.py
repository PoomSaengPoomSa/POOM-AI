import os
import pickle
import numpy as np
import pandas as pd
import pymysql
from dotenv import load_dotenv, find_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor

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
            print(f"[CSV] Loaded preprocessed data successfully from local final_dataset.csv: {csv_path}")
            return df
        raise ValueError(f"No DB credentials and final CSV not found at: {csv_path}")
        
    try:
        connection = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=int(DB_PORT),
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
        print("[DB] Loaded preprocessed data successfully from MySQL table 'ml_realestate_preprocessed'.")
        return df
    except Exception as e:
        print(f"[Warning] MySQL query failed ({e}). Falling back to local final_dataset.csv...")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"[CSV] Loaded preprocessed data successfully from local CSV: {csv_path}")
            return df
        raise RuntimeError(f"Database connection failed and local CSV not found at: {csv_path}")

def run_test(valid_mode=False):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    
    # Check if models exist
    model_path = os.path.join(models_dir, 'ensemble_model.pkl')
    features_path = os.path.join(models_dir, 'selected_features.pkl')
    
    if not (os.path.exists(model_path) and os.path.exists(features_path)):
        print("[Error] Trained models not found. Run train.py first.")
        return
        
    with open(model_path, 'rb') as f:
        ensemble = pickle.load(f)
    with open(features_path, 'rb') as f:
        selected_features = pickle.load(f)
        
    # Get preprocessed data
    df = load_data_from_mysql()
    
    from model import RealEstateEnsembleRegressor as cfg
    df['date_ym'] = df['date_ym'].astype(str).str.strip()
    TARGET = "next_change_rate"

    if valid_mode:
        train_df = df[df['date_ym'] <= cfg.TRAIN_END].copy()
        test_df  = df[df['date_ym'].between(cfg.VALID_START, cfg.VALID_END)].copy()
        eval_name = "Validation"
    else:
        train_df = df[df['date_ym'] <= cfg.VALID_END].copy()
        test_df  = df[df['date_ym'].between(cfg.TEST_START, cfg.TEST_END)].copy()
        eval_name = "Test"
        
    train_clean = train_df.dropna(subset=[TARGET]).copy()
    test_clean  = test_df.dropna(subset=[TARGET]).copy()
    
    X_train_sc = train_clean[selected_features].values
    X_test_sc  = test_clean[selected_features].values
    y_train = train_clean[TARGET].values
    y_test  = test_clean[TARGET].values
    test_df = test_clean
    
    # -----------------------------------------
    # Predictions
    # -----------------------------------------
    # 1. Baseline: Simple Linear Regression (OLS)
    lr = LinearRegression().fit(X_train_sc, y_train)
    lr_pred = lr.predict(X_test_sc)
    
    # 2. Ridge Regression
    ridge = Ridge(alpha=1.0).fit(X_train_sc, y_train)
    ridge_pred = ridge.predict(X_test_sc)
    
    # 3. Lasso Regression
    lasso = Lasso(alpha=0.01).fit(X_train_sc, y_train)
    lasso_pred = lasso.predict(X_test_sc)
    
    # 4. ElasticNet Regression
    elastic = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X_train_sc, y_train)
    elastic_pred = elastic.predict(X_test_sc)
    
    # 5. RandomForest Regressor (Baseline Tree)
    rf_baseline = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1).fit(X_train_sc, y_train)
    rf_pred = rf_baseline.predict(X_test_sc)
    
    # 6. ExtraTrees Regressor (Baseline Tree)
    et_baseline = ExtraTreesRegressor(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1).fit(X_train_sc, y_train)
    et_pred = et_baseline.predict(X_test_sc)
    
    # 7. DecisionTree Regressor
    dt_baseline = DecisionTreeRegressor(max_depth=4, random_state=42).fit(X_train_sc, y_train)
    dt_pred = dt_baseline.predict(X_test_sc)
    
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
    results["Lasso Regression (alpha=0.01)"] = evaluate(y_test, lasso_pred)
    results["ElasticNet (alpha=0.1)"] = evaluate(y_test, elastic_pred)
    results["RandomForest (max_depth=4)"] = evaluate(y_test, rf_pred)
    results["ExtraTrees (max_depth=4)"] = evaluate(y_test, et_pred)
    results["DecisionTree (max_depth=4)"] = evaluate(y_test, dt_pred)
    
    for name, pred in ind_preds.items():
        results[f"Individual {name}"] = evaluate(y_test, pred)
        
    results["Our Model (Tuned XGBoost)"] = evaluate(y_test, ensemble_pred)
    
    eval_name = "Validation" if valid_mode else "Test"

    # Print results
    print("\n" + "=" * 75)
    print(f"Model Comparison and Evaluation ({eval_name} Set)")
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
    metrics_path = os.path.join(results_dir, f'{eval_name.lower()}_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
    print(f"Saved evaluation metrics to: {metrics_path}")
    
    # Save predictions alongside actual values for visualization/reporting
    pred_df = test_df[['date_ym', 'house_price_idx', 'next_change_rate']].copy()
    pred_df['pred_baseline'] = lr_pred
    pred_df['pred_ensemble'] = ensemble_pred
    pred_df['error_ensemble'] = pred_df['next_change_rate'] - pred_df['pred_ensemble']
    pred_df['abs_error_ensemble'] = pred_df['error_ensemble'].abs()
    
    pred_path = os.path.join(results_dir, f'{eval_name.lower()}_predictions.csv')
    pred_df.to_csv(pred_path, index=False, encoding='utf-8-sig')
    print(f"Saved predictions comparison to: {pred_path}")
    
    # Find months with highest errors for local XAI waterfall analysis
    top_errors = pred_df.sort_values(by='abs_error_ensemble', ascending=False).head(3)
    print("\n[Outlier / Highest Error Months for local XAI]")
    for idx, row in top_errors.iterrows():
        print(f"  * Date: {row['date_ym']}, Actual Change Rate: {row['next_change_rate']:.4f}%, Predicted: {row['pred_ensemble']:.4f}% (Error: {row['error_ensemble']:.4f}%)")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--valid', action='store_true', help='Validation mode')
    args = parser.parse_known_args()[0]
    run_test(valid_mode=args.valid)
