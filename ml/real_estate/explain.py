import os
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from utils.preprocess import preprocess_data

def save_contributions_to_mysql(features, shap_values):
    import pymysql
    from dotenv import load_dotenv, find_dotenv
    import numpy as np
    
    # 1. Calculate absolute mean SHAP values for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # 2. Map to base variable names (removing _change, _lag1, _ma3, seasonality, etc.)
    base_mapping = {
        "house_price_idx": "house_price_idx",
        "kr_cpi": "kr_cpi",
        "kr_unemployment": "kr_unemployment",
        "kr_base_rate": "kr_base_rate",
        "kr_mortgage_rate": "kr_mortgage_rate",
        "kospi200": "kospi200",
        "apt_trade_count": "apt_trade_count",
        "kr_m2": "kr_m2",
        "buyer_dominance": "buyer_dominance"
    }
    
    grouped_shap = {}
    for feat, val in zip(features, mean_abs_shap):
        # find matching base variable
        base_var = None
        for k in base_mapping.keys():
            if feat.startswith(k):
                base_var = base_mapping[k]
                break
        
        if not base_var:
            base_var = feat # fallback
            
        grouped_shap[base_var] = grouped_shap.get(base_var, 0.0) + val
        
    # 3. Normalize weights so they sum to 1.0 (or to 100 in backend)
    total_shap = sum(grouped_shap.values())
    if total_shap > 0:
        contributions = {k: v / total_shap for k, v in grouped_shap.items()}
    else:
        contributions = {k: 1.0 / len(grouped_shap) for k in grouped_shap.keys()}
        
    # 4. Insert into database
    load_dotenv(find_dotenv())
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    DB_NAME = os.getenv('DB_NAME')
    
    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
        print("[Warning] Missing DB config. Skipping SHAP contributions DB save.")
        return
        
    try:
        connection = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=int(DB_PORT),
            charset='utf8mb4'
        )
        try:
            with connection.cursor() as cursor:
                # Delete old contributions for real_estate
                cursor.execute("DELETE FROM economic_indicator_contribution WHERE type = 'real_estate'")
                
                # Insert new contributions
                sql = """
                INSERT INTO economic_indicator_contribution (type, variable, weight)
                VALUES (%s, %s, %s)
                """
                for var, weight in contributions.items():
                    cursor.execute(sql, ("real_estate", var, float(weight)))
            connection.commit()
            print(f"[DB] Successfully saved real_estate SHAP contributions to MySQL ({len(contributions)} features).")
        finally:
            connection.close()
    except Exception as e:
        print(f"[Error] Failed to save SHAP contributions to MySQL: {e}")


def run_explain(valid_mode=False):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Load model and features
    model_path = os.path.join(models_dir, 'ensemble_model.pkl')
    features_path = os.path.join(models_dir, 'selected_features.pkl')
    
    if not (os.path.exists(model_path) and os.path.exists(features_path)):
        print("[Error] Models not found. Run train.py first.")
        return
        
    with open(model_path, 'rb') as f:
        ensemble = pickle.load(f)
    with open(features_path, 'rb') as f:
        selected_features = pickle.load(f)
        
    # Get preprocessed data
    data = preprocess_data(vif_threshold=20.0, valid_mode=valid_mode)
    if data is None:
        print("[Error] Preprocessing failed.")
        return
        
    X_train_sc = data['X_train_sc']
    X_test_sc = data['X_test_sc']
    y_test = data['y_test']
    test_df = data['test_df']
    
    # Convert standardized test set back to dataframe for beautiful SHAP labels
    X_test_df = pd.DataFrame(X_test_sc, columns=selected_features)
    X_train_df = pd.DataFrame(X_train_sc, columns=selected_features)
    
    print("\n" + "=" * 55)
    print("Computing SHAP Values (Explainable AI)")
    print("=" * 55)
    
    # -----------------------------------------
    # Compute SHAP for each model and average them
    # -----------------------------------------
    shap_values_list = []
    
    for name, model in ensemble.models.items():
        print(f"  * Explaining {name}...")
        try:
            # TreeExplainer is fast and accurate for tree ensembles
            explainer = shap.TreeExplainer(model, data=X_train_df)
            sv = explainer(X_test_df)
            shap_values_list.append(sv.values)
        except Exception as e:
            print(f"    - Explainer fallback for {name} due to: {e}")
            explainer = shap.Explainer(model, X_train_df)
            sv = explainer(X_test_df)
            shap_values_list.append(sv.values)
            
    # Average SHAP values for the ensemble
    mean_shap_values = np.mean(shap_values_list, axis=0)
    
    # Create a custom Explanation object for SHAP plotting
    base_values_list = []
    for name, model in ensemble.models.items():
        try:
            explainer = shap.TreeExplainer(model, data=X_train_df)
            base_values_list.append(explainer.expected_value)
        except:
            explainer = shap.Explainer(model, X_train_df)
            base_values_list.append(explainer.expected_value)
            
    mean_base_value = np.mean(base_values_list)
    if isinstance(mean_base_value, np.ndarray) and len(mean_base_value) > 0:
        mean_base_value = mean_base_value[0]
        
    ensemble_explanation = shap.Explanation(
        values=mean_shap_values,
        base_values=np.repeat(mean_base_value, len(X_test_df)),
        data=X_test_df.values,
        feature_names=selected_features
    )
    
    # -----------------------------------------
    # 1. Save Beeswarm Plot (Global Feature Impact)
    # -----------------------------------------
    print("  Generating SHAP Beeswarm Plot...")
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(ensemble_explanation, show=False)
    plt.tight_layout()
    beeswarm_path = os.path.join(results_dir, 'shap_beeswarm.png')
    plt.savefig(beeswarm_path, dpi=150)
    plt.close()
    print(f"    - Saved beeswarm plot to: {beeswarm_path}")
    
    # -----------------------------------------
    # 2. Save Feature Importance Bar Chart
    # -----------------------------------------
    print("  Generating SHAP Feature Importance Plot...")
    plt.figure(figsize=(10, 6))
    shap.plots.bar(ensemble_explanation, show=False)
    plt.tight_layout()
    bar_path = os.path.join(results_dir, 'shap_importance.png')
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"    - Saved feature importance to: {bar_path}")
    
    # -----------------------------------------
    # 3. Save Local Waterfall Plots for High-Error Months
    # -----------------------------------------
    eval_name = "Validation" if valid_mode else "Test"
    predictions_path = os.path.join(results_dir, f'{eval_name.lower()}_predictions.csv')
    if os.path.exists(predictions_path):
        pred_df = pd.read_csv(predictions_path)
        # Find index of month with maximum absolute error
        max_err_idx = pred_df['abs_error_ensemble'].idxmax()
        max_err_row = pred_df.loc[max_err_idx]
        
        print(f"  Generating Waterfall Plot for Max Error Month: {max_err_row['date_ym']}...")
        plt.figure(figsize=(10, 6))
        # Plot waterfall for this specific sample
        shap.plots.waterfall(ensemble_explanation[max_err_idx], show=False)
        plt.tight_layout()
        waterfall_path = os.path.join(results_dir, f"shap_waterfall_{max_err_row['date_ym']}.png")
        plt.savefig(waterfall_path, dpi=150)
        plt.close()
        print(f"    - Saved waterfall plot to: {waterfall_path}")
        
        # Save indices of top 3 errors for the text report
        top_err_indices = pred_df.sort_values(by='abs_error_ensemble', ascending=False).index.tolist()[:3]
        
        # Save SHAP values CSV for reporting
        shap_df = pd.DataFrame(mean_shap_values, columns=[f"shap_{f}" for f in selected_features])
        shap_df['date_ym'] = test_df['date_ym'].values
        shap_df.to_csv(os.path.join(results_dir, 'shap_values.csv'), index=False, encoding='utf-8-sig')
        
        # Save dynamic contributions to MySQL DB for real-time dashboard binding
        save_contributions_to_mysql(selected_features, mean_shap_values)
        
    print("\nSHAP XAI Analysis Completed Successfully!")
 
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--valid', action='store_true', help='Validation mode')
    args = parser.parse_known_args()[0]
    run_explain(valid_mode=args.valid)
