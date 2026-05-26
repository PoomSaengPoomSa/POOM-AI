import os
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from utils.preprocess import preprocess_data

def run_explain():
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
    data = preprocess_data(test_months=24, vif_threshold=10.0)
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
    # Load prediction errors from test
    predictions_path = os.path.join(results_dir, 'predictions.csv')
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
        
    print("\nSHAP XAI Analysis Completed Successfully!")

if __name__ == '__main__':
    run_explain()
