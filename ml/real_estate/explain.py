import os
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import pymysql
from dotenv import load_dotenv, find_dotenv

def save_contributions_to_mysql(features, shap_values):
    import pymysql
    from dotenv import load_dotenv, find_dotenv
    import numpy as np
    
    # 1. Calculate absolute mean SHAP values for each feature
    shap_values = np.array(shap_values)
    if shap_values.ndim == 1:
        mean_abs_shap = np.abs(shap_values)
    else:
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
    y_test  = test_clean[TARGET].values
    test_df = test_clean
    
    # Convert standardized test set back to dataframe for beautiful SHAP labels
    X_test_df = pd.DataFrame(X_test_sc, columns=selected_features)
    X_train_df = pd.DataFrame(X_train_sc, columns=selected_features)
    
    # Load latest row for explanation (aligned with predict.py)
    latest_row = df.iloc[-1:]
    latest_date = latest_row['date_ym'].values[0]
    X_latest = latest_row[selected_features].copy()
    for col in X_latest.columns:
        X_latest[col] = pd.to_numeric(X_latest[col], errors='coerce')
    X_latest_df = pd.DataFrame(X_latest.values, columns=selected_features)
    
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
    # Compute SHAP for the latest row (XAI explanation target)
    # -----------------------------------------
    print("  Computing SHAP for the latest prediction row...")
    shap_values_latest_list = []
    for name, model in ensemble.models.items():
        try:
            explainer = shap.TreeExplainer(model, data=X_train_df)
            sv = explainer(X_latest_df)
            shap_values_latest_list.append(sv.values)
        except Exception as e:
            explainer = shap.Explainer(model, X_train_df)
            sv = explainer(X_latest_df)
            shap_values_latest_list.append(sv.values)
            
    mean_shap_values_latest = np.mean(shap_values_latest_list, axis=0) # shape (1, n_features)
    
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
        
        # Save SHAP values CSV for reporting (representing the latest prediction)
        shap_df = pd.DataFrame(mean_shap_values_latest, columns=[f"shap_{f}" for f in selected_features])
        shap_df['date_ym'] = [latest_date]
        shap_df.to_csv(os.path.join(results_dir, 'shap_values.csv'), index=False, encoding='utf-8-sig')
        print(f"    - Saved latest SHAP values CSV to: {os.path.join(results_dir, 'shap_values.csv')}")
        
        # Save dynamic contributions to MySQL DB for real-time dashboard binding
        save_contributions_to_mysql(selected_features, mean_shap_values_latest[0])
        
        # ── SHAP Waterfall Plot for the Latest Predicted Row ──
        try:
            latest_explanation = shap.Explanation(
                values=mean_shap_values_latest[0],
                base_values=mean_base_value,
                data=X_latest_df.iloc[0].values,
                feature_names=selected_features
            )
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(latest_explanation, show=False)
            
            fig = plt.gcf()
            for ax_obj in fig.axes:
                for text_obj in ax_obj.texts:
                    txt = text_obj.get_text()
                    if '\u2212' in txt:
                        text_obj.set_text(txt.replace('\u2212', '-'))
                new_lbls = [label.get_text().replace('\u2212', '-') for label in ax_obj.get_yticklabels()]
                ax_obj.set_yticklabels(new_lbls)
                
            plt.tight_layout()
            waterfall_latest_path = os.path.join(results_dir, 'shap_waterfall_latest.png')
            plt.savefig(waterfall_latest_path, dpi=150)
            plt.close()
            print(f"    - Saved latest waterfall plot to: {waterfall_latest_path}")
        except Exception as e:
            print(f"    - [WARNING] Waterfall plot for latest prediction failed: {e}")
        
    print("\nSHAP XAI Analysis Completed Successfully!")
 
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--valid', action='store_true', help='Validation mode')
    args = parser.parse_known_args()[0]
    run_explain(valid_mode=args.valid)
