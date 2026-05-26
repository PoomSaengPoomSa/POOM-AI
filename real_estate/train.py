import os
import pickle
from utils.preprocess import preprocess_data
from model import RealEstateEnsembleRegressor

def run_train():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Preprocess
    data = preprocess_data(test_months=24, vif_threshold=10.0)
    if data is None:
        print("[Error] Preprocessing failed.")
        return
        
    X_train_sc = data['X_train_sc']
    y_train = data['y_train']
    selected_features = data['features']
    scaler = data['scaler']
    
    # -----------------------------------------
    # TimeSeriesSplit Cross-Validation (5 Splits)
    # -----------------------------------------
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import r2_score, mean_absolute_error
    import numpy as np
    from sklearn.linear_model import LinearRegression
    
    print("\n" + "=" * 55)
    print("  [TimeSeriesSplit Cross-Validation (5 Splits) on Train Set]")
    print("=" * 55)
    
    tscv = TimeSeriesSplit(n_splits=5)
    cv_r2_scores = []
    cv_mae_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_sc)):
        X_tr, X_val = X_train_sc[train_idx], X_train_sc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        fold_model = LinearRegression().fit(X_tr, y_tr)
        fold_pred = fold_model.predict(X_val)
        
        fold_r2 = r2_score(y_val, fold_pred)
        fold_mae = mean_absolute_error(y_val, fold_pred)
        
        cv_r2_scores.append(fold_r2)
        cv_mae_scores.append(fold_mae)
        
        print(f"    * Fold {fold+1} | Train: {len(X_tr)} months, Val: {len(X_val)} months | Val R2: {fold_r2:.4f} | Val MAE: {fold_mae:.4f}%")
        
    print(f"  --> Mean CV R2  : {np.mean(cv_r2_scores):.4f}")
    print(f"  --> Mean CV MAE : {np.mean(cv_mae_scores):.4f}%")
    print("=" * 55 + "\n")
    
    # Train final ensemble model
    ensemble = RealEstateEnsembleRegressor(random_state=42)
    ensemble.fit(X_train_sc, y_train)
    
    # Setup directories and save
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'ensemble_model.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    features_path = os.path.join(models_dir, 'selected_features.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(ensemble, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    with open(features_path, 'wb') as f:
        pickle.dump(selected_features, f)
        
    # Also save features as readable text
    txt_features_path = os.path.join(models_dir, 'selected_features.txt')
    with open(txt_features_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(selected_features))
        
    print("=" * 55)
    print("Training Pipeline Completed Successfully!")
    print("=" * 55)
    print(f"  Saved Model   : {model_path}")
    print(f"  Saved Scaler  : {scaler_path}")
    print(f"  Saved Features: {features_path} and .txt")
    print(f"  Features size : {len(selected_features)}")

if __name__ == '__main__':
    run_train()
