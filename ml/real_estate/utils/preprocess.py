import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def calculate_vif_custom(df, features):
    vif_dict = {}
    for feature in features:
        other_features = [f for f in features if f != feature]
        if not other_features:
            vif_dict[feature] = 1.0
            continue
        
        X = df[other_features].values
        y = df[feature].values
        
        reg = LinearRegression().fit(X, y)
        r2 = reg.score(X, y)
        
        if r2 >= 1.0:
            vif = float('inf')
        else:
            vif = 1.0 / (1.0 - r2)
        vif_dict[feature] = vif
        
    return pd.Series(vif_dict)

def filter_features_by_vif(df, features, threshold=5.0, max_features=6):
    current_features = list(features)
    print("  [Aggressive VIF Feature Pruning for Production]")
    
    # Core target-related indicators that we protect from being dropped
    protected_features = ["buyer_dominance_change", "kr_mortgage_rate_change", "house_price_idx_change"]
    
    while True:
        if len(current_features) <= 4:
            break
            
        vif_series = calculate_vif_custom(df, current_features)
        
        # If we have too many features, we drop candidates even if VIF is slightly below threshold
        candidates = vif_series.drop(labels=[f for f in protected_features if f in vif_series.index])
        if candidates.empty:
            break
            
        max_vif = candidates.max()
        max_feature = candidates.idxmax()
        
        if max_vif > threshold or len(current_features) > max_features:
            print(f"    - Dropping '{max_feature}' with VIF = {max_vif:.4f}")
            current_features.remove(max_feature)
        else:
            break
            
    print(f"  Final selected features ({len(current_features)}): {current_features}")
    
    final_vifs = calculate_vif_custom(df, current_features)
    for feat, v in final_vifs.items():
        print(f"    * {feat:<25}: VIF = {v:.4f}")
        
    return current_features

def preprocess_data(test_months=24, vif_threshold=5.0):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_dir, 'data', 'raw_data.csv')
    
    if not os.path.exists(raw_path):
        print(f"[Error] Raw data file not found at {raw_path}. Run get_data.py first.")
        return None
        
    df = pd.read_csv(raw_path)
    df = df.sort_values("date_ym").reset_index(drop=True)
    
    # 1. Target creation (shift -1 to pull next month's price index change rate)
    df["next_house_price_idx"] = df["house_price_idx"].shift(-1)
    TARGET = "next_change_rate"
    df[TARGET] = (df["next_house_price_idx"] - df["house_price_idx"]) / df["house_price_idx"] * 100
    
    # Drop rows without target (the last row)
    df = df.dropna(subset=[TARGET]).copy()
    
    # 2. Impute missing values in raw features first
    raw_features = [
        "house_price_idx", "kr_cpi", "kr_unemployment", 
        "kr_base_rate", "kr_mortgage_rate", "kospi200", 
        "apt_trade_count", "kr_m2", "buyer_dominance"
    ]
    df[raw_features] = df[raw_features].ffill().bfill()
    
    # -----------------------------------------
    # FEATURE ENGINEERING: Stationary Transformation, Lags, Moving Averages, Seasonality
    # -----------------------------------------
    # Transform raw trending variables into stationary changes/returns (MoM)
    df["house_price_idx_change"] = df["house_price_idx"].pct_change() * 100
    df["kr_cpi_change"] = df["kr_cpi"].pct_change() * 100
    df["kr_unemployment_change"] = df["kr_unemployment"].diff()
    df["kr_base_rate_change"] = df["kr_base_rate"].diff()
    df["kr_mortgage_rate_change"] = df["kr_mortgage_rate"].diff()
    df["kospi200_change"] = df["kospi200"].pct_change() * 100
    df["apt_trade_count_change"] = df["apt_trade_count"].pct_change() * 100
    df["kr_m2_change"] = df["kr_m2"].pct_change() * 100
    df["buyer_dominance_change"] = df["buyer_dominance"].diff()
    
    # 1. Seasonality Features (Month Sin/Cos)
    month_series = pd.to_datetime(df['date_ym'], format='%Y%m').dt.month
    df['month_sin'] = np.sin(2 * np.pi * month_series / 12)
    df['month_cos'] = np.cos(2 * np.pi * month_series / 12)
    seasonality_features = ['month_sin', 'month_cos']
    
    # 2. Multi-month Lags (Lag 1, 2, 3) to capture delayed impacts
    stationary_cols = [
        "house_price_idx_change", "kr_cpi_change", "kr_unemployment_change",
        "kr_mortgage_rate_change", "kospi200_change", "buyer_dominance_change",
        "apt_trade_count_change"
    ]
    
    lagged_features = []
    for col in stationary_cols:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag2"] = df[col].shift(2)
        df[f"{col}_lag3"] = df[col].shift(3)
        lagged_features.extend([f"{col}_lag1", f"{col}_lag2", f"{col}_lag3"])
        
    # 3. Moving Averages (ma3, ma6) to capture momentum and smooth noise
    rolling_features = []
    for col in ["house_price_idx_change", "buyer_dominance_change", "apt_trade_count_change", "kr_mortgage_rate_change"]:
        df[f"{col}_ma3"] = df[col].rolling(window=3).mean()
        df[f"{col}_ma6"] = df[col].rolling(window=6).mean()
        rolling_features.extend([f"{col}_ma3", f"{col}_ma6"])
        
    # Candidate features (All Stationary!)
    candidate_features = [
        "house_price_idx_change", "kr_cpi_change", "kr_unemployment_change", 
        "kr_base_rate_change", "kr_mortgage_rate_change", "kospi200_change", 
        "apt_trade_count_change", "kr_m2_change", "buyer_dominance_change"
    ] + lagged_features + rolling_features + seasonality_features
    
    # Train/Test split FIRST
    train_df = df.iloc[:-test_months].copy()
    test_df = df.iloc[-test_months:].copy()
    
    # Fill NAs generated by diffs/lags/rolling windows separately on train and test subsets to prevent data leakage
    train_df[candidate_features] = train_df[candidate_features].ffill().bfill()
    test_df[candidate_features] = test_df[candidate_features].ffill().bfill()
    
    print("=" * 55)
    print("Data Preprocessing & Train/Test Splitting")
    print("=" * 55)
    print(f"  Total samples     : {len(df)}")
    print(f"  Train period      : {train_df['date_ym'].min()} ~ {train_df['date_ym'].max()} ({len(train_df)} months)")
    print(f"  Test period       : {test_df['date_ym'].min()} ~ {test_df['date_ym'].max()} ({len(test_df)} months)")
    
    # Perform Aggressive VIF selection on Train subset (limiting to max 6 features)
    selected_features = filter_features_by_vif(train_df, candidate_features, threshold=vif_threshold, max_features=6)
    
    # Subsets
    X_train = train_df[selected_features]
    y_train = train_df[TARGET]
    X_test = test_df[selected_features]
    y_test = test_df[TARGET]
    
    # Standardize
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    preprocessed_data = {
        'df': df,
        'train_df': train_df,
        'test_df': test_df,
        'X_train_sc': X_train_sc,
        'X_test_sc': X_test_sc,
        'y_train': y_train,
        'y_test': y_test,
        'features': selected_features,
        'scaler': scaler
    }
    
    return preprocessed_data

if __name__ == '__main__':
    preprocess_data()
