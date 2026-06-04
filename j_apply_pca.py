#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import joblib

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Paths
    input_path = os.path.join(script_dir, "data", "data_for_pca.csv")
    pca_model_path = os.path.join(project_root, "c_Dimension_Reduction", "pca_man.joblib")
    output_path = os.path.join(script_dir, "data", "data_pca_transformed.csv")

    # 1. Load Data
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return
    
    df = pd.read_csv(input_path)
    print(f"Loaded data with {len(df)} rows.")

    # 1a. Feature list exactly matching X_train.csv
    pca_features = [
        'attendance', 'home_club_position_before_game', 'away_club_position_before_game', 
        'home_form', 'away_form', 'home_market_value_Goalkeeper', 'away_market_value_Goalkeeper', 
        'home_rating_overall_Goalkeeper', 'away_rating_overall_Goalkeeper', 
        'home_rating_goalkeeping_diving_Goalkeeper', 'away_rating_goalkeeping_diving_Goalkeeper', 
        'home_rating_goalkeeping_handling_Goalkeeper', 'away_rating_goalkeeping_handling_Goalkeeper', 
        'home_rating_goalkeeping_kicking_Goalkeeper', 'away_rating_goalkeeping_kicking_Goalkeeper', 
        'home_rating_goalkeeping_positioning_Goalkeeper', 'away_rating_goalkeeping_positioning_Goalkeeper', 
        'home_rating_goalkeeping_reflexes_Goalkeeper', 'away_rating_goalkeeping_reflexes_Goalkeeper', 
        'home_rating_goalkeeping_speed_Goalkeeper', 'away_rating_goalkeeping_speed_Goalkeeper', 
        'home_market_value_Defenders', 'away_market_value_Defenders', 
        'home_rating_overall_Defenders', 'away_rating_overall_Defenders', 
        'home_rating_pace_Defenders', 'away_rating_pace_Defenders', 
        'home_rating_shooting_Defenders', 'away_rating_shooting_Defenders', 
        'home_rating_passing_Defenders', 'away_rating_passing_Defenders', 
        'home_rating_dribbling_Defenders', 'away_rating_dribbling_Defenders', 
        'home_rating_defending_Defenders', 'away_rating_defending_Defenders', 
        'home_rating_physic_Defenders', 'away_rating_physic_Defenders', 
        'home_market_value_Midfielders', 'away_market_value_Midfielders', 
        'home_rating_overall_Midfielders', 'away_rating_overall_Midfielders', 
        'home_rating_pace_Midfielders', 'away_rating_pace_Midfielders', 
        'home_rating_shooting_Midfielders', 'away_rating_shooting_Midfielders', 
        'home_rating_passing_Midfielders', 'away_rating_passing_Midfielders', 
        'home_rating_dribbling_Midfielders', 'away_rating_dribbling_Midfielders', 
        'home_rating_defending_Midfielders', 'away_rating_defending_Midfielders', 
        'home_rating_physic_Midfielders', 'away_rating_physic_Midfielders', 
        'home_market_value_Attackers', 'away_market_value_Attackers', 
        'home_rating_overall_Attackers', 'away_rating_overall_Attackers', 
        'home_rating_pace_Attackers', 'away_rating_pace_Attackers', 
        'home_rating_shooting_Attackers', 'away_rating_shooting_Attackers', 
        'home_rating_passing_Attackers', 'away_rating_passing_Attackers', 
        'home_rating_dribbling_Attackers', 'away_rating_dribbling_Attackers', 
        'home_rating_defending_Attackers', 'away_rating_defending_Attackers', 
        'home_rating_physic_Attackers', 'away_rating_physic_Attackers'
    ]

    # 1b. Standardize the data
    # Map dates to seasonal scalers
    df['Date'] = pd.to_datetime(df['Date'])
    
    def get_season(date):
        # Logic matches ids_train: seasons generally start in Aug
        if date.month >= 8:
            return float(date.year + 1)
        else:
            return float(date.year)

    df['season'] = df['Date'].apply(get_season)
    
    scaler_path = os.path.join(project_root, "c_Dimension_Reduction", "scalers_by_season.joblib")
    if not os.path.exists(scaler_path):
        print(f"Error: Scalers not found at {scaler_path}")
        return
    
    scalers = joblib.load(scaler_path)
    
    # Initialize scaled features with 0
    X_scaled = df.copy()
    for col in pca_features:
        if col not in X_scaled.columns:
            X_scaled[col] = 0.0
            
    # Apply seasonal scaling
    for season, scaler in scalers.items():
        mask = (df['season'] == season)
        if mask.any():
            X_season = X_scaled.loc[mask, pca_features].fillna(0.0)
            X_scaled.loc[mask, pca_features] = scaler.transform(X_season)
            print(f"Scaled {mask.sum()} rows for season {season}")
    
    # Handle future seasons without a scaler by using the most recent one (2023.0)
    unscaled_mask = ~df['season'].isin(scalers.keys())
    if unscaled_mask.any():
        latest_season = max(scalers.keys())
        latest_scaler = scalers[latest_season]
        X_future = X_scaled.loc[unscaled_mask, pca_features].fillna(0.0)
        X_scaled.loc[unscaled_mask, pca_features] = latest_scaler.transform(X_future)
        print(f"Scaled {unscaled_mask.sum()} rows for unknown season(s) using {latest_season} scaler")

    # 2. Position group mapping - using scaled columns
    group_configs = {
        "home_Goalkeeper": ["home_market_value_Goalkeeper", "home_rating_overall_Goalkeeper", "home_rating_goalkeeping_diving_Goalkeeper", "home_rating_goalkeeping_handling_Goalkeeper", "home_rating_goalkeeping_kicking_Goalkeeper", "home_rating_goalkeeping_positioning_Goalkeeper", "home_rating_goalkeeping_reflexes_Goalkeeper", "home_rating_goalkeeping_speed_Goalkeeper"],
        "away_Goalkeeper": ["away_market_value_Goalkeeper", "away_rating_overall_Goalkeeper", "away_rating_goalkeeping_diving_Goalkeeper", "away_rating_goalkeeping_handling_Goalkeeper", "away_rating_goalkeeping_kicking_Goalkeeper", "away_rating_goalkeeping_positioning_Goalkeeper", "away_rating_goalkeeping_reflexes_Goalkeeper", "away_rating_goalkeeping_speed_Goalkeeper"],
        "home_Defenders": ["home_market_value_Defenders", "home_rating_overall_Defenders", "home_rating_pace_Defenders", "home_rating_shooting_Defenders", "home_rating_passing_Defenders", "home_rating_dribbling_Defenders", "home_rating_defending_Defenders", "home_rating_physic_Defenders"],
        "away_Defenders": ["away_market_value_Defenders", "away_rating_overall_Defenders", "away_rating_pace_Defenders", "away_rating_shooting_Defenders", "away_rating_passing_Defenders", "away_rating_dribbling_Defenders", "away_rating_defending_Defenders", "away_rating_physic_Defenders"],
        "home_Midfielders": ["home_market_value_Midfielders", "home_rating_overall_Midfielders", "home_rating_pace_Midfielders", "home_rating_shooting_Midfielders", "home_rating_passing_Midfielders", "home_rating_dribbling_Midfielders", "home_rating_defending_Midfielders", "home_rating_physic_Midfielders"],
        "away_Midfielders": ["away_market_value_Midfielders", "away_rating_overall_Midfielders", "away_rating_pace_Midfielders", "away_rating_shooting_Midfielders", "away_rating_passing_Midfielders", "away_rating_dribbling_Midfielders", "away_rating_defending_Midfielders", "away_rating_physic_Midfielders"],
        "home_Attackers": ["home_market_value_Attackers", "home_rating_overall_Attackers", "home_rating_pace_Attackers", "home_rating_shooting_Attackers", "home_rating_passing_Attackers", "home_rating_dribbling_Attackers", "home_rating_defending_Attackers", "home_rating_physic_Attackers"],
        "away_Attackers": ["away_market_value_Attackers", "away_rating_overall_Attackers", "away_rating_pace_Attackers", "away_rating_shooting_Attackers", "away_rating_passing_Attackers", "away_rating_dribbling_Attackers", "away_rating_defending_Attackers", "away_rating_physic_Attackers"]
    }

    # 2a. Global features (already scaled above)
    global_features = ["attendance", "home_club_position_before_game", "away_club_position_before_game", "home_form", "away_form"]
    final_df_parts = [df[["Date", "Home Team", "Away Team"]], X_scaled[global_features]]

    # 3. Load PCA Model (Dictionary of models)
    if not os.path.exists(pca_model_path):
        print(f"Error: PCA model not found at {pca_model_path}")
        return
    
    pca_dict = joblib.load(pca_model_path)
    print(f"PCA model dictionary loaded with groups: {list(pca_dict.keys())}")

    # 4. Transform each group using Scaled data
    for group_key, group_cols in group_configs.items():
        if group_key not in pca_dict:
            print(f"Warning: Group {group_key} expected but not found in PCA dictionary.")
            continue
            
        X_group = X_scaled[group_cols].fillna(0.0)
        
        # Apply transformation
        transformed = pca_dict[group_key].transform(X_group)
        
        # Add to results
        comp_df = pd.DataFrame(transformed, columns=[f"{group_key}"])
        final_df_parts.append(comp_df)

    # Combine everything
    df_final = pd.concat(final_df_parts, axis=1)

    # 5. Save
    df_final.to_csv(output_path, index=False)
    print(f"PCA transformation complete. Saved to {output_path}")

if __name__ == "__main__":
    main()
