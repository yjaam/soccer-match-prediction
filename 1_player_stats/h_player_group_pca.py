#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import warnings
warnings.filterwarnings('ignore')

def main():
    # FIXED: Go up one level from the script directory to reach project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)  # Go up from 1_player_stats to project root
    
    # FIXED: Paths now point to project-level directories
    input_path = os.path.join(project_dir, "data", "data_before_pca.csv")
    target_vars_path = os.path.join(project_dir, "data", "target_variables.csv")
    output_path = os.path.join(project_dir, "data", "player_group_pca.csv")
    misc_dir = os.path.join(project_dir, "misc")
    
    # Create necessary directories
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(misc_dir, exist_ok=True)

    # 1. Load Data
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return
    
    df = pd.read_csv(input_path)
    print(f"Loaded PCA data with {len(df)} rows and {df.shape[1]} columns.")
    
    # Load target variables
    if not os.path.exists(target_vars_path):
        print(f"Warning: Target variables file not found at {target_vars_path}")
        print("Will only use goals from data_before_pca.csv")
        df_targets = None
    else:
        df_targets = pd.read_csv(target_vars_path)
        print(f"Loaded target variables with {len(df_targets)} rows and {df_targets.shape[1]} columns")
        print(f"Target variables columns: {list(df_targets.columns)}")

    # 2. Position group mapping - matching actual column names
    group_configs = {
        "home_Goalkeeper": [
            "Home_Goalkeeper_Avg_Value",
            "HomeGoalkeeper_overall_Avg",
            "HomeGoalkeeper_goalkeeping_diving_Avg",
            "HomeGoalkeeper_goalkeeping_handling_Avg",
            "HomeGoalkeeper_goalkeeping_kicking_Avg",
            "HomeGoalkeeper_goalkeeping_positioning_Avg",
            "HomeGoalkeeper_goalkeeping_reflexes_Avg",
            "HomeGoalkeeper_goalkeeping_speed_Avg"
        ],
        "away_Goalkeeper": [
            "Away_Goalkeeper_Avg_Value",
            "AwayGoalkeeper_overall_Avg",
            "AwayGoalkeeper_goalkeeping_diving_Avg",
            "AwayGoalkeeper_goalkeeping_handling_Avg",
            "AwayGoalkeeper_goalkeeping_kicking_Avg",
            "AwayGoalkeeper_goalkeeping_positioning_Avg",
            "AwayGoalkeeper_goalkeeping_reflexes_Avg",
            "AwayGoalkeeper_goalkeeping_speed_Avg"
        ],
        "home_Defenders": [
            "Home_Defender_Avg_Value",
            "HomeDefender_overall_Avg",
            "HomeDefender_pace_Avg",
            "HomeDefender_shooting_Avg",
            "HomeDefender_passing_Avg",
            "HomeDefender_dribbling_Avg",
            "HomeDefender_defending_Avg",
            "HomeDefender_physic_Avg"
        ],
        "away_Defenders": [
            "Away_Defender_Avg_Value",
            "AwayDefender_overall_Avg",
            "AwayDefender_pace_Avg",
            "AwayDefender_shooting_Avg",
            "AwayDefender_passing_Avg",
            "AwayDefender_dribbling_Avg",
            "AwayDefender_defending_Avg",
            "AwayDefender_physic_Avg"
        ],
        "home_Midfielders": [
            "Home_Midfielder_Avg_Value",
            "HomeMidfielder_overall_Avg",
            "HomeMidfielder_pace_Avg",
            "HomeMidfielder_shooting_Avg",
            "HomeMidfielder_passing_Avg",
            "HomeMidfielder_dribbling_Avg",
            "HomeMidfielder_defending_Avg",
            "HomeMidfielder_physic_Avg"
        ],
        "away_Midfielders": [
            "Away_Midfielder_Avg_Value",
            "AwayMidfielder_overall_Avg",
            "AwayMidfielder_pace_Avg",
            "AwayMidfielder_shooting_Avg",
            "AwayMidfielder_passing_Avg",
            "AwayMidfielder_dribbling_Avg",
            "AwayMidfielder_defending_Avg",
            "AwayMidfielder_physic_Avg"
        ],
        "home_Attackers": [
            "Home_Attacker_Avg_Value",
            "HomeAttacker_overall_Avg",
            "HomeAttacker_pace_Avg",
            "HomeAttacker_shooting_Avg",
            "HomeAttacker_passing_Avg",
            "HomeAttacker_dribbling_Avg",
            "HomeAttacker_defending_Avg",
            "HomeAttacker_physic_Avg"
        ],
        "away_Attackers": [
            "Away_Attacker_Avg_Value",
            "AwayAttacker_overall_Avg",
            "AwayAttacker_pace_Avg",
            "AwayAttacker_shooting_Avg",
            "AwayAttacker_passing_Avg",
            "AwayAttacker_dribbling_Avg",
            "AwayAttacker_defending_Avg",
            "AwayAttacker_physic_Avg"
        ]
    }

    # Filter group configs to only include available features
    filtered_group_configs = {}
    all_group_features = []
    
    print("\n" + "="*60)
    print("CHECKING AVAILABLE FEATURES PER GROUP")
    print("="*60)
    
    for group_key, group_cols in group_configs.items():
        available_group_cols = [col for col in group_cols if col in df.columns]
        if available_group_cols:
            filtered_group_configs[group_key] = available_group_cols
            all_group_features.extend(available_group_cols)
            
            if len(available_group_cols) < len(group_cols):
                missing = set(group_cols) - set(available_group_cols)
                print(f"{group_key}: {len(available_group_cols)}/{len(group_cols)} features")
                print(f"  Missing: {missing}")
            else:
                print(f"{group_key}: {len(available_group_cols)}/{len(group_cols)} features ✓")
        else:
            print(f"{group_key}: 0/{len(group_cols)} features - SKIPPED (no features available)")
    
    if not filtered_group_configs:
        print("\nERROR: No matching features found! Please check column names.")
        return
    
    total_features = sum(len(cols) for cols in filtered_group_configs.values())
    print(f"\nTotal groups with features: {len(filtered_group_configs)}")
    print(f"Total features to process: {total_features}")

    # 3. Handle missing values and standardize
    print("\n" + "="*60)
    print("STANDARDIZING FEATURES")
    print("="*60)
    
    # Check for missing values
    all_features = list(set(all_group_features))
    missing_counts = df[all_features].isnull().sum()
    total_missing = missing_counts.sum()
    
    if total_missing > 0:
        print(f"Total missing values across all features: {total_missing}")
        print("Features with missing values:")
        for feat, count in missing_counts[missing_counts > 0].items():
            pct = (count / len(df)) * 100
            print(f"  {feat}: {count} ({pct:.1f}%)")
        
        # Fill missing values with median for each column
        print("\nFilling missing values with column medians...")
        for col in all_features:
            if df[col].isnull().any():
                median_val = df[col].median()
                if pd.isna(median_val):
                    # If all values are NaN, fill with 0
                    df[col] = df[col].fillna(0)
                    print(f"  {col}: All values NaN, filled with 0")
                else:
                    df[col] = df[col].fillna(median_val)
    else:
        print("No missing values found ✓")
    
    # Create scaled DataFrame
    X_scaled = df.copy()
    
    # Standardize all group features together using a single scaler
    scaler = StandardScaler()
    X_scaled[all_features] = scaler.fit_transform(X_scaled[all_features])
    
    # Save scaler for future use
    joblib.dump(scaler, os.path.join(misc_dir, "player_group_scaler.joblib"))
    print("Scaler saved to misc/player_group_scaler.joblib")

    # 4. Perform PCA on each group and take ONLY the first PC
    pca_dict = {}
    pca_results_summary = []
    
    print("\n" + "="*60)
    print("PERFORMING PCA BY POSITION GROUP")
    print("="*60)
    
    # Dictionary to collect PCA components
    pca_components = {}
    
    for group_key, group_cols in filtered_group_configs.items():
        print(f"\nProcessing {group_key}...")
        print(f"  Features ({len(group_cols)}): {group_cols}")
        
        # Extract group data (already scaled)
        X_group = X_scaled[group_cols].copy()
        
        # Double-check for any remaining NaN or infinite values
        if X_group.isnull().any().any():
            print(f"  Warning: Found {X_group.isnull().sum().sum()} NaN values, filling with 0")
            X_group = X_group.fillna(0)
        
        X_group = X_group.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Perform PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_group)
        
        # Take only the first principal component
        first_pc = X_pca[:, 0]
        pca_components[group_key] = first_pc
        
        # Store in dictionary
        pca_dict[group_key] = pca
        
        # Save results summary
        explained_var = pca.explained_variance_ratio_[0]
        cumulative_var_2 = np.sum(pca.explained_variance_ratio_[:2])
        
        pca_results_summary.append({
            'Group': group_key,
            'Features_Count': len(group_cols),
            'Features': ', '.join(group_cols),
            'PC1_Explained_Variance': explained_var,
            'PC1_Explained_Variance_Pct': explained_var * 100,
            'PC2_Cumulative_Variance_Pct': cumulative_var_2 * 100
        })
        
        print(f"  PC1 explained variance: {explained_var:.3f} ({explained_var*100:.1f}%)")
        print(f"  First 2 PCs cumulative: {cumulative_var_2*100:.1f}%")
        
        # Print top loadings for PC1
        loadings = pd.Series(pca.components_[0], index=group_cols)
        top_features = loadings.abs().sort_values(ascending=False).head(3)
        print(f"  Top 3 features for PC1:")
        for feat in top_features.index:
            loading = loadings[feat]
            direction = "+" if loading > 0 else "-"
            print(f"    {feat}: {loading:.3f} ({direction})")

    # 5. Cross-reference and fill goals from both sources
    print("\n" + "="*60)
    print("CROSS-REFERENCING GOALS FROM BOTH SOURCES")
    print("="*60)
    
    # Extract existing goals from data_before_pca.csv
    existing_hg = None
    existing_ag = None
    
    if 'FTHG' in df.columns:
        existing_hg = df['FTHG'].copy()
        print(f"Found FTHG column in PCA data: {existing_hg.notna().sum()} non-null values")
    elif 'HG' in df.columns:
        existing_hg = df['HG'].copy()
        print(f"Found HG column in PCA data: {existing_hg.notna().sum()} non-null values")
    else:
        print("No home goals column found in PCA data")
    
    if 'FTAG' in df.columns:
        existing_ag = df['FTAG'].copy()
        print(f"Found FTAG column in PCA data: {existing_ag.notna().sum()} non-null values")
    elif 'AG' in df.columns:
        existing_ag = df['AG'].copy()
        print(f"Found AG column in PCA data: {existing_ag.notna().sum()} non-null values")
    else:
        print("No away goals column found in PCA data")
    
    # Initialize final goal columns with existing data
    final_hg = existing_hg.copy() if existing_hg is not None else pd.Series([np.nan] * len(df))
    final_ag = existing_ag.copy() if existing_ag is not None else pd.Series([np.nan] * len(df))
    
    # Track filling statistics
    hg_filled_from_target = 0
    ag_filled_from_target = 0
    hg_still_missing = final_hg.isna().sum()
    ag_still_missing = final_ag.isna().sum()
    
    # Cross-reference with target_variables.csv if available
    if df_targets is not None and 'game_id' in df.columns:
        print(f"\nCross-referencing with target_variables.csv...")
        print(f"Target variables rows: {len(df_targets)}")
        
        # Create lookup dictionaries from target variables
        target_hg_lookup = dict(zip(df_targets['game_id'], df_targets['HG']))
        target_ag_lookup = dict(zip(df_targets['game_id'], df_targets['AG']))
        
        # Fill missing goals from target variables
        for idx, game_id in enumerate(df['game_id']):
            if pd.notna(game_id) and game_id in target_hg_lookup:
                # Fill HG if missing
                if pd.isna(final_hg.iloc[idx]) and pd.notna(target_hg_lookup[game_id]):
                    final_hg.iloc[idx] = target_hg_lookup[game_id]
                    hg_filled_from_target += 1
                
                # Fill AG if missing
                if pd.isna(final_ag.iloc[idx]) and pd.notna(target_ag_lookup[game_id]):
                    final_ag.iloc[idx] = target_ag_lookup[game_id]
                    ag_filled_from_target += 1
        
        # Also check if target variables has goals where PCA data has different values
        # (this could indicate data quality issues)
        conflicts_hg = 0
        conflicts_ag = 0
        for idx, game_id in enumerate(df['game_id']):
            if pd.notna(game_id) and game_id in target_hg_lookup:
                # Check HG conflicts
                if (pd.notna(final_hg.iloc[idx]) and 
                    pd.notna(target_hg_lookup[game_id]) and 
                    final_hg.iloc[idx] != target_hg_lookup[game_id]):
                    conflicts_hg += 1
                
                # Check AG conflicts
                if (pd.notna(final_ag.iloc[idx]) and 
                    pd.notna(target_ag_lookup[game_id]) and 
                    final_ag.iloc[idx] != target_ag_lookup[game_id]):
                    conflicts_ag += 1
        
        print(f"\nGoals filling summary:")
        print(f"  HG - Originally present: {existing_hg.notna().sum() if existing_hg is not None else 0}")
        print(f"  HG - Filled from target_variables: {hg_filled_from_target}")
        print(f"  HG - Still missing: {final_hg.isna().sum()}")
        if conflicts_hg > 0:
            print(f"  ⚠ HG conflicts between sources: {conflicts_hg} (keeping PCA data values)")
        
        print(f"\n  AG - Originally present: {existing_ag.notna().sum() if existing_ag is not None else 0}")
        print(f"  AG - Filled from target_variables: {ag_filled_from_target}")
        print(f"  AG - Still missing: {final_ag.isna().sum()}")
        if conflicts_ag > 0:
            print(f"  ⚠ AG conflicts between sources: {conflicts_ag} (keeping PCA data values)")
        
    else:
        if df_targets is None:
            print("\nNo target_variables.csv available, using only PCA data goals")
        elif 'game_id' not in df.columns:
            print("\nNo game_id column in PCA data, cannot cross-reference with target_variables.csv")
    
    # 6. Build the final dataframe with ONLY specified columns
    print("\n" + "="*60)
    print("BUILDING FINAL DATASET")
    print("="*60)
    
    # Start with game_id
    final_columns = {}
    
    # Add game_id
    if 'game_id' in df.columns:
        final_columns['game_id'] = df['game_id'].values
        print("✓ Added game_id column")
    else:
        print("⚠ Warning: game_id column not found")
    
    # Add PCA components
    for group_key, pc_values in pca_components.items():
        final_columns[group_key] = pc_values
        print(f"✓ Added {group_key} PCA component")
    
    # Add cross-referenced goals
    final_columns['HG'] = final_hg.values
    final_columns['AG'] = final_ag.values
    print("✓ Added HG and AG columns (cross-referenced from both sources)")
    
    # Create final DataFrame
    df_final = pd.DataFrame(final_columns)
    
    # Remove rows where goals are still missing (if any)
    initial_final_rows = len(df_final)
    df_final = df_final.dropna(subset=['HG', 'AG'])
    rows_removed_missing_goals = initial_final_rows - len(df_final)
    
    if rows_removed_missing_goals > 0:
        print(f"\n⚠ Removed {rows_removed_missing_goals} rows with missing goals after cross-referencing")
    
    # Save the transformed dataset
    df_final.to_csv(output_path, index=False)
    print(f"\n" + "="*60)
    print(f"PCA TRANSFORMATION COMPLETE!")
    print(f"Saved to: {output_path}")
    print(f"Final shape: {df_final.shape}")
    print(f"Columns: {list(df_final.columns)}")
    
    # Save PCA models and results to misc folder
    joblib.dump(pca_dict, os.path.join(misc_dir, "pca_models_by_group.joblib"))
    print(f"\nPCA models saved to misc/pca_models_by_group.joblib")
    
    # Save summary statistics
    summary_df = pd.DataFrame(pca_results_summary)
    summary_df.to_csv(os.path.join(misc_dir, "pca_summary.csv"), index=False)
    print(f"PCA summary saved to misc/pca_summary.csv")
    
    # Save goals cross-reference report
    goals_report = pd.DataFrame({
        'Metric': [
            'Total rows before goal filtering',
            'Rows removed (missing goals)',
            'Final rows',
            'HG from PCA data',
            'HG filled from target_variables',
            'HG conflicts between sources',
            'AG from PCA data',
            'AG filled from target_variables',
            'AG conflicts between sources'
        ],
        'Value': [
            initial_final_rows,
            rows_removed_missing_goals,
            len(df_final),
            existing_hg.notna().sum() if existing_hg is not None else 0,
            hg_filled_from_target,
            conflicts_hg if df_targets is not None else 0,
            existing_ag.notna().sum() if existing_ag is not None else 0,
            ag_filled_from_target,
            conflicts_ag if df_targets is not None else 0
        ]
    })
    goals_report.to_csv(os.path.join(misc_dir, "goals_cross_reference_report.csv"), index=False)
    print(f"Goals cross-reference report saved to misc/goals_cross_reference_report.csv")
    
    # Create detailed Excel report
    with pd.ExcelWriter(os.path.join(misc_dir, "player_group_pca_results.xlsx")) as writer:
        # Summary sheet
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Goals report
        goals_report.to_excel(writer, sheet_name='Goals_Report', index=False)
        
        # Detailed loadings for each group
        for group_key, group_cols in filtered_group_configs.items():
            if group_key in pca_dict:
                pca = pca_dict[group_key]
                
                # Create loadings DataFrame
                loadings_df = pd.DataFrame(
                    pca.components_.T,
                    columns=[f'PC{i+1}' for i in range(len(group_cols))],
                    index=group_cols
                )
                
                # Add explained variance
                var_df = pd.DataFrame({
                    'Explained_Variance': pca.explained_variance_ratio_,
                    'Cumulative_Variance': np.cumsum(pca.explained_variance_ratio_)
                }, index=[f'PC{i+1}' for i in range(len(group_cols))])
                
                # Save to Excel (truncate sheet name if needed)
                sheet_name = group_key[:31]
                loadings_df.to_excel(writer, sheet_name=f'{sheet_name}_loadings')
                var_df.to_excel(writer, sheet_name=f'{sheet_name}_variance')
    
    print(f"Detailed results saved to misc/player_group_pca_results.xlsx")
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Total groups processed: {len(filtered_group_configs)}")
    print(f"Total input features used in PCA: {total_features}")
    print(f"Output features: {df_final.shape[1]}")
    print(f"Data points: {df_final.shape[0]}")
    
    avg_var = np.mean([s['PC1_Explained_Variance'] for s in pca_results_summary])
    print(f"\nAverage PC1 explained variance: {avg_var:.3f} ({avg_var*100:.1f}%)")
    
    print(f"\nPC1 Variance by group:")
    for summary in pca_results_summary:
        print(f"  {summary['Group']}: {summary['PC1_Explained_Variance']*100:.1f}%")
    
    print(f"\nGoal completion rate: {(len(df_final) / initial_final_rows * 100):.1f}%")
    print(f"\nOutput columns:")
    for col in df_final.columns:
        print(f"  - {col}")
    
    print(f"\nOutput saved to: {output_path}")

if __name__ == "__main__":
    main()