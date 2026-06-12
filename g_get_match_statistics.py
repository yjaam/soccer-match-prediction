#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Process and clean match statistics data independently
# path to match statistics data: matchhistory_data/match_statistics.csv
# output path: data/match_statistics_before_pca.csv

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

# Import the FINAL team name mapping
from team_name_mapping_FINAL import soccer_teams, resolve_team_name


def clean_match_statistics(df):
    """
    Clean and standardize the match statistics dataframe.
    """
    print("\n--- Cleaning Match Statistics ---")
    
    # Make a copy to avoid warnings
    df = df.copy()
    
    # Drop completely empty rows
    before = len(df)
    df = df.dropna(how='all')
    print(f"Dropped {before - len(df)} completely empty rows")
    
    # Drop completely empty columns
    before_cols = len(df.columns)
    df = df.dropna(axis=1, how='all')
    print(f"Dropped {before_cols - len(df.columns)} completely empty columns")
    
    # Standardize date column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Clean numeric columns (remove any string artifacts)
    numeric_cols = df.select_dtypes(include=['object']).columns
    for col in numeric_cols:
        # Try to convert to numeric where possible
        try:
            # Check if column looks numeric
            sample = df[col].dropna().head(100)
            if len(sample) > 0 and sample.astype(str).str.match(r'^-?\d+\.?\d*$').all():
                df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            pass
    
    return df


def map_team_names(df):
    """
    Map team names in match statistics to three-letter codes.
    """
    print("\n--- Mapping Team Names to Three-Letter Codes ---")
    
    # Check which columns contain team names
    team_columns = []
    for col in df.columns:
        col_lower = col.lower()
        if any(term in col_lower for term in ['team', 'home', 'away', 'ht', 'at']):
            if df[col].dtype == 'object':  # Only string columns
                team_columns.append(col)
    
    print(f"Found potential team columns: {team_columns}")
    
    mapping_stats = {}
    
    for col in team_columns:
        # Get unique team names in this column
        unique_teams = df[col].dropna().unique()
        print(f"\nColumn '{col}': {len(unique_teams)} unique values")
        
        # Try to map each team name
        mapped_count = 0
        unmapped = []
        
        for team_name in unique_teams:
            if pd.isna(team_name) or str(team_name).strip() == '':
                continue
            
            team_str = str(team_name).strip()
            code = resolve_team_name(team_str)
            
            if code:
                mapped_count += 1
            else:
                unmapped.append(team_str)
        
        mapping_stats[col] = {
            'total': len(unique_teams),
            'mapped': mapped_count,
            'unmapped': unmapped
        }
        
        print(f"  Mapped: {mapped_count}/{len(unique_teams)}")
        if unmapped and len(unmapped) <= 10:
            print(f"  Unmapped: {unmapped}")
        elif unmapped:
            print(f"  Unmapped: {len(unmapped)} teams (showing first 10):")
            for team in unmapped[:10]:
                print(f"    - '{team}'")
        
        # Apply mapping to create new coded column
        if mapped_count > 0:
            new_col = col + '_code'
            df[new_col] = df[col].apply(
                lambda x: resolve_team_name(str(x).strip()) if pd.notna(x) and str(x).strip() else None
            )
    
    return df, mapping_stats


def add_derived_features(df):
    """
    Add useful derived features from match statistics.
    """
    print("\n--- Adding Derived Features ---")
    features_added = []
    
    # Add season/year feature if date exists
    if 'date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['date']):
        df['season_year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        features_added.extend(['season_year', 'month'])
        print("Added: season_year, month")
    
    # Add total goals if home/away goals exist
    if 'home_score' in df.columns and 'away_score' in df.columns:
        df['total_goals'] = pd.to_numeric(df['home_score'], errors='coerce') + \
                           pd.to_numeric(df['away_score'], errors='coerce')
        features_added.append('total_goals')
        print("Added: total_goals")
    
    # Add goal difference
    if 'home_score' in df.columns and 'away_score' in df.columns:
        df['goal_difference'] = pd.to_numeric(df['home_score'], errors='coerce') - \
                               pd.to_numeric(df['away_score'], errors='coerce')
        features_added.append('goal_difference')
        print("Added: goal_difference")
    
    # Add result (Home Win, Draw, Away Win)
    if 'home_score' in df.columns and 'away_score' in df.columns:
        home_score = pd.to_numeric(df['home_score'], errors='coerce')
        away_score = pd.to_numeric(df['away_score'], errors='coerce')
        
        conditions = [
            home_score > away_score,
            home_score == away_score,
            home_score < away_score
        ]
        choices = ['H', 'D', 'A']
        df['result'] = np.select(conditions, choices, default=None)
        features_added.append('result')
        print("Added: result (H=Home Win, D=Draw, A=Away Win)")
    
    return df, features_added


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    match_stats_path = os.path.join(script_dir, "matchhistory_data", "match_statistics.csv")
    output_path = os.path.join(script_dir, "data", "match_statistics_before_pca.csv")

    # ============================================================================
    # Load data
    # ============================================================================
    print("="*60)
    print("MATCH STATISTICS PROCESSING")
    print("="*60)
    
    print("\nLoading match statistics...")
    match_stats_df = pd.read_csv(match_stats_path, low_memory=False)
    
    initial_shape = match_stats_df.shape
    print(f"Initial shape: {initial_shape}")
    print(f"Columns: {len(match_stats_df.columns)}")
    
    # ============================================================================
    # Data overview
    # ============================================================================
    print("\n--- Data Overview ---")
    print(f"Missing values per column (top 10):")
    missing = match_stats_df.isnull().sum().sort_values(ascending=False)
    for col, count in missing.head(10).items():
        pct = (count / len(match_stats_df)) * 100
        print(f"  {col}: {count} ({pct:.1f}%)")
    
    # Show column types
    print(f"\nColumn types:")
    dtype_counts = match_stats_df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")
    
    # ============================================================================
    # Clean data
    # ============================================================================
    match_stats_df = clean_match_statistics(match_stats_df)
    
    # ============================================================================
    # Map team names
    # ============================================================================
    match_stats_df, mapping_stats = map_team_names(match_stats_df)
    
    # ============================================================================
    # Add derived features
    # ============================================================================
    match_stats_df, added_features = add_derived_features(match_stats_df)
    
    # ============================================================================
    # Final cleanup
    # ============================================================================
    print("\n--- Final Cleanup ---")
    
    # Remove duplicate columns
    duplicated_cols = match_stats_df.columns[match_stats_df.columns.duplicated()].tolist()
    if duplicated_cols:
        match_stats_df = match_stats_df.loc[:, ~match_stats_df.columns.duplicated()]
        print(f"Removed {len(duplicated_cols)} duplicate columns")
    
    # Remove constant columns (only one unique value)
    constant_cols = [col for col in match_stats_df.columns 
                    if match_stats_df[col].nunique() <= 1]
    if constant_cols:
        match_stats_df = match_stats_df.drop(columns=constant_cols)
        print(f"Removed {len(constant_cols)} constant columns: {constant_cols}")
    
    final_shape = match_stats_df.shape
    print(f"Final shape: {final_shape}")
    print(f"Rows kept: {final_shape[0]}/{initial_shape[0]} ({final_shape[0]/initial_shape[0]*100:.1f}%)")
    print(f"Columns kept: {final_shape[1]}/{initial_shape[1]}")
    
    # ============================================================================
    # Save
    # ============================================================================
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    match_stats_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Saved to: {output_path}")
    print(f"Rows: {final_shape[0]}")
    print(f"Columns: {final_shape[1]}")
    print(f"Features added: {', '.join(added_features) if added_features else 'None'}")
    
    # Summary of team mapping
    total_team_cols = len(mapping_stats)
    mapped_cols = sum(1 for v in mapping_stats.values() if v['mapped'] > 0)
    print(f"Team columns mapped: {mapped_cols}/{total_team_cols}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()