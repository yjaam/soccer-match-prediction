#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

src_dir = os.path.join(project_dir, "src")
sys.path.append(src_dir)

# Import the team name mapping functions
from team_name_mapping_FINAL import resolve_team_name, resolve_team_name_fuzzy

def create_game_id(date, home_team_code, away_team_code):
    """
    Create a game_id in the format: YYYYMMDD_HOME_AWAY
    Example: "20230812_MUN_BAR"
    """
    # Parse date if it's a string
    if isinstance(date, str):
        date = pd.to_datetime(date)
    
    # Format date as YYYYMMDD
    date_str = date.strftime('%Y%m%d')
    
    # Create game_id
    game_id = f"{date_str}_{home_team_code}_{away_team_code}"
    
    return game_id

def main():
    print("="*60)
    print("PROCESSING TRANSFERMARKT DATA")
    print("="*60)
    
    # Paths
    # Paths
    input_path = os.path.join(project_dir, "transfermarkt_data", "games.csv")
    output_path = os.path.join(project_dir, "data", "goals_and_formations.csv")
    misc_dir = os.path.join(project_dir, "misc")
    
    # Create necessary directories
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(misc_dir, exist_ok=True)
    
    # 1. Load the data
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return
    
    print(f"\nLoading data from {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows and {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    
    # Check for missing team names
    missing_home = df['home_club_name'].isnull().sum()
    missing_away = df['away_club_name'].isnull().sum()
    if missing_home > 0 or missing_away > 0:
        print(f"\n⚠ Missing team names found:")
        print(f"  Home team names missing: {missing_home}")
        print(f"  Away team names missing: {missing_away}")
        print(f"  These rows will be removed")
    
    # 2. Map team names to three-letter codes
    print("\n" + "="*60)
    print("MAPPING TEAM NAMES TO THREE-LETTER CODES")
    print("="*60)
    
    # Apply fuzzy matching to home and away team names
    df['home_team_code'] = df['home_club_name'].apply(resolve_team_name_fuzzy)
    df['away_team_code'] = df['away_club_name'].apply(resolve_team_name_fuzzy)
    
    # Check for unmapped teams (excluding NaN)
    unmapped_home_mask = df['home_team_code'].isna() & df['home_club_name'].notna()
    unmapped_away_mask = df['away_team_code'].isna() & df['away_club_name'].notna()
    
    unmapped_home = df.loc[unmapped_home_mask, 'home_club_name'].unique()
    unmapped_away = df.loc[unmapped_away_mask, 'away_club_name'].unique()
    
    # Combine and clean unmapped names
    all_unmapped = []
    for name in unmapped_home:
        if pd.notna(name) and str(name).strip():
            all_unmapped.append(str(name).strip())
    for name in unmapped_away:
        if pd.notna(name) and str(name).strip():
            if str(name).strip() not in all_unmapped:
                all_unmapped.append(str(name).strip())
    
    if len(all_unmapped) > 0:
        print(f"\n⚠ Found {len(all_unmapped)} unmapped team names:")
        for name in sorted(all_unmapped)[:50]:  # Show first 50
            home_count = (df['home_club_name'] == name).sum()
            away_count = (df['away_club_name'] == name).sum()
            print(f"  - {name} (Home: {home_count}, Away: {away_count})")
        
        if len(all_unmapped) > 50:
            print(f"  ... and {len(all_unmapped) - 50} more (see misc/unmapped_teams.csv for full list)")
    else:
        print("\n✓ All team names successfully mapped!")
    
    # Save mapping statistics
    if len(all_unmapped) > 0:
        mapping_stats_data = []
        for name in all_unmapped:
            home_count = (df['home_club_name'] == name).sum()
            away_count = (df['away_club_name'] == name).sum()
            mapping_stats_data.append({
                'team_name': name,
                'occurrences_home': home_count,
                'occurrences_away': away_count,
                'total_occurrences': home_count + away_count
            })
        
        mapping_stats = pd.DataFrame(mapping_stats_data)
        mapping_stats = mapping_stats.sort_values('total_occurrences', ascending=False)
        mapping_stats.to_csv(os.path.join(misc_dir, "unmapped_teams.csv"), index=False)
        print(f"\nUnmapped teams saved to misc/unmapped_teams.csv")
    
    # 3. Remove rows where team mapping failed or names were missing
    initial_rows = len(df)
    df_mapped = df.dropna(subset=['home_team_code', 'away_team_code']).copy()
    removed_rows = initial_rows - len(df_mapped)
    
    print(f"\n" + "="*60)
    print("FILTERING DATA")
    print("="*60)
    print(f"Rows before filtering: {initial_rows}")
    print(f"Rows removed (missing names or unmapped): {removed_rows} ({removed_rows/initial_rows*100:.1f}%)")
    print(f"Rows after filtering: {len(df_mapped)}")
    
    if len(df_mapped) == 0:
        print("\nERROR: No rows remaining after filtering! Check team name mappings.")
        return
    
    # 4. Create game_id
    print("\n" + "="*60)
    print("CREATING GAME IDs")
    print("="*60)
    
    # Parse dates
    df_mapped['date_parsed'] = pd.to_datetime(df_mapped['date'])
    
    # Create game_id
    df_mapped['game_id'] = df_mapped.apply(
        lambda row: create_game_id(row['date_parsed'], row['home_team_code'], row['away_team_code']),
        axis=1
    )
    
    # Check for duplicate game_ids
    duplicate_ids = df_mapped['game_id'].duplicated().sum()
    if duplicate_ids > 0:
        print(f"⚠ Found {duplicate_ids} duplicate game_ids")
        # Show some examples
        dup_examples = df_mapped[df_mapped['game_id'].duplicated(keep=False)]['game_id'].value_counts().head(5)
        print("Examples of duplicate IDs:")
        for game_id, count in dup_examples.items():
            print(f"  {game_id}: {count} occurrences")
    else:
        print("✓ All game_ids are unique")
    
    print(f"\nSample game_ids:")
    sample_rows = min(5, len(df_mapped))
    for i in range(sample_rows):
        row = df_mapped.iloc[i]
        print(f"  {row['date_parsed'].strftime('%Y-%m-%d')}: {row['home_club_name']} ({row['home_team_code']}) vs {row['away_club_name']} ({row['away_team_code']}) → {row['game_id']}")
    
    # 5. Create the target variables dataframe
    print("\n" + "="*60)
    print("CREATING TARGET VARIABLES DATASET")
    print("="*60)
    
    target_df = pd.DataFrame()
    
    # Add game_id
    target_df['game_id'] = df_mapped['game_id']
    
    # Add home and away goals (renamed to HG and AG)
    target_df['HG'] = df_mapped['home_club_goals']
    target_df['AG'] = df_mapped['away_club_goals']
    
    # Add formations
    target_df['home_formation'] = df_mapped['home_club_formation']
    target_df['away_formation'] = df_mapped['away_club_formation']
    
    # Check for missing values
    print(f"\nMissing values in target variables:")
    has_missing = False
    for col in target_df.columns:
        missing = target_df[col].isnull().sum()
        if missing > 0:
            pct = (missing / len(target_df)) * 100
            print(f"  {col}: {missing} ({pct:.1f}%)")
            has_missing = True
        else:
            print(f"  {col}: 0 ✓")
    
    # Handle missing formations
    if target_df['home_formation'].isnull().any():
        target_df['home_formation'] = target_df['home_formation'].fillna('unknown')
        print(f"  → Filled missing home formations with 'unknown'")
    
    if target_df['away_formation'].isnull().any():
        target_df['away_formation'] = target_df['away_formation'].fillna('unknown')
        print(f"  → Filled missing away formations with 'unknown'")
    
    # Handle missing goals (shouldn't happen, but just in case)
    if target_df['HG'].isnull().any() or target_df['AG'].isnull().any():
        print(f"  ⚠ WARNING: Missing goal values found! Removing affected rows.")
        target_df = target_df.dropna(subset=['HG', 'AG'])
    
    # 6. Save the target variables
    target_df.to_csv(output_path, index=False)
    
    print(f"\n" + "="*60)
    print(f"TARGET VARIABLES SAVED!")
    print(f"Saved to: {output_path}")
    print(f"Shape: {target_df.shape}")
    print(f"Columns: {list(target_df.columns)}")
    
    # 7. Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print(f"\nGoals distribution:")
    print(f"  Home goals - Mean: {target_df['HG'].mean():.2f}, Median: {target_df['HG'].median():.1f}")
    print(f"  Away goals - Mean: {target_df['AG'].mean():.2f}, Median: {target_df['AG'].median():.1f}")
    print(f"  Home goals range: {target_df['HG'].min():.0f} - {target_df['HG'].max():.0f}")
    print(f"  Away goals range: {target_df['AG'].min():.0f} - {target_df['AG'].max():.0f}")
    
    print(f"\nTop 5 home formations:")
    home_formations = target_df['home_formation'].value_counts().head()
    for form, count in home_formations.items():
        print(f"  {form}: {count} ({count/len(target_df)*100:.1f}%)")
    
    print(f"\nTop 5 away formations:")
    away_formations = target_df['away_formation'].value_counts().head()
    for form, count in away_formations.items():
        print(f"  {form}: {count} ({count/len(target_df)*100:.1f}%)")
    
    print(f"\nDate range: {df_mapped['date_parsed'].min().strftime('%Y-%m-%d')} to {df_mapped['date_parsed'].max().strftime('%Y-%m-%d')}")
    print(f"Unique competitions: {df_mapped['competition_id'].nunique()}")
    
    # 8. Save a detailed mapping report
    print(f"\n" + "="*60)
    print("SAVING MAPPING REPORT")
    print("="*60)
    
    # Create a mapping report showing which team names map to which codes
    home_mapping = df_mapped[['home_club_name', 'home_team_code']].drop_duplicates()
    home_mapping.columns = ['team_name', 'team_code']
    home_mapping['type'] = 'home'
    
    away_mapping = df_mapped[['away_club_name', 'away_team_code']].drop_duplicates()
    away_mapping.columns = ['team_name', 'team_code']
    away_mapping['type'] = 'away'
    
    all_mappings = pd.concat([home_mapping, away_mapping]).drop_duplicates(subset=['team_name', 'team_code'])
    all_mappings = all_mappings.sort_values('team_name')
    all_mappings.to_csv(os.path.join(misc_dir, "team_name_mapping_used.csv"), index=False)
    print(f"Team mappings saved to misc/team_name_mapping_used.csv")
    print(f"Total unique team name to code mappings: {len(all_mappings)}")
    
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE!")
    print(f"{'='*60}")
    print(f"Output saved to: {output_path}")
    print(f"Supporting files saved to: {misc_dir}/")

if __name__ == "__main__":
    main()