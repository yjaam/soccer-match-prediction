"""
Script to extract historic lineups from transfermarkt data and append to lineups.csv
Matches the column format of the existing lineups.csv file
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import warnings

warnings.filterwarnings('ignore')

# Define paths
TRANSFERMARKT_DIR = Path('transfermarkt_data')
DATA_DIR = Path('data')
GAMES_FILE = TRANSFERMARKT_DIR / 'games.csv'
GAME_LINEUPS_FILE = TRANSFERMARKT_DIR / 'game_lineups.csv'
COMPETITIONS_FILE = TRANSFERMARKT_DIR / 'competitions.csv'
OUTPUT_LINEUPS_FILE = DATA_DIR / 'lineups.csv'
BACKUP_LINEUPS_FILE = DATA_DIR / 'lineups_backup.csv'

def get_position_category(position):
    """Map Transfermarkt positions to the 4 standardized groups.
    
    Logic aligned with a1_scrape_new_formations.py buckets:
    1. GK -> Goalkeeper
    2. Centre-Forward, Left Winger, Right Winger, Second Striker, Attack -> Attacker
    3. Central Midfield, Defensive Midfield, Attacking Midfield, Right Midfield, Left Midfield, Midfield -> Midfielder
    4. Centre-Back, Left-Back, Right-Back, Defender, Sweeper -> Defender
    5. Default -> Midfielder
    """
    position = str(position).strip() if pd.notna(position) else ""

    if not position:
        return 'Midfielder'

    # Standardize phrasing (Transfermarkt uses hyphens)
    pos_lower = position.lower()

    # 1. Goalkeeper
    if 'goalkeeper' in pos_lower:
        return 'Goalkeeper'

    # 2. Attacker
    attacker_terms = ['forward', 'winger', 'striker', 'attack']
    if any(term in pos_lower for term in attacker_terms):
        return 'Attacker'

    # 3. Defender
    defender_terms = ['back', 'defender', 'sweeper', 'libero']
    if any(term in pos_lower for term in defender_terms):
        return 'Defender'

    # 4. Midfielder
    midfield_terms = ['midfield']
    if any(term in pos_lower for term in midfield_terms):
        return 'Midfielder'

    # Default
    return 'Midfielder'

def build_historic_lineups():
    """Build historic lineups from transfermarkt data"""
    
    print("Loading data files...")
    
    # Load competitions mapping
    competitions_df = pd.read_csv(COMPETITIONS_FILE)
    comp_mapping = dict(zip(competitions_df['competition_id'], competitions_df['name']))
    
    # Load games
    games_df = pd.read_csv(GAMES_FILE)
    print(f"Loaded {len(games_df)} games")
    
    # Load game lineups
    game_lineups_df = pd.read_csv(GAME_LINEUPS_FILE)
    print(f"Loaded {len(game_lineups_df)} lineup entries")
    
    # Filter for starting lineups only
    game_lineups_df = game_lineups_df[game_lineups_df['type'] == 'starting_lineup'].copy()
    print(f"Filtered to {len(game_lineups_df)} starting lineup entries")
    
    # Add position categories
    game_lineups_df['position_category'] = game_lineups_df['position'].apply(get_position_category)
    
    # Get the expected lineups.csv format
    if OUTPUT_LINEUPS_FILE.exists():
        existing_lineups = pd.read_csv(OUTPUT_LINEUPS_FILE)
        expected_columns = list(existing_lineups.columns)
        print(f"\nExpected columns from existing lineups: {len(expected_columns)}")
        print(expected_columns[:10], "...")
    else:
        expected_columns = None
    
    # Merge games with game_lineups - rename date columns to avoid conflict
    game_lineups_df = game_lineups_df.rename(columns={'date': 'lineup_date'})
    merged_df = game_lineups_df.merge(games_df, left_on='game_id', right_on='game_id', how='left')
    print(f"\nMerged data shape: {merged_df.shape}")
    
    # Group by game to build lineup rows
    lineup_rows = []
    games_processed = 0
    
    for game_id, group in merged_df.groupby('game_id'):
        try:
            # Get game info
            game_info = group.iloc[0]
            
            competition_id = game_info['competition_id']
            competition_name = comp_mapping.get(competition_id, 'Unknown')
            date = game_info['date']  # This is now the date from games.csv
            home_team = game_info['home_club_name']
            away_team = game_info['away_club_name']
            
            # Skip if missing essential info
            if pd.isna(home_team) or pd.isna(away_team) or pd.isna(date):
                continue
            
            # Separate home and away players
            home_players = group[group['club_id'] == game_info['home_club_id']].copy()
            away_players = group[group['club_id'] == game_info['away_club_id']].copy()
            
            # Create row dict
            row = {
                'Competition': competition_name,
                'Date': date,
                'Home Team': home_team,
                'Away Team': away_team,
            }
            
            # Helper function to safely sort players by number
            def sort_players_by_number(players_df):
                """Sort players by number, converting to numeric and handling mixed types"""
                try:
                    players_df = players_df.copy()
                    players_df['number_numeric'] = pd.to_numeric(players_df['number'], errors='coerce')
                    players_df = players_df.dropna(subset=['number_numeric'])
                    return players_df.sort_values('number_numeric')
                except:
                    return players_df
            
            # Process home team positions
            for pos_cat in ['Goalkeeper', 'Defender', 'Midfielder', 'Attacker']:
                pos_players = sort_players_by_number(home_players[home_players['position_category'] == pos_cat])
                for idx, (_, player) in enumerate(pos_players.iterrows(), 1):
                    if pos_cat == 'Goalkeeper':
                        key = 'Home_Goalkeeper'
                    else:
                        key = f'Home_{pos_cat}_{idx}'
                    row[key] = player['player_name']
            
            # Process away team positions
            for pos_cat in ['Goalkeeper', 'Defender', 'Midfielder', 'Attacker']:
                pos_players = sort_players_by_number(away_players[away_players['position_category'] == pos_cat])
                for idx, (_, player) in enumerate(pos_players.iterrows(), 1):
                    if pos_cat == 'Goalkeeper':
                        key = 'Away_Goalkeeper'
                    else:
                        key = f'Away_{pos_cat}_{idx}'
                    row[key] = player['player_name']
            
            lineup_rows.append(row)
            games_processed += 1
            
            if games_processed % 5000 == 0:
                print(f"Processed {games_processed} games...")
        
        except Exception as e:
            print(f"Error processing game {game_id}: {e}")
            continue
    
    print(f"\nSuccessfully processed {games_processed} games")
    
    # Create dataframe
    historic_lineups = pd.DataFrame(lineup_rows)
    print(f"\nHistoric lineups shape: {historic_lineups.shape}")
    print(f"Columns: {list(historic_lineups.columns)[:10]}...")
    
    # Ensure all expected columns exist (fill missing with empty strings)
    if expected_columns:
        for col in expected_columns:
            if col not in historic_lineups.columns:
                historic_lineups[col] = ''
        
        # Reorder columns to match existing lineups
        historic_lineups = historic_lineups[expected_columns]
        
        print(f"Column alignment completed. Using {len(expected_columns)} columns")
    
    return historic_lineups

def append_to_lineups(historic_lineups):
    """Safely append historic lineups to existing lineups.csv"""
    
    if not OUTPUT_LINEUPS_FILE.exists():
        print(f"\n{OUTPUT_LINEUPS_FILE} does not exist. Creating new file...")
        historic_lineups.to_csv(OUTPUT_LINEUPS_FILE, index=False)
        print(f"Created {OUTPUT_LINEUPS_FILE} with {len(historic_lineups)} rows")
        return
    
    # Create backup
    print(f"\nCreating backup of existing lineups...")
    existing_lineups = pd.read_csv(OUTPUT_LINEUPS_FILE)
    existing_lineups.to_csv(BACKUP_LINEUPS_FILE, index=False)
    print(f"Backup created: {BACKUP_LINEUPS_FILE}")
    
    # Identify duplicate rows based on Competition, Date, Home Team, Away Team
    print(f"\nRemoving duplicates...")
    existing_lineups['game_key'] = (
        existing_lineups['Competition'] + '|' +
        existing_lineups['Date'].astype(str) + '|' +
        existing_lineups['Home Team'] + '|' +
        existing_lineups['Away Team']
    )
    
    historic_lineups['game_key'] = (
        historic_lineups['Competition'] + '|' +
        historic_lineups['Date'].astype(str) + '|' +
        historic_lineups['Home Team'] + '|' +
        historic_lineups['Away Team']
    )
    
    existing_keys = set(existing_lineups['game_key'])
    historic_new = historic_lineups[~historic_lineups['game_key'].isin(existing_keys)].copy()
    
    print(f"Existing rows: {len(existing_lineups)}")
    print(f"Historic rows attempted: {len(historic_lineups)}")
    print(f"New rows to append: {len(historic_new)}")
    
    # Drop the temporary key column
    existing_lineups = existing_lineups.drop(columns=['game_key'])
    historic_new = historic_new.drop(columns=['game_key'])
    
    # Append
    combined = pd.concat([existing_lineups, historic_new], ignore_index=True)
    
    # Sort by date
    combined['Date'] = pd.to_datetime(combined['Date'])
    combined = combined.sort_values('Date').reset_index(drop=True)
    
    # Save
    print(f"\nSaving combined lineups...")
    combined.to_csv(OUTPUT_LINEUPS_FILE, index=False)
    print(f"Successfully saved {len(combined)} total rows to {OUTPUT_LINEUPS_FILE}")
    
    print(f"\nSummary:")
    print(f"  - Original rows: {len(existing_lineups)}")
    print(f"  - New rows added: {len(historic_new)}")
    print(f"  - Total rows: {len(combined)}")

if __name__ == "__main__":
    print("=" * 60)
    print("Extracting Historic Lineups from Transfermarkt Data")
    print("=" * 60)
    
    # Build historic lineups
    historic_lineups = build_historic_lineups()
    
    # Append to existing lineups
    append_to_lineups(historic_lineups)
    
    print("\n" + "=" * 60)
    print("Process completed successfully!")
    print("=" * 60)
