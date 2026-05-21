import kagglehub
import shutil
import json
import pandas as pd
import numpy as np
from pathlib import Path
import requests

script_dir = Path(__file__).resolve().parent
fifa_data_dir = script_dir / "fifa_data"
fifa_ratings_dir = script_dir / "fifa_ratings"

# Create directories if they don't exist
fifa_data_dir.mkdir(exist_ok=True)
fifa_ratings_dir.mkdir(exist_ok=True)

# Download latest version
print("Downloading EA Sports FC 24 dataset from kagglehub...")
path = kagglehub.dataset_download("stefanoleone992/ea-sports-fc-24-complete-player-dataset")

print(f"Downloaded to: {path}")

# Find and copy male_players.csv to fifa_data directory
source_file = Path(path) / "male_players.csv"
if source_file.exists():
    dest_file = fifa_data_dir / "male_players.csv"
    print(f"Copying {source_file.name} to {dest_file}...")
    shutil.copy2(source_file, dest_file)
    print(f"✓ Successfully saved to {dest_file}")
else:
    print(f"⚠ male_players.csv not found in {path}")
    print(f"Available files: {list(Path(path).glob('*.csv'))[:5]}")
    exit(1)

# Create players.json from male_players.csv for faster loading
print("\nCreating players.json...")
fifa_cols = ['short_name', 'fifa_version', 'overall', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
             'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 
             'goalkeeping_positioning', 'goalkeeping_reflexes', 'goalkeeping_speed']

df_fifa = pd.read_csv(dest_file, usecols=fifa_cols)

# Sort by overall rating descending to prioritize top-flight players
df_fifa = df_fifa.sort_values(by='overall', ascending=False)

# Save to CSV in fifa_ratings directory
csv_dest = fifa_ratings_dir / "player_ratings.csv"
print(f"Saving player ratings to {csv_dest}...")
df_fifa.to_csv(csv_dest, index=False, encoding='utf-8-sig')
print(f"✓ Successfully saved to {csv_dest}")

# Convert to list of dicts
players_list = df_fifa.to_dict('records')

# Replace NaN with None for JSON compatibility
def convert_nan_to_none(obj):
    if isinstance(obj, dict):
        return {k: convert_nan_to_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_nan_to_none(v) for v in obj]
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    else:
        return obj

players_list = convert_nan_to_none(players_list)

# Save to players.json
players_json_path = script_dir / "players.json"
with open(players_json_path, 'w', encoding='utf-8') as f:
    json.dump(players_list, f)

file_size_mb = players_json_path.stat().st_size / 1024 / 1024
print(f"✓ Created {players_json_path} ({file_size_mb:.2f} MB) with {len(players_list)} players")

# Fetch and append data from MSMC API for newer FIFA versions (FC25, FC26)
print("\n--- Fetching newer FIFA data from MSMC API ---")
api_base_url = 'https://api.msmc.cc/api/eafc/players'

# Column mapping from API field names to our existing column names
api_column_map = {
    'name': 'short_name',
    'ovr': 'overall',
    'pac': 'pace',
    'sho': 'shooting',
    'pas': 'passing',
    'dri': 'dribbling',
    'def': 'defending',
    'phy': 'physic',
    'diving': 'goalkeeping_diving',
    'handling': 'goalkeeping_handling',
    'kicking': 'goalkeeping_kicking',
    'positioning': 'goalkeeping_positioning',
    'reflexes': 'goalkeeping_reflexes',
}

required_columns = ['fifa_version', 'short_name', 'overall', 'pace', 'shooting', 'passing', 
                    'dribbling', 'defending', 'physic', 'goalkeeping_diving', 'goalkeeping_handling', 
                    'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes', 'goalkeeping_speed']

# Load existing CSV data
csv_path = fifa_ratings_dir / "player_ratings.csv"
if csv_path.exists():
    df_existing = pd.read_csv(csv_path)
    print(f"Loaded existing data: {len(df_existing)} players")
    existing_players = set(f"{row['short_name']}_{row['fifa_version']}" for _, row in df_existing.iterrows())
else:
    df_existing = pd.DataFrame(columns=required_columns)
    existing_players = set()

# Fetch data from API for FC25 and FC26
api_data_list = []
games = [('fc25', '1'), ('fc26', '1'), ('fc26', '2')]

for game, update in games:
    fifa_num = game.replace('fc', '')
    print(f"\nFetching {game} update {update}...")
    
    try:
        # Fetch players for this game/update
        url = f'{api_base_url}?game={game}'
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            api_players = response.json()
            print(f"  Downloaded {len(api_players)} players from API")
            
            # Filter and map the data
            added_count = 0
            for player in api_players:
                # Create unique identifier
                player_id = f"{player.get('name')}_{fifa_num}"
                
                # Skip if already exists in our dataset
                if player_id in existing_players:
                    continue
                
                # Extract relevant columns only
                mapped_player = {'fifa_version': float(fifa_num)}
                
                for api_field, csv_field in api_column_map.items():
                    value = player.get(api_field)
                    # Convert to float if it's a numeric field
                    if value and csv_field != 'short_name':
                        try:
                            mapped_player[csv_field] = float(value) if value != '' else None
                        except (ValueError, TypeError):
                            mapped_player[csv_field] = None
                    else:
                        mapped_player[csv_field] = value
                
                # Add goalkeeping_speed if available (defaults to None if missing)
                if 'goalkeeping_speed' not in mapped_player:
                    mapped_player['goalkeeping_speed'] = None
                
                api_data_list.append(mapped_player)
                added_count += 1
            
            print(f"  Added {added_count} new players to dataset")
        else:
            print(f"  Failed to fetch (Status: {response.status_code})")
            
    except Exception as e:
        print(f"  Error fetching {game}: {e}")

# Combine existing and new API data
if api_data_list:
    df_api = pd.DataFrame(api_data_list)
    print(f"\nCombining data: {len(df_existing)} existing + {len(df_api)} API players")
    
    # Ensure columns match
    for col in required_columns:
        if col not in df_api.columns:
            df_api[col] = None
    
    # Concatenate with existing data
    df_combined = pd.concat([df_existing, df_api[required_columns]], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=['short_name', 'fifa_version'], keep='first')
else:
    df_combined = df_existing

print(f"Final dataset: {len(df_combined)} total players")

# Save combined CSV
df_combined.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"✓ Updated {csv_path}")

# Update players.json with combined data (for existing columns)
existing_cols = ['short_name', 'overall', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
                 'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 
                 'goalkeeping_positioning', 'goalkeeping_reflexes', 'goalkeeping_speed']

# Keep the fifa_version in json for reference
json_cols = ['fifa_version'] + existing_cols

# Filter to only these columns for JSON
df_for_json = df_combined[json_cols].copy()

# Sort by overall rating descending
df_for_json = df_for_json.sort_values('overall', ascending=False, na_position='last')

# Convert to list and replace NaN with None
combined_players = df_for_json.to_dict('records')
combined_players = convert_nan_to_none(combined_players)

with open(players_json_path, 'w', encoding='utf-8') as f:
    json.dump(combined_players, f)

file_size_mb = players_json_path.stat().st_size / 1024 / 1024
print(f"✓ Updated {players_json_path} ({file_size_mb:.2f} MB) with {len(combined_players)} players")

print("\n--- FIFA Data Loading Complete ---")
print(f"fifa_data/male_players.csv: Ready")
print(f"fifa_ratings/player_ratings.csv: Ready (filtered columns + fifa_version)")
print(f"players.json: Ready (optimized for fast lookups)")