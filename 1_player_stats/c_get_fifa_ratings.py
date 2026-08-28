import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from tqdm import tqdm

# Manual mapping dictionary for edge cases
MANUAL_NAME_MAP = {
    'F Arp': 'J Arp', 'J Bruun Larsen': 'J Larsen', 'K Prince Redondo': 'K Redondo', 
    'M Oliver Kempf': 'M Kempf', 'I Kiese Thelin': 'I Thelin', 'P de Blasis': 'P De Blasis', 
    'N Sarpei': 'H Sarpei', 'F Ole Becker': 'F Becker', 'P Ciljan Skjelbred': 'P Skjelbred', 
    'J Simun Edmundsson': 'J Edmundsson', 'H AlGhaddioui': 'H Al Ghaddioui', 
    'Cauly': 'C Oliveira Souza', 'J Petter Hauge': 'J Hauge', 'A Rahman Baba': 'B Rahman', 
    'P Osei Owusu': 'P Owusu', 'F Sorensen': 'F Srensen', 'M Cunha': 'Matheus Cunha', 
    'R Sanches': 'Renato Sanches', 'M Roca': 'Marc Roca', 'M Morey': 'Mateu Morey', 
    'J Koo': 'Koo Ja Cheol', 'F Ronnow': 'F Rnnow', 'J Blaszczykowski': 'J Baszczykowski', 
    'P Alcacer': 'Paco Alcacer', 'S Papastathopoulos': 'Sokratis', 'F Trevizan': 'Felipe', 
    'H Novoa': 'Hugo Novoa', 'G Ramos': 'Guilherme Ramos', 'A Buta': 'Aurelio Buta', 
    'T Tomas': 'Tiago Tomas', 'C Kwon': 'Kwon Chang Hoon', 'N de Medina': 'N De Medina', 
    'D Ji': 'Ji Dong Won', 'L Torro': 'Lucas Torro', 'D Lee': 'Lee Dong Jun', 
    'G Paciencia': 'Goncalo Paciencia', 'W Jeong': 'Jeong Woo Yeong', 'J Mere': 'Jorge Mere', 
    'G Fernandes': 'Gelson Fernandes', 'D Leite': 'Diogo Leite', 'O Mascarell': 'Omar Mascarell', 
    'H Hwang': 'Hwang Hee Chan', 'P Maffeo': 'Pablo Maffeo', 'I Medeiros': 'Iuri Medeiros', 
    'B Bialek': 'B Biaek', 'G Dias': 'Gil Dias', 'J Samperio': 'Jairo', 'C Lee': 'Lee Chung Yong', 
    'J Cancelo': 'Joao Cancelo', 'I Abass': 'A Issah', 'T Dantas': 'Tiago Dantas', 
    'M Bartra': 'Marc Bartra', 'L Piszczek': 'Piszczek', 'I Camacho': 'Camacho', 
    'L Oztunali': 'L Oztunal', 'D Santos': 'Douglas Santos', 'J Anthony Brooks': 'J Brooks', 
    'J Bernat': 'Juan Bernat', 'M Shabani': 'E Shabani', 'N Joel Sarenren Bazee': 'N Sarenren Bazee', 
    'A BellaKotchap': 'A Bella Kotchap', 'P Coutinho': 'Coutinho', 'P Otavio': 'Paulo Otavio', 
    'D Olmo': 'Dani Olmo', 'K Ofori': 'E Ofori', 'J Barrett Laursen': 'J Laursen', 
    'J Manuel Mbom': 'J Mbom', 'J Klauss': 'Klauss', 'E Quaresma': 'Eduardo Quaresma', 
    'L Barreiro': 'L Barreiro Martins', 'D Soares': 'Danilo', 'P Kunde': 'K Malong', 
    'Jordan': 'J Siebatcheu', 'Piszczek': ' Piszczek', 'E Maxim ChoupoMoting': 'E ChoupoMoting', 
    'J Lee': 'Lee Jae Sung', 'A Martin': 'Aaron', 'A Silva': 'Andre Silva', 'M Kone': 'K Kone', 
    'Javi Martinez': 'Javier Martinez Aginaga', '': ''
}

# Hardcoded position column groups - exactly as they appear in lineups.csv
POSITION_COLUMNS = {
    'Home_Goalkeeper': ['Home_Goalkeeper'],
    'Home_Defender': ['Home_Defender_1', 'Home_Defender_2', 'Home_Defender_3', 'Home_Defender_4', 
                      'Home_Defender_5', 'Home_Defender_6', 'Home_Defender_7'],
    'Home_Midfielder': ['Home_Midfielder_1', 'Home_Midfielder_2', 'Home_Midfielder_3', 'Home_Midfielder_4', 
                        'Home_Midfielder_5', 'Home_Midfielder_6', 'Home_Midfielder_7', 'Home_Midfielder_8', 
                        'Home_Midfielder_9'],
    'Home_Attacker': ['Home_Attacker_1', 'Home_Attacker_2', 'Home_Attacker_3', 'Home_Attacker_4', 
                      'Home_Attacker_5', 'Home_Attacker_6', 'Home_Attacker_7'],
    'Away_Goalkeeper': ['Away_Goalkeeper'],
    'Away_Defender': ['Away_Defender_1', 'Away_Defender_2', 'Away_Defender_3', 'Away_Defender_4', 
                      'Away_Defender_5', 'Away_Defender_6', 'Away_Defender_7'],
    'Away_Midfielder': ['Away_Midfielder_1', 'Away_Midfielder_2', 'Away_Midfielder_3', 'Away_Midfielder_4', 
                        'Away_Midfielder_5', 'Away_Midfielder_6', 'Away_Midfielder_7'],
    'Away_Attacker': ['Away_Attacker_1', 'Away_Attacker_2', 'Away_Attacker_3', 'Away_Attacker_4', 
                      'Away_Attacker_5', 'Away_Attacker_6']
}

# FIFA version release dates (approximate September/October of each year)
FIFA_VERSION_DATES = {
    15: datetime(2014, 9, 1),
    16: datetime(2015, 9, 1),
    17: datetime(2016, 9, 1),
    18: datetime(2017, 9, 1),
    19: datetime(2018, 9, 1),
    20: datetime(2019, 9, 1),
    21: datetime(2020, 10, 1),
    22: datetime(2021, 10, 1),
    23: datetime(2022, 9, 1),
    24: datetime(2023, 9, 1),
    25: datetime(2024, 9, 1),
    26: datetime(2025, 9, 1),
}


def get_fifa_version_for_date(match_date):
    """
    Find the closest FIFA version available for a given date.
    Returns the FIFA version number that should be used for that date.
    """
    if isinstance(match_date, str):
        match_date = pd.to_datetime(match_date)
    
    # Find the FIFA version closest to (but not after) the match date
    applicable_versions = {v: date for v, date in FIFA_VERSION_DATES.items() if date <= match_date}
    
    if not applicable_versions:
        # If match is before FIFA 15, use FIFA 15
        return 15
    
    # Return the latest version that applies to this date
    return max(applicable_versions.keys())


def process_fifa_ratings():
    # FIXED: Go up one level from the script directory to reach project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)  # Go up from 1_player_stats to project root
    
    # FIXED: Paths now point to project-level directories
    lineups_path = os.path.join(project_dir, 'data', 'lineups.csv')
    values_path = os.path.join(project_dir, 'data', 'lineups_values.csv')
    player_ratings_path = os.path.join(project_dir, 'fifa_ratings', 'player_ratings.csv')
    output_path = os.path.join(project_dir, 'data', 'lineups_values_ratings.csv')
    fifa_cache_path = os.path.join(project_dir, 'data', 'fifa_name_cache.json')

    try:
        df_lineups = pd.read_csv(lineups_path)
        df_values = pd.read_csv(values_path)
        df_player_ratings = pd.read_csv(player_ratings_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print(f"Make sure lineups.csv, lineups_values.csv, and fifa_ratings/player_ratings.csv exist.")
        return

    # Ensure game_id exists in lineups and values (needed for incremental joins).
    if 'game_id' not in df_lineups.columns:
        if {'Date', 'Home Team', 'Away Team'}.issubset(df_lineups.columns):
            df_lineups['game_id'] = (
                df_lineups['Date'].astype(str).str.replace('-', '', regex=False)
                + '_' + df_lineups['Home Team'].astype(str)
                + '_' + df_lineups['Away Team'].astype(str)
            )
            print("Warning: game_id missing in lineups.csv, derived fallback IDs from Date/Home/Away.")
        else:
            raise KeyError("lineups.csv must contain game_id or Date/Home Team/Away Team columns.")

    if 'game_id' not in df_values.columns:
        if {'Date', 'Home Team', 'Away Team'}.issubset(df_values.columns):
            game_key_map = (
                df_lineups[['Date', 'Home Team', 'Away Team', 'game_id']]
                .drop_duplicates(subset=['Date', 'Home Team', 'Away Team'])
            )
            df_values = df_values.merge(game_key_map, on=['Date', 'Home Team', 'Away Team'], how='left')
            print("Backfilled game_id in lineups_values.csv using Date/Home/Away.")
        else:
            raise KeyError("lineups_values.csv must contain game_id or Date/Home Team/Away Team columns.")

    # Determine pending games: only compute FIFA features for values rows not present yet.
    df_existing = None
    processed_ids = set()
    if os.path.exists(output_path):
        df_existing = pd.read_csv(output_path, low_memory=False)
        if 'game_id' not in df_existing.columns:
            if {'Date', 'Home Team', 'Away Team'}.issubset(df_existing.columns):
                game_key_map_values = (
                    df_values[['Date', 'Home Team', 'Away Team', 'game_id']]
                    .drop_duplicates(subset=['Date', 'Home Team', 'Away Team'])
                )
                df_existing = df_existing.merge(
                    game_key_map_values,
                    on=['Date', 'Home Team', 'Away Team'],
                    how='left'
                )
                print("Backfilled game_id in existing lineups_values_ratings.csv using Date/Home/Away.")
            else:
                df_existing['game_id'] = np.nan
        processed_ids = set(df_existing['game_id'].dropna().astype(str))

    df_values_pending = df_values[~df_values['game_id'].astype(str).isin(processed_ids)].copy()
    df_lineups_pending = df_lineups[df_lineups['game_id'].isin(df_values_pending['game_id'])].copy()

    print(f"Existing processed games: {len(processed_ids)}")
    print(f"Pending games for FIFA lookup: {len(df_values_pending)}")

    if df_values_pending.empty:
        if df_existing is not None:
            df_existing.to_csv(output_path, index=False, encoding='utf-8-sig')
            print("No pending games. Kept existing lineups_values_ratings.csv (with game_id column).")
        else:
            print("No pending games and no existing output. Nothing to process.")
        return

    # Validate which position columns actually exist in the dataframe
    available_position_columns = {}
    missing_columns = []
    
    for position, cols in POSITION_COLUMNS.items():
        available_cols = [col for col in cols if col in df_lineups.columns]
        if available_cols:
            available_position_columns[position] = available_cols
        else:
            missing_columns.extend(cols)
    
    if missing_columns:
        print(f"Warning: {len(missing_columns)} expected columns not found in lineups.csv")
    
    print(f"Using {sum(len(v) for v in available_position_columns.values())} player columns in {len(available_position_columns)} position groups")

    print("Loading player ratings data...")
    # Convert fifa_version to int for easier comparison
    df_player_ratings['fifa_version'] = df_player_ratings['fifa_version'].astype(int)
    
    # Create lookup dictionaries for each FIFA version
    fifa_lookups = {}
    for fifa_ver in sorted(df_player_ratings['fifa_version'].unique()):
        df_ver = df_player_ratings[df_player_ratings['fifa_version'] == fifa_ver]
        # Create a mapping of player name to their ratings
        fifa_lookups[fifa_ver] = {}
        for _, row in df_ver.iterrows():
            player_name = row['short_name']
            fifa_lookups[fifa_ver][player_name] = row.to_dict()
        print(f"  Loaded FIFA {fifa_ver}: {len(fifa_lookups[fifa_ver])} players")

    # Cache for player lookup resolution: key is "<fifa_version>|<player_name>"
    fifa_name_cache = {}
    if os.path.exists(fifa_cache_path):
        try:
            with open(fifa_cache_path, 'r', encoding='utf-8') as f:
                raw_cache = json.load(f)
            fifa_name_cache = {str(k): v for k, v in raw_cache.items()}
            print(f"Loaded FIFA name cache entries: {len(fifa_name_cache)}")
        except Exception as e:
            print(f"Warning: Failed to load FIFA name cache ({e}). Rebuilding cache this run.")
            fifa_name_cache = {}

    # Enhanced match statistics
    match_stats = {
        'exact_full_name': 0,
        'exact_last_name': 0,
        'manual_mapping': 0,
        'fuzzy': 0,
        'missing': 0,
        'total_lookups': 0
    }
    missing_players = []
    matched_players = {}  # Track which players were matched and how

    def get_player_stats_for_version(player_name, fifa_version):
        """Get player stats from a specific FIFA version."""
        match_stats['total_lookups'] += 1
        
        if fifa_version not in fifa_lookups:
            return None, 'missing'

        cache_key = f"{int(fifa_version)}|{player_name}"
        cached = fifa_name_cache.get(cache_key)
        if cached is not None:
            matched_name = cached.get('matched_name')
            match_type = cached.get('match_type', 'missing')
            if matched_name and matched_name in fifa_lookups[fifa_version]:
                return fifa_lookups[fifa_version][matched_name], match_type
            if not matched_name:
                return None, 'missing'
        
        # Check if this player was manually mapped
        was_manually_mapped = player_name in MANUAL_NAME_MAP and MANUAL_NAME_MAP[player_name] != player_name
        
        # Apply manual mapping
        mapped_name = MANUAL_NAME_MAP.get(player_name, player_name)
        
        # Try exact match
        if mapped_name in fifa_lookups[fifa_version]:
            resolved_type = 'manual_mapping' if was_manually_mapped else 'exact_full_name'
            fifa_name_cache[cache_key] = {
                'matched_name': mapped_name,
                'match_type': resolved_type,
            }
            if was_manually_mapped:
                return fifa_lookups[fifa_version][mapped_name], 'manual_mapping'
            return fifa_lookups[fifa_version][mapped_name], 'exact_full_name'
        
        # Try last word match
        last_word = mapped_name.split()[-1] if mapped_name.split() else mapped_name
        for player_key, stats in fifa_lookups[fifa_version].items():
            if player_key.endswith(last_word):
                fifa_name_cache[cache_key] = {
                    'matched_name': player_key,
                    'match_type': 'exact_last_name',
                }
                return stats, 'exact_last_name'
        
        fifa_name_cache[cache_key] = {
            'matched_name': None,
            'match_type': 'missing',
        }
        return None, 'missing'

    def get_group_stats(row, position_group, fifa_version):
        """
        Calculate average stats for a group of players for a specific position and team.
        Uses the specified FIFA version and hardcoded column groups.
        """
        if position_group not in available_position_columns:
            return {}
        
        cols = available_position_columns[position_group]
        
        # Determine which stats to collect based on position
        if 'Goalkeeper' in position_group:
            stat_keys = ['overall', 'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking',
                        'goalkeeping_positioning', 'goalkeeping_reflexes', 'goalkeeping_speed']
        else:
            stat_keys = ['overall', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
        
        collected_stats = {key: [] for key in stat_keys}
        
        for col in cols:
            player_name = str(row[col]).strip()
            
            if player_name and player_name.lower() != 'nan':
                stats_dict, match_type = get_player_stats_for_version(player_name, fifa_version)
                
                if stats_dict is not None:
                    match_stats[match_type] += 1
                    # Track match quality
                    if player_name not in matched_players:
                        matched_players[player_name] = match_type
                    
                    for key in stat_keys:
                        val = stats_dict.get(key)
                        if pd.notna(val):
                            try:
                                collected_stats[key].append(float(val))
                            except (ValueError, TypeError):
                                pass
                else:
                    match_stats['missing'] += 1
                    missing_players.append(player_name)
        
        # Calculate averages
        result_avgs = {}
        # Extract prefix and group for column naming
        parts = position_group.split('_', 1)
        if len(parts) == 2:
            prefix, group = parts
            for key in stat_keys:
                col_name = f"{prefix}{group}_{key}_Avg"
                if collected_stats[key]:
                    result_avgs[col_name] = round(np.mean(collected_stats[key]), 2)
                else:
                    result_avgs[col_name] = np.nan
        
        return result_avgs

    # Process each match
    print("\nProcessing matches and calculating averaged FIFA ratings...")
    fifa_results = []
    
    for idx, row in tqdm(df_lineups_pending.iterrows(), total=len(df_lineups_pending), desc="Processing matches"):
        match_date = row.get('Date', '')
        fifa_version = get_fifa_version_for_date(match_date)
        
        match_summary = {
            'game_id': row.get('game_id', ''),
            'Date': row.get('Date', ''),
            'Home Team': row.get('Home Team', ''),
            'Away Team': row.get('Away Team', ''),
            'FIFA_Version_Used': fifa_version
        }
        
        # Process each position group using hardcoded columns
        for position_group in available_position_columns.keys():
            group_averages = get_group_stats(row, position_group, fifa_version)
            match_summary.update(group_averages)
        
        fifa_results.append(match_summary)

    df_fifa_results = pd.DataFrame(fifa_results)
    
    # Merge FIFA ratings with market values
    print("Merging FIFA ratings with market values...")
    df_combined_new = pd.merge(df_values_pending, df_fifa_results, on='game_id', how='left', suffixes=('', '_fifa'))

    # Resolve duplicate key columns from merge and keep canonical base columns from df_values_pending.
    for col in ['Date', 'Home Team', 'Away Team']:
        alt = f"{col}_fifa"
        if alt in df_combined_new.columns:
            if col not in df_combined_new.columns:
                df_combined_new[col] = df_combined_new[alt]
            df_combined_new = df_combined_new.drop(columns=[alt])

    if df_existing is not None:
        df_combined = pd.concat([df_existing, df_combined_new], ignore_index=True)
    else:
        df_combined = df_combined_new

    if 'game_id' in df_combined.columns:
        df_combined = df_combined.drop_duplicates(subset=['game_id'], keep='first')

    if 'Date' in df_combined.columns:
        df_combined = df_combined.sort_values('Date').reset_index(drop=True)

    # Save the result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_combined.to_csv(output_path, index=False, encoding='utf-8-sig')

    # Persist FIFA name cache for future runs.
    os.makedirs(os.path.dirname(fifa_cache_path), exist_ok=True)
    with open(fifa_cache_path, 'w', encoding='utf-8') as f:
        json.dump(fifa_name_cache, f, ensure_ascii=False, indent=2)
    print(f"Saved FIFA name cache: {fifa_cache_path} ({len(fifa_name_cache)} entries)")
    
    # Enhanced matching statistics output
    total_matched = match_stats['exact_full_name'] + match_stats['exact_last_name'] + match_stats['manual_mapping']
    total_lookups = match_stats['total_lookups']
    total_lookups_safe = max(total_lookups, 1)
    
    print("\n" + "="*60)
    print("FIFA RATINGS MATCHING STATISTICS")
    print("="*60)
    print(f"Total player lookups: {total_lookups:,}")
    print(f"")
    print(f"Match Quality Breakdown:")
    print(f"  Exact full name matches: {match_stats['exact_full_name']:,} ({match_stats['exact_full_name']/total_lookups_safe*100:.1f}%)")
    print(f"  Exact last name matches: {match_stats['exact_last_name']:,} ({match_stats['exact_last_name']/total_lookups_safe*100:.1f}%)")
    print(f"  Manual mapping matches:  {match_stats['manual_mapping']:,} ({match_stats['manual_mapping']/total_lookups_safe*100:.1f}%)")
    print(f"  Total matched:           {total_matched:,} ({total_matched/total_lookups_safe*100:.1f}%)")
    print(f"  Missing:                 {match_stats['missing']:,} ({match_stats['missing']/total_lookups_safe*100:.1f}%)")
    
    # Unique player statistics
    unique_players_looked_up = len(set(list(matched_players.keys()) + missing_players))
    unique_matched = len(matched_players)
    unique_missing = len(set(missing_players))
    unique_players_looked_up_safe = max(unique_players_looked_up, 1)
    
    print(f"\nUnique Player Statistics:")
    print(f"  Unique players found:    {unique_players_looked_up:,}")
    print(f"  Unique players matched:  {unique_matched:,} ({unique_matched/unique_players_looked_up_safe*100:.1f}%)")
    print(f"  Unique players missing:  {unique_missing:,} ({unique_missing/unique_players_looked_up_safe*100:.1f}%)")
    
    # Match type distribution for unique players
    unique_match_types = {}
    for player, match_type in matched_players.items():
        unique_match_types[match_type] = unique_match_types.get(match_type, 0) + 1
    
    print(f"\nUnique Player Match Types:")
    for match_type in ['exact_full_name', 'exact_last_name', 'manual_mapping']:
        count = unique_match_types.get(match_type, 0)
        print(f"  {match_type}: {count:,} ({count/unique_matched*100:.1f}% of matched)")
    
    if unique_missing > 0:
        unique_missing_list = list(set(missing_players))
        print(f"\nSample of missing players (first 15):")
        for player in unique_missing_list[:15]:
            print(f"  - {player}")
        if len(unique_missing_list) > 15:
            print(f"  ... and {len(unique_missing_list) - 15} more")
    
    print(f"\nNew matches processed: {len(df_lineups_pending)}")
    print(f"Total matches in output: {len(df_combined)}")
    print(f"Output columns: {len(df_combined.columns)}")
    print(f"Saved to: {output_path}")
    print("="*60)


if __name__ == "__main__":
    process_fifa_ratings()