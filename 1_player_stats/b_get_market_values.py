import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm

# Try to import rapidfuzz for faster matching, fall back to difflib
try:
    from rapidfuzz import process, fuzz
    USING_RAPIDFUZZ = True
except ImportError:
    import difflib
    USING_RAPIDFUZZ = False
    print("Note: Install 'rapidfuzz' for faster fuzzy matching: pip install rapidfuzz")

# Your manual mapping dictionary
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

# Hardcoded player columns - these are all the columns that contain player names
PLAYER_COLS = [
    'Home_Goalkeeper', 'Home_Defender_1', 'Home_Defender_2', 'Home_Defender_3', 
    'Home_Defender_4', 'Home_Midfielder_1', 'Home_Midfielder_2', 'Home_Attacker_1', 
    'Home_Attacker_2', 'Home_Attacker_3', 'Home_Attacker_4', 'Away_Goalkeeper', 
    'Away_Defender_1', 'Away_Defender_2', 'Away_Defender_3', 'Away_Defender_4', 
    'Away_Midfielder_1', 'Away_Midfielder_2', 'Away_Attacker_1', 'Away_Attacker_2', 
    'Away_Attacker_3', 'Away_Attacker_4', 'Home_Midfielder_3', 'Home_Midfielder_4', 
    'Away_Midfielder_3', 'Away_Midfielder_4', 'Home_Midfielder_5', 'Away_Attacker_5', 
    'Home_Attacker_5', 'Away_Defender_5', 'Away_Midfielder_5', 'Home_Defender_5', 
    'Home_Midfielder_6', 'Home_Defender_6', 'Away_Defender_6', 'Away_Attacker_6', 
    'Home_Midfielder_7', 'Home_Attacker_6', 'Away_Midfielder_6', 'Home_Defender_7', 
    'Away_Midfielder_7', 'Away_Defender_7', 'Home_Midfielder_8', 'Home_Midfielder_9', 
    'Home_Attacker_7'
]

# Position group mapping for computing averages
POSITION_GROUPS = {
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

# Matching threshold
SIMILARITY_THRESHOLD = 0.8
THRESHOLD_STEP = 0.05
MIN_THRESHOLD = 0.5
MIN_IMPROVEMENT_PERCENT = 1.0

def calculate_similarity_matrix(lineup_names, market_names):
    """
    Calculate similarity matrix between all lineup names and market value names.
    Returns a matrix where each cell [i,j] contains the similarity score.
    """
    n_lineups = len(lineup_names)
    n_market = len(market_names)
    
    print(f"Creating similarity matrix: {n_lineups} lineup names × {n_market} market names")
    print(f"Total comparisons: {n_lineups * n_market:,}")
    
    # Initialize similarity matrix
    similarity_matrix = np.zeros((n_lineups, n_market))
    
    if USING_RAPIDFUZZ:
        # rapidfuzz can process in bulk - much faster
        for i, lineup_name in enumerate(tqdm(lineup_names, desc="Computing similarities")):
            # Use rapidfuzz's cdist for vectorized similarity calculation
            scores = process.cdist(
                [lineup_name], 
                market_names, 
                scorer=fuzz.ratio
            )
            similarity_matrix[i, :] = scores[0] / 100.0  # Convert from 0-100 to 0-1 scale
    else:
        # difflib - slower but works without additional dependencies
        for i, lineup_name in enumerate(tqdm(lineup_names, desc="Computing similarities")):
            for j, market_name in enumerate(market_names):
                similarity_matrix[i, j] = difflib.SequenceMatcher(
                    None, lineup_name, market_name
                ).ratio()
    
    return similarity_matrix

def find_best_matches(similarity_matrix, lineup_names, market_names, market_values, threshold):
    """
    For each lineup name, find the market name with the highest similarity score.
    Only assign matches if the score is above the threshold.
    """
    matches = {}
    match_details = {}
    
    for i, lineup_name in enumerate(lineup_names):
        # Get the best match for this lineup name
        best_idx = np.argmax(similarity_matrix[i, :])
        best_score = similarity_matrix[i, best_idx]
        
        if best_score >= threshold:
            best_market_name = market_names[best_idx]
            matches[lineup_name] = market_values[best_market_name]
            match_details[lineup_name] = (best_market_name, best_score)
        else:
            matches[lineup_name] = np.nan
            match_details[lineup_name] = (None, best_score)
    
    return matches, match_details

def process_lineup_values():
    # FIXED: Go up one level from the script directory to reach project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)  # Go up from 1_player_stats to project root
    
    # FIXED: Paths now point to project-level directories
    lineups_path = os.path.join(project_dir, 'data', 'lineups.csv')
    players_path = os.path.join(project_dir, 'transfermarkt_data', 'players.csv')
    output_path = os.path.join(project_dir, 'data', 'lineups_values.csv')
    cache_path = os.path.join(project_dir, 'data', 'market_value_name_cache.json')

    try:
        df_lineups = pd.read_csv(lineups_path, low_memory=False)
        df_players = pd.read_csv(players_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # Ensure game_id exists in lineups (required for incremental appends).
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

    # Load previous output and keep it untouched; only process unseen game_ids.
    df_existing = None
    processed_ids = set()
    if os.path.exists(output_path):
        df_existing = pd.read_csv(output_path, low_memory=False)
        if 'game_id' not in df_existing.columns:
            if {'Date', 'Home Team', 'Away Team'}.issubset(df_existing.columns):
                game_key_map = (
                    df_lineups[['Date', 'Home Team', 'Away Team', 'game_id']]
                    .drop_duplicates(subset=['Date', 'Home Team', 'Away Team'])
                )
                df_existing = df_existing.merge(
                    game_key_map,
                    on=['Date', 'Home Team', 'Away Team'],
                    how='left'
                )
                print("Backfilled game_id in existing lineups_values.csv using Date/Home/Away.")
            else:
                df_existing['game_id'] = np.nan
        processed_ids = set(df_existing['game_id'].dropna().astype(str))

    df_pending = df_lineups[~df_lineups['game_id'].astype(str).isin(processed_ids)].copy()
    print(f"Existing processed games: {len(processed_ids)}")
    print(f"Pending games for market value lookup: {len(df_pending)}")

    if df_pending.empty:
        if df_existing is not None:
            df_existing.to_csv(output_path, index=False, encoding='utf-8-sig')
            print("No pending games. Kept existing lineups_values.csv (with game_id column).")
        else:
            print("No pending games and no existing output. Nothing to process.")
        return

    # Verify that all hardcoded columns exist in the dataframe
    missing_cols = [col for col in PLAYER_COLS if col not in df_lineups.columns]
    if missing_cols:
        print(f"Warning: The following expected columns are missing from lineups.csv:")
        for col in missing_cols:
            print(f"  - {col}")
    
    # Filter to only columns that actually exist
    available_player_cols = [col for col in PLAYER_COLS if col in df_lineups.columns]
    print(f"Using {len(available_player_cols)} player columns for name extraction")

    # Sort players by market value (highest first) for deduplication
    df_players = df_players.sort_values(by='market_value_in_eur', ascending=False)
    
    # Create market value lookup
    print("Preparing market value data...")
    market_names = []
    market_values = {}
    
    # Add full names (prioritized)
    for _, row in df_players.dropna(subset=['name']).drop_duplicates('name').iterrows():
        name = row['name']
        value = row['market_value_in_eur']
        if name and name not in market_values:
            market_names.append(name)
            market_values[name] = value
    
    # Add last names (as fallback, only if not already present)
    for _, row in df_players.dropna(subset=['last_name']).drop_duplicates('last_name').iterrows():
        name = row['last_name']
        value = row['market_value_in_eur']
        if name and name not in market_values:
            market_names.append(name)
            market_values[name] = value
    
    print(f"Market value names: {len(market_names)}")
    
    # Extract unique lineup names from player columns only
    print("Extracting unique player names from pending lineups...")
    unique_lineup_names = set()
    
    for col in available_player_cols:
        for name in df_pending[col].dropna():
            name_str = str(name).strip()
            if name_str and name_str.lower() != 'nan':
                # Apply manual mapping
                mapped_name = MANUAL_NAME_MAP.get(name_str, name_str)
                if mapped_name:  # Skip empty mappings
                    unique_lineup_names.add(mapped_name)
    
    unique_lineup_names = list(unique_lineup_names)
    print(f"Unique lineup player names: {len(unique_lineup_names)}")

    # Load cached name->value matches so we only fuzzy-match unseen names.
    name_cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                raw_cache = json.load(f)
            name_cache = {str(k): v for k, v in raw_cache.items()}
            print(f"Loaded market value cache entries: {len(name_cache)}")
        except Exception as e:
            print(f"Warning: Failed to load market value cache ({e}). Rebuilding cache this run.")
            name_cache = {}

    if len(unique_lineup_names) == 0:
        print("No player names found in pending rows. Writing empty-value rows for pending games.")
        available_position_groups = {}
        for position, cols in POSITION_GROUPS.items():
            available_cols = [col for col in cols if col in df_lineups.columns]
            if available_cols:
                available_position_groups[position] = available_cols

        results = []
        for _, row in df_pending.iterrows():
            match_summary = {
                'game_id': row.get('game_id', ''),
                'Competition': row.get('Competition', ''),
                'Date': row.get('Date', ''),
                'Home Team': row.get('Home Team', ''),
                'Away Team': row.get('Away Team', '')
            }
            for position in available_position_groups:
                match_summary[f'{position}_Avg_Value'] = ""
            results.append(match_summary)

        df_results_new = pd.DataFrame(results)
        if df_existing is not None:
            df_results = pd.concat([df_existing, df_results_new], ignore_index=True)
        else:
            df_results = df_results_new

        if 'game_id' in df_results.columns:
            df_results = df_results.drop_duplicates(subset=['game_id'], keep='first')
        if 'Date' in df_results.columns:
            df_results = df_results.sort_values('Date').reset_index(drop=True)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"New games processed: {len(df_results_new)}")
        print(f"Total games in output: {len(df_results)}")
        print(f"Saved summary to: {output_path}")
        return
    
    # Show a sample to verify we're getting real names
    if unique_lineup_names:
        print(f"\nSample of extracted names:")
        for name in unique_lineup_names[:10]:
            print(f"  - {name}")
    
    # Start from cache and only fuzzy-match names not seen before.
    best_matches = {}
    best_details = {}
    names_to_match = []
    for name in unique_lineup_names:
        cached = name_cache.get(name)
        if cached is None:
            names_to_match.append(name)
            continue
        value = cached.get('value')
        if value is None:
            best_matches[name] = np.nan
        else:
            best_matches[name] = float(value)
        best_details[name] = (cached.get('matched_name'), float(cached.get('score', 0.0)))

    print(f"Cached name hits: {len(unique_lineup_names) - len(names_to_match)}")
    print(f"New names requiring fuzzy matching: {len(names_to_match)}")

    current_threshold = SIMILARITY_THRESHOLD
    iteration = 0
    if names_to_match:
        # PHASE 1: Calculate similarity matrix only for new names.
        print("\n--- Phase 1: Computing similarity matrix (new names only) ---")
        similarity_matrix = calculate_similarity_matrix(names_to_match, market_names)

        # PHASE 2: Find best matches with iterative threshold for new names.
        print("\n--- Phase 2: Finding best matches (new names only) ---")
        current_threshold = SIMILARITY_THRESHOLD
        iteration = 1
        previous_missing = len(names_to_match)

        while current_threshold >= MIN_THRESHOLD:
            print(f"\nIteration {iteration}: Threshold = {current_threshold:.2f}")

            matches, details = find_best_matches(
                similarity_matrix, names_to_match, market_names,
                market_values, current_threshold
            )

            matched_count = sum(1 for v in matches.values() if pd.notna(v))
            missing_count = len(names_to_match) - matched_count
            print(f"  Matched: {matched_count} ({matched_count/len(names_to_match)*100:.1f}%)")
            print(f"  Missing: {missing_count} ({missing_count/len(names_to_match)*100:.1f}%)")

            for name in names_to_match:
                if name not in best_matches or pd.isna(best_matches[name]):
                    if pd.notna(matches[name]):
                        best_matches[name] = matches[name]
                        best_details[name] = details[name]

            improvement = previous_missing - missing_count
            improvement_percent = (improvement / previous_missing * 100) if previous_missing > 0 else 0
            print(f"  New matches: {improvement} ({improvement_percent:.1f}% improvement)")

            if missing_count == 0:
                print("\n✓ All new players matched!")
                break

            if improvement_percent < MIN_IMPROVEMENT_PERCENT:
                print(f"\n✓ Convergence reached: Improvement less than {MIN_IMPROVEMENT_PERCENT}%")
                break

            if current_threshold <= MIN_THRESHOLD:
                print(f"\n✓ Reached minimum threshold: {MIN_THRESHOLD}")
                break

            previous_missing = missing_count
            current_threshold = round(current_threshold - THRESHOLD_STEP, 2)
            iteration += 1

    for name in unique_lineup_names:
        if name not in best_matches:
            best_matches[name] = np.nan
            best_details[name] = (None, 0.0)
    
    # Final statistics
    final_matched = sum(1 for v in best_matches.values() if pd.notna(v))
    final_missing = len(unique_lineup_names) - final_matched
    
    print(f"\n{'='*50}")
    print(f"FINAL MATCHING RESULTS")
    print(f"{'='*50}")
    print(f"  Total unique players: {len(unique_lineup_names)}")
    print(f"  Successfully matched: {final_matched} ({final_matched/len(unique_lineup_names)*100:.1f}%)")
    print(f"  Missing:              {final_missing} ({final_missing/len(unique_lineup_names)*100:.1f}%)")
    print(f"  Lowest threshold used: {current_threshold:.2f}")
    print(f"  Iterations performed:  {iteration}")

    # Persist cache so future runs skip expensive fuzzy matching for seen names.
    for name in unique_lineup_names:
        matched_name, score = best_details.get(name, (None, 0.0))
        val = best_matches.get(name, np.nan)
        name_cache[name] = {
            'value': None if pd.isna(val) else float(val),
            'matched_name': matched_name,
            'score': float(score),
        }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(name_cache, f, ensure_ascii=False, indent=2)
    print(f"Saved market value cache: {cache_path} ({len(name_cache)} names)")
    
    # Show best and worst matches for quality verification
    if final_matched > 0:
        print(f"\nTop 10 best matches (for quality check):")
        matched_with_scores = [(name, best_details[name][0], best_details[name][1]) 
                              for name in unique_lineup_names 
                              if pd.notna(best_matches[name])]
        matched_with_scores.sort(key=lambda x: x[2], reverse=True)
        for name, market_name, score in matched_with_scores[:10]:
            print(f"  {name} → {market_name} (score: {score:.3f})")
    
    if final_missing > 0:
        print(f"\nSample of unmatched players:")
        unmatched = [name for name in unique_lineup_names if pd.isna(best_matches[name])]
        for name in unmatched[:15]:
            best_attempt = best_details[name][0]
            best_score = best_details[name][1]
            print(f"  {name} (best attempt: {best_attempt}, score: {best_score:.3f})")
        if len(unmatched) > 15:
            print(f"  ... and {len(unmatched) - 15} more")
    
    # PHASE 3: Compute position averages
    print("\n--- Phase 3: Computing position averages ---")
    
    # Filter position groups to only available columns
    available_position_groups = {}
    for position, cols in POSITION_GROUPS.items():
        available_cols = [col for col in cols if col in df_lineups.columns]
        if available_cols:
            available_position_groups[position] = available_cols
    
    results = []
    for _, row in tqdm(df_pending.iterrows(), total=len(df_pending), desc="Computing averages"):
        match_summary = {
            'game_id': row.get('game_id', ''),
            'Competition': row.get('Competition', ''),
            'Date': row.get('Date', ''),
            'Home Team': row.get('Home Team', ''),
            'Away Team': row.get('Away Team', '')
        }
        
        for position, cols in available_position_groups.items():
            group_values = []
            
            for col in cols:
                player_name = str(row[col]).strip()
                if player_name and player_name.lower() != 'nan':
                    mapped_name = MANUAL_NAME_MAP.get(player_name, player_name)
                    if mapped_name:
                        val = best_matches.get(mapped_name, np.nan)
                        if pd.notna(val):
                            group_values.append(val)
            
            avg_val = round(np.mean(group_values), 2) if group_values else ""
            match_summary[f'{position}_Avg_Value'] = avg_val
        
        results.append(match_summary)
    
    # Save results (append only new game_ids)
    df_results_new = pd.DataFrame(results)
    if df_existing is not None:
        df_results = pd.concat([df_existing, df_results_new], ignore_index=True)
    else:
        df_results = df_results_new

    if 'game_id' in df_results.columns:
        df_results = df_results.drop_duplicates(subset=['game_id'], keep='first')

    if 'Date' in df_results.columns:
        df_results = df_results.sort_values('Date').reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print("\n--- Processing Complete ---")
    print(f"New games processed: {len(df_results_new)}")
    print(f"Total games in output: {len(df_results)}")
    print(f"Saved summary to: {output_path}")

if __name__ == "__main__":
    process_lineup_values()