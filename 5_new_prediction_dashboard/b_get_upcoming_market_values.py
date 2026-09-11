#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compute market value aggregates for UPCOMING lineups (dashboard/prediction use).

Differs from the historical script:
- Reads fresh scrapes from ./upcoming_lineups/upcoming_lineups_latest.csv
- Writes to ./prediction_data/upcoming_lineups_values.csv
- Never touches data/lineups.csv or data/lineups_values.csv (training artifacts)
- Reuses the exact same matching algorithm + cache
"""

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

SIMILARITY_THRESHOLD = 0.8
THRESHOLD_STEP = 0.05
MIN_THRESHOLD = 0.5
MIN_IMPROVEMENT_PERCENT = 1.0


def calculate_similarity_matrix(lineup_names, market_names):
    n_lineups = len(lineup_names)
    n_market = len(market_names)

    print(f"Creating similarity matrix: {n_lineups} lineup names × {n_market} market names")
    print(f"Total comparisons: {n_lineups * n_market:,}")

    similarity_matrix = np.zeros((n_lineups, n_market))

    if USING_RAPIDFUZZ:
        for i, lineup_name in enumerate(tqdm(lineup_names, desc="Computing similarities")):
            scores = process.cdist([lineup_name], market_names, scorer=fuzz.ratio)
            similarity_matrix[i, :] = scores[0] / 100.0
    else:
        for i, lineup_name in enumerate(tqdm(lineup_names, desc="Computing similarities")):
            for j, market_name in enumerate(market_names):
                similarity_matrix[i, j] = difflib.SequenceMatcher(
                    None, lineup_name, market_name
                ).ratio()

    return similarity_matrix


def find_best_matches(similarity_matrix, lineup_names, market_names, market_values, threshold):
    matches = {}
    match_details = {}
    for i, lineup_name in enumerate(lineup_names):
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


def process_upcoming_lineup_values():
    # ------------------------------------------------------------------
    # Paths — everything is LOCAL to the dashboard folder
    # ------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)  # project root (for reading players.csv)

    upcoming_dir = os.path.join(script_dir, "upcoming_lineups")
    prediction_data_dir = os.path.join(script_dir, "prediction_data")
    os.makedirs(prediction_data_dir, exist_ok=True)

    # Input: the latest upcoming lineups scraped by the sibling script
    lineups_path = os.path.join(upcoming_dir, "upcoming_lineups_latest.csv")

    # Reference data: Transfermarkt players (read-only, lives in project)
    players_path = os.path.join(project_dir, "transfermarkt_data", "players.csv")

    # Outputs: everything the dashboard needs, kept locally
    output_path = os.path.join(prediction_data_dir, "upcoming_lineups_values.csv")
    cache_path = os.path.join(prediction_data_dir, "market_value_name_cache.json")

    # ------------------------------------------------------------------
    # Load upcoming lineups
    # ------------------------------------------------------------------
    if not os.path.exists(lineups_path):
        print(f"Error: upcoming lineups file not found at {lineups_path}")
        print("Run a_scrape_upcoming_lineups.py first.")
        return

    try:
        df_lineups = pd.read_csv(lineups_path, low_memory=False)
        df_players = pd.read_csv(players_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # Ensure game_id exists
    if 'game_id' not in df_lineups.columns:
        if {'Date', 'Home Team', 'Away Team'}.issubset(df_lineups.columns):
            df_lineups['game_id'] = (
                df_lineups['Date'].astype(str).str.replace('-', '', regex=False)
                + '_' + df_lineups['Home Team'].astype(str)
                + '_' + df_lineups['Away Team'].astype(str)
            )
            print("Derived game_id from Date/Home/Away for upcoming lineups.")
        else:
            raise KeyError("Upcoming lineups must contain game_id or Date/Home Team/Away Team.")

    print(f"Loaded {len(df_lineups)} upcoming matches.")

    # ------------------------------------------------------------------
    # Incremental behavior: skip game_ids already in output
    # ------------------------------------------------------------------
    df_existing = None
    processed_ids = set()
    if os.path.exists(output_path):
        df_existing = pd.read_csv(output_path, low_memory=False)
        if 'game_id' in df_existing.columns:
            # Keep only rows whose game_id still exists upstream
            upstream_ids = set(df_lineups['game_id'].astype(str))
            before = len(df_existing)
            df_existing = df_existing[df_existing['game_id'].astype(str).isin(upstream_ids)].copy()
            pruned = before - len(df_existing)
            if pruned > 0:
                print(f"Pruned {pruned} stale rows from existing output "
                      f"(no longer present in upstream lineups).")
            processed_ids = set(df_existing['game_id'].dropna().astype(str))

    df_pending = df_lineups[~df_lineups['game_id'].astype(str).isin(processed_ids)].copy()
    print(f"Existing processed games: {len(processed_ids)}")
    print(f"Pending games for market value lookup: {len(df_pending)}")

    if df_pending.empty:
        if df_existing is not None:
            df_existing.to_csv(output_path, index=False, encoding='utf-8-sig')
            print("No pending games. Kept existing upcoming_lineups_values.csv.")
        else:
            print("No pending games and no existing output. Nothing to process.")
        return

    # ------------------------------------------------------------------
    # Prepare player/market value reference
    # ------------------------------------------------------------------
    missing_cols = [col for col in PLAYER_COLS if col not in df_lineups.columns]
    if missing_cols:
        print(f"Warning: The following expected columns are missing from upcoming lineups:")
        for col in missing_cols:
            print(f"  - {col}")

    available_player_cols = [col for col in PLAYER_COLS if col in df_lineups.columns]
    print(f"Using {len(available_player_cols)} player columns for name extraction")

    df_players = df_players.sort_values(by='market_value_in_eur', ascending=False)

    market_names = []
    market_values = {}

    for _, row in df_players.dropna(subset=['name']).drop_duplicates('name').iterrows():
        name = row['name']
        value = row['market_value_in_eur']
        if name and name not in market_values:
            market_names.append(name)
            market_values[name] = value

    for _, row in df_players.dropna(subset=['last_name']).drop_duplicates('last_name').iterrows():
        name = row['last_name']
        value = row['market_value_in_eur']
        if name and name not in market_values:
            market_names.append(name)
            market_values[name] = value

    print(f"Market value names: {len(market_names)}")

    # ------------------------------------------------------------------
    # Extract unique player names from pending matches
    # ------------------------------------------------------------------
    unique_lineup_names = set()
    for col in available_player_cols:
        for name in df_pending[col].dropna():
            name_str = str(name).strip()
            if name_str and name_str.lower() != 'nan':
                mapped_name = MANUAL_NAME_MAP.get(name_str, name_str)
                if mapped_name:
                    unique_lineup_names.add(mapped_name)

    unique_lineup_names = list(unique_lineup_names)
    print(f"Unique lineup player names: {len(unique_lineup_names)}")

    # ------------------------------------------------------------------
    # Load local name cache
    # ------------------------------------------------------------------
    name_cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                raw_cache = json.load(f)
            name_cache = {str(k): v for k, v in raw_cache.items()}
            print(f"Loaded market value cache entries: {len(name_cache)}")
        except Exception as e:
            print(f"Warning: Failed to load market value cache ({e}). Rebuilding.")
            name_cache = {}

    # ------------------------------------------------------------------
    # Handle empty case (no player names found)
    # ------------------------------------------------------------------
    if len(unique_lineup_names) == 0:
        print("No player names found in pending rows. Writing empty-value rows.")
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
        df_results = pd.concat([df_existing, df_results_new], ignore_index=True) if df_existing is not None else df_results_new
        if 'game_id' in df_results.columns:
            df_results = df_results.drop_duplicates(subset=['game_id'], keep='first')
        if 'Date' in df_results.columns:
            df_results = df_results.sort_values('Date').reset_index(drop=True)

        df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"New games processed: {len(df_results_new)}")
        print(f"Total games in output: {len(df_results)}")
        print(f"Saved to: {output_path}")
        return

    # ------------------------------------------------------------------
    # Cache hits + fuzzy matching for unseen names
    # ------------------------------------------------------------------
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
        print("\n--- Phase 1: Computing similarity matrix (new names only) ---")
        similarity_matrix = calculate_similarity_matrix(names_to_match, market_names)

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
                print(f"\n✓ Convergence reached.")
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

    final_matched = sum(1 for v in best_matches.values() if pd.notna(v))
    final_missing = len(unique_lineup_names) - final_matched

    print(f"\n{'='*50}")
    print(f"FINAL MATCHING RESULTS (upcoming matches)")
    print(f"{'='*50}")
    print(f"  Total unique players: {len(unique_lineup_names)}")
    print(f"  Successfully matched: {final_matched} ({final_matched/len(unique_lineup_names)*100:.1f}%)")
    print(f"  Missing:              {final_missing} ({final_missing/len(unique_lineup_names)*100:.1f}%)")

    # Update cache
    for name in unique_lineup_names:
        matched_name, score = best_details.get(name, (None, 0.0))
        val = best_matches.get(name, np.nan)
        name_cache[name] = {
            'value': None if pd.isna(val) else float(val),
            'matched_name': matched_name,
            'score': float(score),
        }
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(name_cache, f, ensure_ascii=False, indent=2)
    print(f"Saved market value cache: {cache_path} ({len(name_cache)} names)")

    # ------------------------------------------------------------------
    # Phase 3: Compute position averages per pending match
    # ------------------------------------------------------------------
    print("\n--- Phase 3: Computing position averages ---")

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

    # ------------------------------------------------------------------
    # Save incremental output
    # ------------------------------------------------------------------
    df_results_new = pd.DataFrame(results)
    df_results = pd.concat([df_existing, df_results_new], ignore_index=True) if df_existing is not None else df_results_new

    if 'game_id' in df_results.columns:
        df_results = df_results.drop_duplicates(subset=['game_id'], keep='first')
    if 'Date' in df_results.columns:
        df_results = df_results.sort_values('Date').reset_index(drop=True)

    df_results.to_csv(output_path, index=False, encoding='utf-8-sig')

    print("\n--- Processing Complete ---")
    print(f"New games processed: {len(df_results_new)}")
    print(f"Total games in output: {len(df_results)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    process_upcoming_lineup_values()