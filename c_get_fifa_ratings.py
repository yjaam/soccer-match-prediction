import pandas as pd
import numpy as np
import os
import difflib
import re

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

SIMILARITY_THRESHOLD = 0.8

def format_column_name(col):
    """
    Converts dynamically generated column names into the requested standardized format.
    Example: 'Home_Defender_Avg_Value' -> 'home_market_value_Defenders'
    Example: 'Away_Midfielder_pace_Avg' -> 'away_rating_pace_Midfielders'
    """
    if col in ['Date', 'Home Team', 'Away Team']:
        return col

    parts = col.split('_')
    if len(parts) < 3:
        return col

    team = parts[0].lower() # 'home' or 'away'
    pos = parts[1]

    # Pluralize field players
    if pos == 'Defender': pos = 'Defenders'
    elif pos == 'Midfielder': pos = 'Midfielders'
    elif pos == 'Attacker': pos = 'Attackers'

    if 'Avg_Value' in col:
        return f"{team}_market_value_{pos}"
    elif 'Avg' in col:
        # Extract the stat name which is sitting between Position and 'Avg'
        stat = "_".join(parts[2:-1])
        return f"{team}_rating_{stat}_{pos}"

    return col

def process_fifa_ratings():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lineups_path = os.path.join(script_dir, 'data', 'lineups.csv')
    values_path = os.path.join(script_dir, 'data', 'lineups_values.csv') 
    fifa_path = os.path.join(script_dir, 'fifa_data', 'male_players.csv')
    output_path = os.path.join(script_dir, 'data', 'lineups_values_ratings.csv') 

    try:
        df_lineups = pd.read_csv(lineups_path) 
        df_values = pd.read_csv(values_path)   
        
        fifa_cols = ['short_name', 'overall', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
                     'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 
                     'goalkeeping_positioning', 'goalkeeping_reflexes', 'goalkeeping_speed']
        df_fifa = pd.read_csv(fifa_path, usecols=lambda c: c in fifa_cols)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}\nMake sure lineups.csv, lineups_values.csv, and male_players.csv exist.")
        return

    # Sort players by overall rating descending to prioritize top-flight players
    df_fifa = df_fifa.sort_values(by='overall', ascending=False)
    
    # Build a highly optimized lookup dictionary mapping names to stats
    fifa_map = {}
    
    for _, row in df_fifa.iterrows():
        if pd.isna(row['short_name']):
            continue
            
        orig_name = row['short_name']
        # Remove "X. " or "X. Y. " from the start of the string for exact matching against scraped last names
        clean_name = re.sub(r'^([A-Z]\.\s*)+', '', orig_name)
        
        stats_dict = row.to_dict()
        
        if orig_name not in fifa_map:
            fifa_map[orig_name] = stats_dict
            
        if clean_name not in fifa_map:
            fifa_map[clean_name] = stats_dict

    all_fifa_names = list(fifa_map.keys())

    match_stats = {'exact_or_manual': 0, 'fuzzy': 0, 'missing': 0}
    missing_players = []

    def get_player_stats(raw_name):
        mapped_name = MANUAL_NAME_MAP.get(raw_name, raw_name)
        
        # 1. Exact Match
        if mapped_name in fifa_map:
            return fifa_map[mapped_name], 'exact_or_manual'
            
        # 2. Exact Match on just the Last Word
        last_word = mapped_name.split()[-1]
        if last_word in fifa_map:
            return fifa_map[last_word], 'exact_or_manual'
            
        # 3. Fuzzy Match
        closest_matches = difflib.get_close_matches(mapped_name, all_fifa_names, n=1, cutoff=SIMILARITY_THRESHOLD)
        if closest_matches:
            return fifa_map[closest_matches[0]], 'fuzzy'
            
        # 4. Fuzzy Match on just the last word
        closest_last_word = difflib.get_close_matches(last_word, all_fifa_names, n=1, cutoff=SIMILARITY_THRESHOLD)
        if closest_last_word:
            return fifa_map[closest_last_word[0]], 'fuzzy'
            
        return None, 'missing'

    def get_group_stats(row, prefix, group_name):
        cols = [col for col in df_lineups.columns if col.startswith(f"{prefix}{group_name}")]
        
        if group_name == 'Goalkeeper':
            stat_keys = ['overall', 'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 
                         'goalkeeping_positioning', 'goalkeeping_reflexes', 'goalkeeping_speed']
        else:
            stat_keys = ['overall', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
            
        collected_stats = {key: [] for key in stat_keys}
        
        for col in cols:
            player_name = str(row[col]).strip()
            
            if player_name and player_name.lower() != 'nan':
                stats_dict, match_type = get_player_stats(player_name)
                
                if stats_dict is not None:
                    match_stats[match_type] += 1
                    for key in stat_keys:
                        val = stats_dict.get(key, np.nan)
                        if pd.notna(val):
                            collected_stats[key].append(val)
                else:
                    match_stats['missing'] += 1
                    missing_players.append(player_name)

        result_avgs = {}
        for key in stat_keys:
            col_name = f"{prefix}{group_name}_{key}_Avg"
            if collected_stats[key]:
                result_avgs[col_name] = round(np.mean(collected_stats[key]), 2)
            else:
                result_avgs[col_name] = ""
                
        return result_avgs

    fifa_results = []
    print("Processing games and matching FIFA stats... (This might take a few seconds due to fuzzy matching)")
    
    for _, row in df_lineups.iterrows():
        match_summary = {
            'Date': row.get('Date', ''),
            'Home Team': row.get('Home Team', ''),
            'Away Team': row.get('Away Team', '')
        }
        
        for prefix in ['Home_', 'Away_']:
            for position in ['Goalkeeper', 'Defender', 'Midfielder', 'Attacker']:
                group_averages = get_group_stats(row, prefix, position)
                match_summary.update(group_averages)
                
        fifa_results.append(match_summary)

    df_fifa_results = pd.DataFrame(fifa_results)
    
    # Merge the new FIFA columns with the existing market values columns
    df_combined = pd.merge(df_values, df_fifa_results, on=['Date', 'Home Team', 'Away Team'], how='left')

    # Apply the column renaming logic
    df_combined = df_combined.rename(columns=lambda c: format_column_name(c))

    # Save logic
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_combined.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print("\n--- FIFA Ratings Calculation Complete ---")
    print(f"Total player ratings found via exact/manual match: {match_stats['exact_or_manual']}")
    print(f"Total player ratings found via fuzzy match:        {match_stats['fuzzy']}")
    print(f"Total player ratings missing:                      {match_stats['missing']}")
    
    if match_stats['missing'] > 0:
        print(f"Sample of missing players: {list(set(missing_players))[:10]}")
    
    print(f"\nSuccessfully merged market values and FIFA ratings!")
    print(f"Saved combined summary to: {output_path}")

if __name__ == "__main__":
    process_fifa_ratings()