import pandas as pd
import numpy as np
import os
import difflib

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

# Tweak this to be more strict (> 0.85) or more relaxed (< 0.75)
SIMILARITY_THRESHOLD = 0.8
THRESHOLD_STEP = 0.05
MIN_THRESHOLD = 0.5

def process_lineup_values():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lineups_path = os.path.join(script_dir, 'data', 'lineups.csv')
    players_path = os.path.join(script_dir, 'transfermarkt_data', 'players.csv')
    output_path = os.path.join(script_dir, 'data', 'lineups_values.csv')

    try:
        df_lineups = pd.read_csv(lineups_path)
        df_players = pd.read_csv(players_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # Sort players by market value (highest first) to prioritize top-flight players during deduplication
    df_players = df_players.sort_values(by='market_value_in_eur', ascending=False)
    
    last_name_map = dict(zip(
        df_players.dropna(subset=['last_name']).drop_duplicates('last_name')['last_name'], 
        df_players.dropna(subset=['last_name']).drop_duplicates('last_name')['market_value_in_eur']
    ))
    
    full_name_map = dict(zip(
        df_players.dropna(subset=['name']).drop_duplicates('name')['name'], 
        df_players.dropna(subset=['name']).drop_duplicates('name')['market_value_in_eur']
    ))
    
    value_map = {**last_name_map, **full_name_map}
    all_player_names = list(value_map.keys())

    print("Extracting unique player names from lineups...")
    
    # Extract all unique player names from lineups once
    unique_players = set()
    player_cols = [col for col in df_lineups.columns if col not in ['Competition', 'Date', 'Home Team', 'Away Team']]
    
    for col in player_cols:
        for name in df_lineups[col].dropna():
            name_str = str(name).strip()
            if name_str and name_str.lower() != 'nan':
                unique_players.add(name_str)
    
    print(f"Found {len(unique_players)} unique player names to match")
    print("Matching player names with threshold fallback...")
    
    # Match all unique players once with threshold fallback
    player_matches = {}
    match_stats = {'exact_or_manual': 0, 'fuzzy': 0, 'missing': 0}
    current_threshold = SIMILARITY_THRESHOLD
    
    for i, raw_name in enumerate(unique_players):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(unique_players)} unique players...")
        
        if raw_name in player_matches:
            continue
        
        mapped_name = MANUAL_NAME_MAP.get(raw_name, raw_name)
        matched_value = None
        match_type = 'missing'
        
        # 1. Try Exact Match (Against full name or last name)
        if mapped_name in value_map:
            matched_value = value_map[mapped_name]
            match_type = 'exact_or_manual'
        else:
            # 2. Try Exact Match on just the Last Word
            last_word = mapped_name.split()[-1]
            if last_word in value_map:
                matched_value = value_map[last_word]
                match_type = 'exact_or_manual'
            else:
                # 3. Fuzzy Match with progressive threshold lowering
                threshold = SIMILARITY_THRESHOLD
                while threshold >= MIN_THRESHOLD and matched_value is None:
                    closest_matches = difflib.get_close_matches(mapped_name, all_player_names, n=1, cutoff=threshold)
                    if closest_matches:
                        matched_value = value_map[closest_matches[0]]
                        match_type = 'fuzzy'
                        break
                    
                    # Try fuzzy on last word
                    closest_last_word = difflib.get_close_matches(last_word, all_player_names, n=1, cutoff=threshold)
                    if closest_last_word:
                        matched_value = value_map[closest_last_word[0]]
                        match_type = 'fuzzy'
                        break
                    
                    threshold = round(threshold - THRESHOLD_STEP, 2)
        
        player_matches[raw_name] = matched_value if pd.notna(matched_value) else np.nan
        match_stats[match_type] += 1
    
    print(f"\nMatching complete!")
    print(f"  Exact/manual matches: {match_stats['exact_or_manual']}")
    print(f"  Fuzzy matches:        {match_stats['fuzzy']}")
    print(f"  Missing values:       {match_stats['missing']}")
    
    # Now compute averages in a single pass using cached matches
    print("\nComputing position averages...")
    results = []
    
    for _, row in df_lineups.iterrows():
        match_summary = {
            'Competition': row.get('Competition', ''),
            'Date': row.get('Date', ''),
            'Home Team': row.get('Home Team', ''),
            'Away Team': row.get('Away Team', '')
        }
        
        for prefix in ['Home_', 'Away_']:
            for position in ['Goalkeeper', 'Defender', 'Midfielder', 'Attacker']:
                cols = [col for col in df_lineups.columns if col.startswith(f"{prefix}{position}")]
                group_values = []
                
                for col in cols:
                    player_name = str(row[col]).strip()
                    if player_name and player_name.lower() != 'nan':
                        val = player_matches.get(player_name, np.nan)
                        if pd.notna(val):
                            group_values.append(val)
                
                avg_val = round(np.mean(group_values), 2) if group_values else ""
                match_summary[f'{prefix}{position}_Avg_Value'] = avg_val
        
        results.append(match_summary)
    
    # Save results
    df_results = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    # Print statistics
    print("\n--- Processing Complete ---")
    print(f"Unique players matched: {len(player_matches)}")
    print(f"Games processed: {len(df_results)}")
    print(f"Saved summary to: {output_path}")

if __name__ == "__main__":
    process_lineup_values()