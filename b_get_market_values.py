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
    
    # Pre-extract all candidate names to run fuzzy matching against
    all_player_names = list(value_map.keys())

    match_stats = {'exact_or_manual': 0, 'fuzzy': 0, 'missing': 0}
    missing_players = [] 

    def get_player_value(raw_name):
        mapped_name = MANUAL_NAME_MAP.get(raw_name, raw_name)
        
        # 1. Try Exact Match (Against full name or last name)
        if mapped_name in value_map:
            return value_map[mapped_name], 'exact_or_manual'
            
        # 2. Try Exact Match on just the Last Word (e.g., 'N. Schlotterbeck' -> 'Schlotterbeck')
        last_word = mapped_name.split()[-1]
        if last_word in value_map:
            return value_map[last_word], 'exact_or_manual'
            
        # 3. Fuzzy Match (Similarity Score)
        # Check against full name first
        closest_matches = difflib.get_close_matches(mapped_name, all_player_names, n=1, cutoff=SIMILARITY_THRESHOLD)
        if closest_matches:
            return value_map[closest_matches[0]], 'fuzzy'
            
        # 4. Fuzzy Match on just the last word
        closest_last_word = difflib.get_close_matches(last_word, all_player_names, n=1, cutoff=SIMILARITY_THRESHOLD)
        if closest_last_word:
            return value_map[closest_last_word[0]], 'fuzzy'
            
        return np.nan, 'missing'

    def get_group_average(row, prefix, group_name):
        cols = [col for col in df_lineups.columns if col.startswith(f"{prefix}{group_name}")]
        group_values = []
        
        for col in cols:
            player_name = str(row[col]).strip()
            
            if player_name and player_name.lower() != 'nan':
                val, match_type = get_player_value(player_name)
                
                if pd.notna(val):
                    group_values.append(val)
                    match_stats[match_type] += 1
                else:
                    match_stats['missing'] += 1
                    missing_players.append(player_name)
        
        if group_values:
            return round(np.mean(group_values), 2)
        return ""

    results = []
    print("Processing games and matching player names... (This might take a few seconds due to fuzzy matching)")
    
    for _, row in df_lineups.iterrows():
        match_summary = {
            'Date': row.get('Date', ''),
            'Home Team': row.get('Home Team', ''),
            'Away Team': row.get('Away Team', '')
        }
        
        for prefix in ['Home_', 'Away_']:
            for position in ['Goalkeeper', 'Defender', 'Midfielder', 'Attacker']:
                avg_val = get_group_average(row, prefix, position)
                match_summary[f'{prefix}{position}_Avg_Value'] = avg_val
                
        results.append(match_summary)

    # Save logic
    df_results = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    # Print statistics
    print("\n--- Scraping & Calculation Complete ---")
    print(f"Total player values found via exact/manual match: {match_stats['exact_or_manual']}")
    print(f"Total player values found via fuzzy match:        {match_stats['fuzzy']}")
    print(f"Total player values missing:                      {match_stats['missing']}")
    
    if match_stats['missing'] > 0:
        print(f"Sample of remaining missing players: {list(set(missing_players))[:10]}")
    
    print(f"\nSaved summary to: {output_path}")

if __name__ == "__main__":
    process_lineup_values()