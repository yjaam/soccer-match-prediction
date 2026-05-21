import pandas as pd
import numpy as np
import os
from datetime import datetime

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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lineups_path = os.path.join(script_dir, 'data', 'lineups.csv')
    values_path = os.path.join(script_dir, 'data', 'lineups_values.csv')
    player_ratings_path = os.path.join(script_dir, 'fifa_ratings', 'player_ratings.csv')
    output_path = os.path.join(script_dir, 'data', 'lineups_values_ratings.csv')

    try:
        df_lineups = pd.read_csv(lineups_path)
        df_values = pd.read_csv(values_path)
        df_player_ratings = pd.read_csv(player_ratings_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print(f"Make sure lineups.csv, lineups_values.csv, and fifa_ratings/player_ratings.csv exist.")
        return

    print("Loading player ratings data...")
    # Convert fifa_version to int for easier comparison
    df_player_ratings['fifa_version'] = df_player_ratings['fifa_version'].astype(int)
    
    # Create lookup dictionaries for each FIFA version
    # This will speed up player lookups significantly
    fifa_lookups = {}
    for fifa_ver in sorted(df_player_ratings['fifa_version'].unique()):
        df_ver = df_player_ratings[df_player_ratings['fifa_version'] == fifa_ver]
        # Create a mapping of player name to their ratings
        fifa_lookups[fifa_ver] = {}
        for _, row in df_ver.iterrows():
            player_name = row['short_name']
            fifa_lookups[fifa_ver][player_name] = row.to_dict()
        print(f"  Loaded FIFA {fifa_ver}: {len(fifa_lookups[fifa_ver])} players")

    match_stats = {'exact_or_manual': 0, 'fuzzy': 0, 'missing': 0}
    missing_players = []

    def get_player_stats_for_version(player_name, fifa_version):
        """Get player stats from a specific FIFA version."""
        if fifa_version not in fifa_lookups:
            return None, 'missing'
        
        # Apply manual mapping
        mapped_name = MANUAL_NAME_MAP.get(player_name, player_name)
        
        # Try exact match
        if mapped_name in fifa_lookups[fifa_version]:
            return fifa_lookups[fifa_version][mapped_name], 'exact_or_manual'
        
        # Try last word match
        last_word = mapped_name.split()[-1]
        for player_key, stats in fifa_lookups[fifa_version].items():
            if player_key.endswith(last_word):
                return stats, 'exact_or_manual'
        
        return None, 'missing'

    def get_group_stats(row, prefix, group_name, fifa_version):
        """
        Calculate average stats for a group of players for a specific position and team.
        Uses the specified FIFA version.
        """
        # Find columns matching this group
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
                stats_dict, match_type = get_player_stats_for_version(player_name, fifa_version)
                
                if stats_dict is not None:
                    match_stats[match_type] += 1
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
        for key in stat_keys:
            col_name = f"{prefix}{group_name}_{key}_Avg"
            if collected_stats[key]:
                result_avgs[col_name] = round(np.mean(collected_stats[key]), 2)
            else:
                result_avgs[col_name] = np.nan
        
        return result_avgs

    # Process each match
    print("\nProcessing matches and calculating averaged FIFA ratings...")
    fifa_results = []
    
    for idx, row in df_lineups.iterrows():
        match_date = row.get('Date', '')
        fifa_version = get_fifa_version_for_date(match_date)
        
        match_summary = {
            'Date': row.get('Date', ''),
            'Home Team': row.get('Home Team', ''),
            'Away Team': row.get('Away Team', ''),
            'FIFA_Version_Used': fifa_version
        }
        
        # Process home and away teams
        for prefix in ['Home_', 'Away_']:
            for position in ['Goalkeeper', 'Defender', 'Midfielder', 'Attacker']:
                group_averages = get_group_stats(row, prefix, position, fifa_version)
                match_summary.update(group_averages)
        
        fifa_results.append(match_summary)
        
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(df_lineups)} matches...")

    df_fifa_results = pd.DataFrame(fifa_results)
    
    # Merge FIFA ratings with market values
    print("Merging FIFA ratings with market values...")
    df_combined = pd.merge(df_values, df_fifa_results, on=['Date', 'Home Team', 'Away Team'], how='left')

    # Save the result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_combined.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print("\n--- FIFA Ratings Processing Complete ---")
    print(f"Total player ratings found via exact/manual match: {match_stats['exact_or_manual']}")
    print(f"Total player ratings missing:                      {match_stats['missing']}")
    
    if match_stats['missing'] > 0:
        unique_missing = list(set(missing_players))
        print(f"Sample of missing players: {unique_missing[:10]}")
    
    print(f"\nProcessed {len(df_lineups)} matches")
    print(f"Output columns: {len(df_combined.columns)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    process_fifa_ratings()