import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from datetime import datetime

def scrape_bundesliga_lineups(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')

    # Try to find the publishing date of the article
    # Using LD+JSON metadata which is more reliable than the <time> tag
    import json
    publish_date = None
    
    ld_json_tags = soup.find_all('script', type='application/ld+json')
    for tag in ld_json_tags:
        try:
            data = json.loads(tag.string)
            if isinstance(data, dict) and 'datePublished' in data:
                # Format is usually YYYY-MM-DDTHH:MM:SSZ
                publish_date = data['datePublished'].split('T')[0]
                break
        except (json.JSONDecodeError, TypeError):
            continue

    if not publish_date:
        # Fallback to <time> tag
        date_tag = soup.find('time')
        if date_tag and date_tag.has_attr('datetime'):
            publish_date = date_tag['datetime'].split('T')[0]
        else:
            # Final fallback to today's date
            publish_date = datetime.today().strftime('%Y-%m-%d')
    
    match_data = []
    paragraphs = soup.find_all('p')
    
    current_match = {}
    match_date = publish_date
    expected_team_type = 'Home'  # Start with expecting Home team
    found_home_lineup = False
    found_away_lineup = False
    
    for p in paragraphs:
        text = p.get_text(strip=True)
        if not text:
            continue
            
        # 1. Identify Match Title (e.g., "Bayer 04 Leverkusen - FC Bayern München")
        has_strong = p.find('strong') is not None
        has_a = p.find('a') is not None
        
        # Check if it has a title format. Exclude it if it looks like a lineup row (contains a colon early on)
        is_lineup_lookalike = ':' in text and len(text.split(':', 1)[0]) < 20
        
        if has_strong and has_a and ' - ' in text and not is_lineup_lookalike:
            # If we already built a match, append it to our final list before starting a new one
            if current_match and found_home_lineup and found_away_lineup:
                match_data.append(current_match)
            
            teams = text.split(' - ')
            if len(teams) >= 2:
                current_match = {
                    'Date': match_date,
                    'Home Team': teams[0].strip(),
                    'Away Team': teams[1].strip()
                }
                # Reset tracking flags for new match
                expected_team_type = 'Home'
                found_home_lineup = False
                found_away_lineup = False
                
        # 2. Identify Lineup Rows
        # Check if there's a colon with a short abbreviation, and " - " in the text
        elif current_match and ':' in text and ' - ' in text:
            prefix, lineup_str = text.split(':', 1)
            
            # The prefix (Team Abbreviation) should be short (e.g., "B04", "FCB", "St. Pauli")
            if len(prefix.strip()) < 20:
                formation_parts = [part.strip() for part in lineup_str.split(' - ')]
                
                # A valid formation must have at least Goalkeeper, Defense, and Attack
                if len(formation_parts) >= 3:
                    goalkeeper = formation_parts[0]
                    defenders = [d.strip() for d in formation_parts[1].split(',')]
                    attackers = [a.strip() for a in formation_parts[-1].split(',')]
                    
                    midfielders = []
                    for mid_part in formation_parts[2:-1]:
                        midfielders.extend([m.strip() for m in mid_part.split(',')])
                    
                    # Determine which team this lineup belongs to by checking the prefix against team names
                    home_team_abbr = get_team_abbreviation(current_match['Home Team'])
                    away_team_abbr = get_team_abbreviation(current_match['Away Team'])
                    
                    # Check if prefix matches either team's abbreviation
                    prefix_clean = prefix.strip().lower()
                    
                    if prefix_clean == home_team_abbr.lower() and not found_home_lineup:
                        team_prefix = 'Home_'
                        found_home_lineup = True
                        expected_team_type = 'Away'  # Next expected is Away
                    elif prefix_clean == away_team_abbr.lower() and not found_away_lineup:
                        team_prefix = 'Away_'
                        found_away_lineup = True
                        expected_team_type = 'Home'  # Next expected is Home
                    elif not found_home_lineup and expected_team_type == 'Home':
                        # If no match found but we're expecting Home, assign to Home
                        team_prefix = 'Home_'
                        found_home_lineup = True
                        expected_team_type = 'Away'
                    elif not found_away_lineup and expected_team_type == 'Away':
                        # If no match found but we're expecting Away, assign to Away
                        team_prefix = 'Away_'
                        found_away_lineup = True
                        expected_team_type = 'Home'
                    else:
                        # Skip if we already have both lineups
                        continue
                    
                    current_match[f'{team_prefix}Goalkeeper'] = goalkeeper
                    
                    for i, defender in enumerate(defenders, 1):
                        current_match[f'{team_prefix}Defender_{i}'] = defender
                        
                    for i, midfielder in enumerate(midfielders, 1):
                        current_match[f'{team_prefix}Midfielder_{i}'] = midfielder
                        
                    for i, attacker in enumerate(attackers, 1):
                        current_match[f'{team_prefix}Attacker_{i}'] = attacker

    # Append the very last match in the loop
    if current_match and found_home_lineup and found_away_lineup:
        match_data.append(current_match)

    # Create a DataFrame
    df = pd.DataFrame(match_data)
    
    # Reorder columns: Date, Home/Away teams first
    if not df.empty:
        base_cols = ['Date', 'Home Team', 'Away Team']
        player_cols = [c for c in df.columns if c not in base_cols]
        df = df[base_cols + player_cols]
    
    df = df.fillna('')
    return df

def get_team_abbreviation(team_name):
    """Extract common abbreviation from team name or return first few words"""
    # Common Bundesliga abbreviations
    abbreviations = {
        'Bayer 04 Leverkusen': 'B04',
        'FC Bayern München': 'FCB',
        'Borussia Dortmund': 'BVB',
        'RB Leipzig': 'RBL',
        'Eintracht Frankfurt': 'SGE',
        'VfB Stuttgart': 'VfB',
        'Borussia Mönchengladbach': 'BMG',
        'VfL Wolfsburg': 'WOB',
        '1. FC Union Berlin': 'FCU',
        'SC Freiburg': 'SCF',
        '1. FSV Mainz 05': 'M05',
        'FC Augsburg': 'FCA',
        'TSG 1899 Hoffenheim': 'TSG',
        'SV Werder Bremen': 'SVW',
        '1. FC Köln': 'KOE',
        'VfL Bochum 1848': 'BOC',
        'FC St. Pauli': 'STP',
        'Holstein Kiel': 'KIE',
        '1. FC Heidenheim': 'HEI'
    }
    
    # Check if team name has a known abbreviation
    for full_name, abbr in abbreviations.items():
        if full_name.lower() in team_name.lower() or team_name.lower() in full_name.lower():
            return abbr
    
    # If no known abbreviation, try to extract from the name
    # Remove common prefixes and take first 3-4 characters
    cleaned = team_name.replace('1. FC ', '').replace('FC ', '').replace('VfL ', '').replace('VfB ', '')
    words = cleaned.split()
    
    if len(words) >= 2:
        # Take first letter of first two words
        abbr = ''.join([word[0] for word in words[:2]]).upper()
    else:
        # Take first 3 letters
        abbr = cleaned[:3].upper()
    
    return abbr

if __name__ == "__main__":
    url = "https://www.bundesliga.com/de/bundesliga/news/voraussichtliche-aufstellungen-spieltag-verletzungen-sperren-ubersicht-26-24397"
    df_lineups = scrape_bundesliga_lineups(url)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'data', 'lineups.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df_lineups.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    pd.set_option('display.max_columns', None)
    print("Scraping successful! Previewing the first match:\n")
    print(df_lineups.head(1).to_string(index=False))
    print(f"\nSaved {len(df_lineups)} row(s) to: {output_path}")