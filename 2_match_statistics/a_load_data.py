import os
os.environ['BOTASAURUS_HEADLESS'] = 'True'

import ScraperFC as sfc
from bs4 import BeautifulSoup
import pandas as pd
import json
import re
import time
from pathlib import Path
from datetime import datetime

fb = sfc.FBref()

# ============================================================================
# CONFIGURATION
# ============================================================================

TEST_MODE = False  # Set to True for testing

if TEST_MODE:
    seasons = ['2023-2024']
else:
    seasons = ['2014-2015', '2015-2016', '2016-2017', '2017-2018', '2018-2019', '2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025', '2025-2026']

big_5_leagues = ['England Premier League', 'France Ligue 1', 'Germany Bundesliga', 'Italy Serie A', 'Spain La Liga']

if TEST_MODE:
    big_5_leagues = ['England Premier League']

# Setup directories
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
CHECKPOINT_DIR = PROJECT_DIR / "scraperfc_data" / "checkpoints"
MATCHES_JSON_DIR = PROJECT_DIR / "scraperfc_data" / "matches_json"
CHECKPOINT_FILE = CHECKPOINT_DIR / "match_scraping_checkpoint.json"

if TEST_MODE:
    CHECKPOINT_FILE = CHECKPOINT_DIR / "match_scraping_checkpoint_TEST.json"
    MATCHES_JSON_DIR = PROJECT_DIR / "scraperfc_data" / "matches_json_test"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
MATCHES_JSON_DIR.mkdir(parents=True, exist_ok=True)

# Final output path
output_name = "big5_matches_TEST.csv" if TEST_MODE else "big5_matches.csv"
OUTPUT_PATH = PROJECT_DIR / "data" / output_name
os.makedirs(OUTPUT_PATH.parent, exist_ok=True)

# Browser restart settings
MATCHES_BEFORE_RESTART = 50

# ============================================================================
# CHECKPOINT FUNCTIONS
# ============================================================================

def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"completed_urls": [], "failed_urls": [], "last_updated": None}

def save_checkpoint(checkpoint_data):
    checkpoint_data["last_updated"] = datetime.now().isoformat()
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)

# ============================================================================
# VALUE CLEANING
# ============================================================================

def clean_value(val):
    """Clean a scraped value: remove commas from numbers, handle edge cases."""
    if val is None:
        return None
    val = str(val).strip()
    if val == '' or val.lower() == 'nan':
        return None
    # Remove thousand separators (e.g., 1,320 -> 1320)
    if ',' in val:
        cleaned = val.replace(',', '')
        # Check if it's a number
        try:
            float(cleaned)
            return cleaned
        except ValueError:
            pass
    return val

# ============================================================================
# EXTRACTION FUNCTION
# ============================================================================

def extract_match_data(url: str, league: str, season: str) -> dict:
    """
    Extract all match-level data from a single FBref match page.
    Returns a dictionary with one row of aggregated features.
    """
    soup = fb._get_soup(url)
    
    row = {}
    row['league'] = league
    row['season'] = season
    row['url'] = url
    
    # Extract match ID from URL
    match_id = url.split('/')[-2]
    row['match_id'] = match_id
    
    # Find team IDs
    team_ids = []
    for div in soup.find_all('div', id=re.compile(r'all_player_stats_')):
        team_ids.append(div['id'].replace('all_player_stats_', ''))
    
    home_id = team_ids[0] if len(team_ids) > 0 else None
    away_id = team_ids[1] if len(team_ids) > 1 else None
    
    # Get team names
    team_stats_tbody = soup.select_one('#team_stats > table > tbody')
    if team_stats_tbody:
        first_row = team_stats_tbody.find_all('tr')[0]
        team_cells = first_row.find_all('th')
        row['home_team'] = team_cells[0].get_text(strip=True) if len(team_cells) > 0 else None
        row['away_team'] = team_cells[1].get_text(strip=True) if len(team_cells) > 1 else None
    
    # ============================================================
    # 1. Match stats from #team_stats
    # ============================================================
    if team_stats_tbody:
        rows = team_stats_tbody.find_all('tr')
        i = 1
        while i < len(rows):
            header = rows[i].find('th')
            if header and header.get('colspan') == '2':
                stat_name = header.get_text(strip=True).lower().replace(' ', '_')
                data_row = rows[i + 1]
                cells = data_row.find_all('td')
                
                if stat_name == 'cards':
                    home_val = str(len(cells[0].find_all('span', class_='yellow_card')))
                    away_val = str(len(cells[1].find_all('span', class_='yellow_card')))
                else:
                    home_text = cells[0].get_text(strip=True)
                    away_text = cells[1].get_text(strip=True)
                    home_pct = re.findall(r'(\d+)%', home_text)
                    away_pct = re.findall(r'(\d+)%', away_text)
                    home_val = home_pct[0] if home_pct else None
                    away_val = away_pct[0] if away_pct else None
                
                row[f'Home_{stat_name}'] = clean_value(home_val)
                row[f'Away_{stat_name}'] = clean_value(away_val)
                i += 2
            else:
                i += 1
    
    # ============================================================
    # 2. Match extras from #team_stats_extra
    # ============================================================
    team_stats_extra = soup.select_one('#team_stats_extra')
    if team_stats_extra:
        columns = team_stats_extra.find_all('div', recursive=False)
        for column in columns:
            divs = column.find_all('div')
            for i in range(3, len(divs), 3):
                if i + 2 < len(divs):
                    home_val = divs[i].get_text(strip=True)
                    stat_name = divs[i + 1].get_text(strip=True).lower().replace(' ', '_')
                    away_val = divs[i + 2].get_text(strip=True)
                    row[f'Home_{stat_name}'] = clean_value(home_val)
                    row[f'Away_{stat_name}'] = clean_value(away_val)
    
    # ============================================================
    # 3. Player summary totals from <tfoot> (skip non-numeric ID cols)
    # ============================================================
    skip_stats = {'shirtnumber', 'nationality', 'position', 'age', 'player'}
    
    for side, team_id in [('Home', home_id), ('Away', away_id)]:
        if team_id:
            tfoot = soup.select_one(f'#stats_{team_id}_summary tfoot tr')
            if tfoot:
                for td in tfoot.find_all(['th', 'td']):
                    stat = td.get('data-stat', '')
                    if stat and stat not in skip_stats:
                        val = td.get_text(strip=True)
                        row[f'{side}_{stat}'] = clean_value(val)
    
    # ============================================================
    # 4. Keeper stats
    # ============================================================
    for side, team_id in [('Home', home_id), ('Away', away_id)]:
        if team_id:
            keeper_table = soup.select_one(f'#keeper_stats_{team_id} tbody')
            if keeper_table:
                for keeper_row in keeper_table.find_all('tr'):
                    for td in keeper_row.find_all(['th', 'td']):
                        stat = td.get('data-stat', '')
                        if stat and stat not in skip_stats:
                            val = td.get_text(strip=True)
                            row[f'{side}_keeper_{stat}'] = clean_value(val)
    
    return row


def extract_match_data_with_retry(url: str, league: str, season: str, max_retries: int = 3) -> dict:
    """Wrapper with retry logic."""
    for attempt in range(max_retries):
        try:
            return extract_match_data(url, league, season)
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 30
                print(f"        Retry {attempt+1}/{max_retries} in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e


# ============================================================================
# MAIN
# ============================================================================

print(f"\n{'='*60}")
print(f"{'TEST MODE' if TEST_MODE else 'PRODUCTION MODE'}")
print(f"{'='*60}")
print(f"Leagues: {big_5_leagues}")
print(f"Seasons: {seasons}")

checkpoint = load_checkpoint()
completed_urls = set(checkpoint["completed_urls"])
failed_urls = set(checkpoint.get("failed_urls", []))
print(f"Already completed: {len(completed_urls)} URLs")
print(f"Previously failed: {len(failed_urls)} URLs")

# ============================================================================
# PHASE 1: Scrape each match, save as individual JSON
# ============================================================================
print("\n" + "="*60)
print("PHASE 1: Scraping matches (saving as JSON files)")
print("="*60)

total_scraped = 0
matches_since_restart = 0

for league in big_5_leagues:
    for season in seasons:
        print(f"\n  Getting URLs for {league} {season}...")
        
        try:
            links = fb.get_match_links(year=season, league=league)
            print(f"  Found {len(links)} match URLs")
            
            if TEST_MODE:
                links = links[:5]
                print(f"  TEST MODE: Limited to {len(links)} matches")
            
            for i, url in enumerate(links):
                match_id = url.split('/')[-2]
                json_path = MATCHES_JSON_DIR / f"{match_id}.json"
                
                # Skip if already completed AND JSON file exists
                if url in completed_urls and json_path.exists():
                    continue
                
                # Restart browser periodically
                if matches_since_restart >= MATCHES_BEFORE_RESTART:
                    print("    🔄 Restarting browser...")
                    try:
                        fb_new = sfc.FBref()
                        fb = fb_new
                    except:
                        pass
                    matches_since_restart = 0
                    time.sleep(5)
                
                print(f"    [{i+1}/{len(links)}] {match_id}: {url.split('/')[-1][:50]}...")
                
                success = False
                for attempt in range(3):
                    try:
                        row = extract_match_data(url, league, season)
                        
                        # Save as individual JSON file (crash-safe!)
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(row, f, ensure_ascii=False)
                        
                        checkpoint["completed_urls"].append(url)
                        save_checkpoint(checkpoint)
                        
                        total_scraped += 1
                        matches_since_restart += 1
                        success = True
                        break
                        
                    except Exception as e:
                        if attempt < 2:
                            wait_time = (attempt + 1) * 30
                            print(f"      Retry {attempt+1}/3 in {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            print(f"      ❌ Failed: {str(e)[:80]}")
                            checkpoint["failed_urls"].append(url)
                            save_checkpoint(checkpoint)
                
                if not success:
                    try:
                        fb_new = sfc.FBref()
                        fb = fb_new
                        matches_since_restart = 0
                    except:
                        pass
                    time.sleep(10)
                    
        except Exception as e:
            print(f"  ❌ Error getting links: {e}")

# ============================================================================
# PHASE 2: Combine all JSON files into one DataFrame
# ============================================================================
print("\n" + "="*60)
print("PHASE 2: Combining all JSON files into CSV")
print("="*60)

json_files = sorted(MATCHES_JSON_DIR.glob("*.json"))
print(f"Found {len(json_files)} JSON files")

if json_files:
    # Load all JSONs - each is one row
    all_rows = []
    for jf in json_files:
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                all_rows.append(json.load(f))
        except Exception as e:
            print(f"  ⚠ Error loading {jf.name}: {e}")
    
    # Create DataFrame - pd.DataFrame handles column alignment by NAME (not position!)
    df_new = pd.DataFrame(all_rows)
    print(f"Combined: {len(df_new)} matches, {len(df_new.columns)} columns")
    
    # Convert numeric columns
    for col in df_new.columns:
        if col not in ['league', 'season', 'url', 'match_id', 'home_team', 'away_team']:
            df_new[col] = pd.to_numeric(df_new[col], errors='coerce')
    
    # Load existing CSV if available and merge
    if OUTPUT_PATH.exists():
        print(f"Loading existing data from {OUTPUT_PATH}...")
        df_existing = pd.read_csv(OUTPUT_PATH, low_memory=False)
        print(f"Existing: {len(df_existing)} matches")
        
        # Combine, deduplicate by match_id
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=['match_id'], keep='last')
        print(f"Combined: {len(df_combined)} matches (removed {len(df_existing) + len(df_new) - len(df_combined)} dupes)")
    else:
        df_combined = df_new
    
    # Save
    df_combined.to_csv(OUTPUT_PATH, index=False)
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"New this run: {total_scraped}")
    print(f"Total in file: {len(df_combined)}")
    print(f"Columns: {len(df_combined.columns)}")
    print(f"Saved: {OUTPUT_PATH}")
    
    if 'league' in df_combined.columns and 'season' in df_combined.columns:
        print(f"\nPer league/season:")
        print(df_combined.groupby(['league', 'season']).size().to_string())
    
else:
    print("\n⚠ No JSON files found!")
    if OUTPUT_PATH.exists():
        df_existing = pd.read_csv(OUTPUT_PATH, low_memory=False)
        print(f"Existing file has {len(df_existing)} matches - nothing to add.")