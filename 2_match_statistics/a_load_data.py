import ScraperFC as sfc
from bs4 import BeautifulSoup
import pandas as pd
import pickle
import json
import os
import re
from pathlib import Path
from datetime import datetime

fb = sfc.FBref()

# ============================================================================
# CONFIGURATION
# ============================================================================

TEST_MODE = False  # Set to False for full run

if TEST_MODE:
    seasons = ['2023-2024']  # Just 1 season for testing
else:
    seasons = ['2014-2015', '2015-2016', '2016-2017', '2017-2018', '2018-2019', '2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025', '2025-2026', '2026-2027']

big_5_leagues = ['England Premier League', 'France Ligue 1', 'Germany Bundesliga', 'Italy Serie A', 'Spain La Liga']

if TEST_MODE:
    big_5_leagues = ['England Premier League']

# Setup directories
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
CHECKPOINT_DIR = PROJECT_DIR / "scraperfc_data" / "checkpoints"
MATCHES_DIR = PROJECT_DIR / "scraperfc_data" / "matches"
CHECKPOINT_FILE = CHECKPOINT_DIR / "match_scraping_checkpoint.json"

if TEST_MODE:
    CHECKPOINT_FILE = CHECKPOINT_DIR / "match_scraping_checkpoint_TEST.json"
    MATCHES_DIR = PROJECT_DIR / "scraperfc_data" / "matches_test"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
MATCHES_DIR.mkdir(parents=True, exist_ok=True)

# Reset test checkpoint
if TEST_MODE and CHECKPOINT_FILE.exists():
    CHECKPOINT_FILE.unlink()

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
    
    # Get team names from match stats section
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
                    home_val = len(cells[0].find_all('span', class_='yellow_card'))
                    away_val = len(cells[1].find_all('span', class_='yellow_card'))
                else:
                    home_text = cells[0].get_text(strip=True)
                    away_text = cells[1].get_text(strip=True)
                    home_pct = re.findall(r'(\d+)%', home_text)
                    away_pct = re.findall(r'(\d+)%', away_text)
                    home_val = home_pct[0] if home_pct else None
                    away_val = away_pct[0] if away_pct else None
                
                row[f'Home_{stat_name}'] = home_val
                row[f'Away_{stat_name}'] = away_val
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
                    row[f'Home_{stat_name}'] = home_val
                    row[f'Away_{stat_name}'] = away_val
    
    # ============================================================
    # 3. Player summary totals from <tfoot>
    # ============================================================
    for side, team_id in [('Home', home_id), ('Away', away_id)]:
        if team_id:
            tfoot = soup.select_one(f'#stats_{team_id}_summary tfoot tr')
            if tfoot:
                for td in tfoot.find_all(['th', 'td']):
                    stat = td.get('data-stat', '')
                    if stat and stat != 'player':
                        val = td.get_text(strip=True)
                        row[f'{side}_{stat}'] = val
    
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
                        if stat and stat != 'player':
                            val = td.get_text(strip=True)
                            row[f'{side}_keeper_{stat}'] = val
    
    return row


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
print(f"Already completed: {len(completed_urls)} URLs")

# ============================================================================
# PHASE 1: Get all match URLs and scrape
# ============================================================================
print("\n" + "="*60)
print("PHASE 1: Getting match URLs and scraping")
print("="*60)

all_rows = []

for league in big_5_leagues:
    for season in seasons:
        print(f"\n  Getting URLs for {league} {season}...")
        
        try:
            links = fb.get_match_links(year=season, league=league)
            print(f"  Found {len(links)} match URLs")
            
            # Limit to 5 matches in test mode
            if TEST_MODE:
                links = links[:5]
                print(f"  TEST MODE: Limited to {len(links)} matches")
            
            # Scrape each match
            for i, url in enumerate(links):
                if url in completed_urls:
                    print(f"    [{i+1}/{len(links)}] Skip: {url.split('/')[-1][:60]}")
                    continue
                
                print(f"    [{i+1}/{len(links)}] Scraping: {url.split('/')[-1][:60]}...")
                
                try:
                    row = extract_match_data(url, league, season)
                    all_rows.append(row)
                    
                    checkpoint["completed_urls"].append(url)
                    save_checkpoint(checkpoint)
                    
                except Exception as e:
                    print(f"      ❌ Error: {e}")
                    checkpoint["failed_urls"].append(url)
                    save_checkpoint(checkpoint)
                    
        except Exception as e:
            print(f"  ❌ Error getting links: {e}")

# ============================================================================
# PHASE 2: Combine with existing data and save
# ============================================================================
print("\n" + "="*60)
print("PHASE 2: Saving results")
print("="*60)

output_name = "big5_matches_TEST.csv" if TEST_MODE else "big5_matches.csv"
output_path = PROJECT_DIR / "data" / output_name
os.makedirs(output_path.parent, exist_ok=True)

if all_rows:
    df_new = pd.DataFrame(all_rows)
    print(f"Newly scraped matches: {len(df_new)}")
    
    # Load existing data if available
    if output_path.exists():
        print(f"Loading existing data from {output_path}...")
        df_existing = pd.read_csv(output_path)
        print(f"Existing matches: {len(df_existing)}")
        
        # Combine, dropping any duplicates by match_id
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        before_dedup = len(df_combined)
        df_combined = df_combined.drop_duplicates(subset=['match_id'], keep='last')
        after_dedup = len(df_combined)
        
        if before_dedup != after_dedup:
            print(f"Removed {before_dedup - after_dedup} duplicate matches")
    else:
        print(f"No existing file found, creating new one.")
        df_combined = df_new
    
    # Save combined data
    df_combined.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"New matches this run: {len(df_new)}")
    print(f"Total matches in file: {len(df_combined)}")
    print(f"Columns: {len(df_combined.columns)}")
    print(f"Saved: {output_path}")
    
    if 'league' in df_combined.columns and 'season' in df_combined.columns:
        print(f"\nPer league/season:")
        print(df_combined.groupby(['league', 'season']).size().to_string())
    
    print(f"\nFirst 5 rows (key columns):")
    key_cols = ['league', 'season', 'match_id', 'home_team', 'away_team']
    key_cols = [c for c in key_cols if c in df_combined.columns]
    if len(key_cols) > 0:
        print(df_combined[key_cols].head(5).to_string())
    
    print(f"\nAll column names:")
    for i, col in enumerate(df_combined.columns):
        non_null = df_combined[col].notna().sum()
        print(f"  {i:3d}: {col} ({non_null}/{len(df_combined)} non-null)")
    
else:
    print("\n⚠ No new matches were scraped!")
    if output_path.exists():
        df_existing = pd.read_csv(output_path)
        print(f"Existing file has {len(df_existing)} matches - nothing to add.")