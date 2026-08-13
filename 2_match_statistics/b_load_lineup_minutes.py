import ScraperFC as sfc
from bs4 import BeautifulSoup
import pandas as pd
import json
import os
import re
import time
from pathlib import Path
from datetime import datetime
from botasaurus import browser

# ============================================================================
# HEADLESS PATCH (correct instance method version)
# ============================================================================

def _get_soup_headless(self, url: str) -> BeautifulSoup:
    """Headless version of _get_soup. Accepts self to work as instance method."""
    @browser(
        headless=True,
        block_images_and_css=True,
        wait_for_complete_page_load=False,
        output=None,
        create_error_logs=False,
    )
    def _(driver, url):
        driver.google_get(url)
        while True:
            try:
                driver.wait_for_element("body.fb", wait=10)
                break
            except Exception:
                driver.reload()
        return BeautifulSoup(driver.page_html, "html.parser")
    return _(url)

# Patch as instance method (NOT staticmethod)
sfc.FBref._get_soup = _get_soup_headless

fb = sfc.FBref()
print("Headless mode active")

# Verify get_match_links works
print("Testing get_match_links...")
try:
    test_links = fb.get_match_links(year='2024-2025', league='England Premier League')
    print(f"  ✓ get_match_links works: {len(test_links)} links found")
except Exception as e:
    print(f"  ❌ get_match_links broken: {e}")

# ============================================================================
# CONFIGURATION
# ============================================================================

TEST_MODE = True  # Set to False for full run

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
LINEUPS_JSON_DIR = PROJECT_DIR / "scraperfc_data" / "lineups_json"
CHECKPOINT_FILE = CHECKPOINT_DIR / "lineup_scraping_checkpoint.json"

if TEST_MODE:
    CHECKPOINT_FILE = CHECKPOINT_DIR / "lineup_scraping_checkpoint_TEST.json"
    LINEUPS_JSON_DIR = PROJECT_DIR / "scraperfc_data" / "lineups_json_test"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LINEUPS_JSON_DIR.mkdir(parents=True, exist_ok=True)

# Final output path
output_name = "detailed_lineups_TEST.csv" if TEST_MODE else "detailed_lineups.csv"
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
    if val is None:
        return None
    val = str(val).strip()
    if val == '' or val.lower() == 'nan':
        return None
    if ',' in val:
        cleaned = val.replace(',', '')
        try:
            float(cleaned)
            return cleaned
        except ValueError:
            pass
    return val

# ============================================================================
# EXTRACTION FUNCTION
# ============================================================================

def extract_lineups(url: str, league: str, season: str) -> list[dict]:
    """
    Extract per-player lineup data from a single FBref match page.
    Returns a list of dicts, one per player.
    """
    soup = fb._get_soup(url)
    
    match_id = url.split('/')[-2]
    
    # Find team IDs
    team_ids = []
    for div in soup.find_all('div', id=re.compile(r'all_player_stats_')):
        team_ids.append(div['id'].replace('all_player_stats_', ''))
    
    home_id = team_ids[0] if len(team_ids) > 0 else None
    away_id = team_ids[1] if len(team_ids) > 1 else None
    
    # Get team names
    team_names = {}
    team_stats_tbody = soup.select_one('#team_stats > table > tbody')
    if team_stats_tbody:
        first_row = team_stats_tbody.find_all('tr')[0]
        team_cells = first_row.find_all('th')
        if len(team_cells) > 0 and home_id:
            team_names[home_id] = team_cells[0].get_text(strip=True)
        if len(team_cells) > 1 and away_id:
            team_names[away_id] = team_cells[1].get_text(strip=True)
    
    players = []
    
    for side, team_id in [('Home', home_id), ('Away', away_id)]:
        if not team_id:
            continue
        
        team_name = team_names.get(team_id, team_id)
        
        tbody = soup.select_one(f'#stats_{team_id}_summary tbody')
        if not tbody:
            continue
        
        for row in tbody.find_all('tr'):
            player_data = {
                'league': league,
                'season': season,
                'game_id': match_id,
                'url': url,
                'side': side,
                'team': team_name,
                'team_id': team_id,
            }
            
            # Player name
            player_th = row.find('th', {'data-stat': 'player'})
            if player_th:
                player_data['player'] = player_th.get_text(strip=True)
            
            # Extract stats
            for td in row.find_all('td'):
                stat = td.get('data-stat', '')
                val = clean_value(td.get_text(strip=True))
                
                if stat == 'shirtnumber':
                    player_data['jersey_number'] = val
                elif stat == 'position':
                    player_data['position'] = val
                elif stat == 'age':
                    player_data['age'] = val
                elif stat == 'minutes':
                    player_data['minutes_played'] = val
                elif stat == 'goals':
                    player_data['goals'] = val
                elif stat == 'assists':
                    player_data['assists'] = val
                elif stat == 'cards_yellow':
                    player_data['cards_yellow'] = val
                elif stat == 'cards_red':
                    player_data['cards_red'] = val
            
            if player_data.get('player'):
                # Determine if starter
                try:
                    mins = int(float(player_data.get('minutes_played', 0)))
                except (ValueError, TypeError):
                    mins = 0
                player_data['is_starter'] = mins >= 45
                
                players.append(player_data)
    
    return players


def extract_lineups_with_retry(url, league, season, max_retries=3):
    for attempt in range(max_retries):
        try:
            return extract_lineups(url, league, season)
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
print(f"LINEUP & MINUTES SCRAPER")
print(f"{'TEST MODE' if TEST_MODE else 'PRODUCTION MODE'}")
print(f"{'='*60}")
print(f"Leagues: {big_5_leagues}")
print(f"Seasons: {seasons}")

checkpoint = load_checkpoint()
completed_urls = set(checkpoint["completed_urls"])
print(f"Already completed: {len(completed_urls)} URLs")

# ============================================================================
# PHASE 1: Scrape lineups, save as JSON
# ============================================================================
print("\n" + "="*60)
print("PHASE 1: Scraping lineups (saving as JSON files)")
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
                json_path = LINEUPS_JSON_DIR / f"{match_id}.json"
                
                # Skip if JSON already exists
                if json_path.exists():
                    continue
                
                # Restart browser periodically
                if matches_since_restart >= MATCHES_BEFORE_RESTART:
                    print("    🔄 Restarting browser...")
                    try:
                        fb_new = sfc.FBref()
                        fb = fb_new
                        # Re-patch _get_soup after re-initialization
                        sfc.FBref._get_soup = _get_soup_headless
                    except:
                        pass
                    matches_since_restart = 0
                    time.sleep(5)
                
                print(f"    [{i+1}/{len(links)}] {match_id}: {url.split('/')[-1][:50]}...")
                
                success = False
                for attempt in range(3):
                    try:
                        players = extract_lineups(url, league, season)
                        
                        # Save as JSON (one file per match, containing list of players)
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(players, f, ensure_ascii=False)
                        
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
                        sfc.FBref._get_soup = _get_soup_headless
                    except:
                        pass
                    matches_since_restart = 0
                    time.sleep(10)
                    
        except Exception as e:
            print(f"  ❌ Error getting links: {e}")

# ============================================================================
# PHASE 2: Combine all JSON files into one DataFrame
# ============================================================================
print("\n" + "="*60)
print("PHASE 2: Combining all JSON files into CSV")
print("="*60)

json_files = sorted(LINEUPS_JSON_DIR.glob("*.json"))
print(f"Found {len(json_files)} JSON files")

if json_files:
    # Load all JSONs - each contains a list of players
    all_players = []
    for jf in json_files:
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                players = json.load(f)
                all_players.extend(players)
        except Exception as e:
            print(f"  ⚠ Error loading {jf.name}: {e}")
    
    # Create DataFrame - proper column alignment by name
    df_new = pd.DataFrame(all_players)
    print(f"Combined: {len(df_new)} player entries, {len(df_new.columns)} columns")
    
    # Convert numeric columns
    numeric_cols = ['jersey_number', 'minutes_played', 'goals', 'assists', 
                    'cards_yellow', 'cards_red']
    for col in numeric_cols:
        if col in df_new.columns:
            df_new[col] = pd.to_numeric(df_new[col], errors='coerce')
    
    # Load existing CSV if available and merge
    if OUTPUT_PATH.exists():
        print(f"Loading existing data from {OUTPUT_PATH}...")
        df_existing = pd.read_csv(OUTPUT_PATH, low_memory=False)
        print(f"Existing: {len(df_existing)} player entries")
        
        # Combine, deduplicate
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        before = len(df_combined)
        df_combined = df_combined.drop_duplicates(
            subset=['game_id', 'team_id', 'player'], 
            keep='last'
        )
        after = len(df_combined)
        print(f"Combined: {after} entries (removed {before - after} dupes)")
    else:
        df_combined = df_new
    
    # Save
    df_combined.to_csv(OUTPUT_PATH, index=False)
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"New matches scraped: {total_scraped}")
    print(f"Total player entries: {len(df_combined)}")
    print(f"Columns: {list(df_combined.columns)}")
    print(f"Saved: {OUTPUT_PATH}")
    
    print(f"\nSample entries (first 10):")
    display_cols = ['league', 'season', 'game_id', 'side', 'team', 'player', 
                    'jersey_number', 'position', 'minutes_played', 'is_starter']
    display_cols = [c for c in display_cols if c in df_combined.columns]
    print(df_combined[display_cols].head(10).to_string())
    
    print(f"\nPer league/season:")
    if 'league' in df_combined.columns and 'season' in df_combined.columns:
        summary = df_combined.groupby(['league', 'season']).agg(
            matches=('game_id', 'nunique'),
            players=('player', 'count')
        )
        print(summary.to_string())
    
else:
    print("\n⚠ No JSON files found!")
    if OUTPUT_PATH.exists():
        df_existing = pd.read_csv(OUTPUT_PATH, low_memory=False)
        print(f"Existing file has {len(df_existing)} entries.")