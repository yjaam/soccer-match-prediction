import ScraperFC as sfc
from bs4 import BeautifulSoup
import pandas as pd
import json
import os
import re
from pathlib import Path
from datetime import datetime

fb = sfc.FBref()

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
CHECKPOINT_FILE = CHECKPOINT_DIR / "lineup_scraping_checkpoint.json"

if TEST_MODE:
    CHECKPOINT_FILE = CHECKPOINT_DIR / "lineup_scraping_checkpoint_TEST.json"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

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
    
    # Get team names from match stats section
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
            
            # Extract stats from td elements
            for td in row.find_all('td'):
                stat = td.get('data-stat', '')
                val = td.get_text(strip=True)
                
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
                    mins = int(player_data.get('minutes_played', 0))
                except (ValueError, TypeError):
                    mins = 0
                player_data['is_starter'] = mins >= 45
                
                players.append(player_data)
    
    return players


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
# PHASE 1: Get URLs and scrape lineups
# ============================================================================
print("\n" + "="*60)
print("PHASE 1: Getting match URLs and scraping lineups")
print("="*60)

all_players = []

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
                if url in completed_urls:
                    print(f"    [{i+1}/{len(links)}] Skip: {url.split('/')[-1][:60]}")
                    continue
                
                print(f"    [{i+1}/{len(links)}] Scraping: {url.split('/')[-1][:60]}...")
                
                try:
                    players = extract_lineups(url, league, season)
                    all_players.extend(players)
                    
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

output_name = "lineups_TEST.csv" if TEST_MODE else "lineups.csv"
output_path = PROJECT_DIR / "data" / output_name
os.makedirs(output_path.parent, exist_ok=True)

if all_players:
    df_new = pd.DataFrame(all_players)
    print(f"Newly scraped player entries: {len(df_new)}")
    
    # Load existing data if available
    if output_path.exists():
        print(f"Loading existing data from {output_path}...")
        df_existing = pd.read_csv(output_path)
        print(f"Existing player entries: {len(df_existing)}")
        
        # Combine, dropping duplicates by game_id + team_id + player
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        before_dedup = len(df_combined)
        df_combined = df_combined.drop_duplicates(
            subset=['game_id', 'team_id', 'player'], 
            keep='last'
        )
        after_dedup = len(df_combined)
        
        if before_dedup != after_dedup:
            print(f"Removed {before_dedup - after_dedup} duplicate entries")
    else:
        print(f"No existing file found, creating new one.")
        df_combined = df_new
    
    # Save combined data
    df_combined.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"New entries this run: {len(df_new)}")
    print(f"Total entries in file: {len(df_combined)}")
    print(f"Columns: {list(df_combined.columns)}")
    print(f"Saved: {output_path}")
    
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
    print("\n⚠ No lineups were scraped!")
    if output_path.exists():
        df_existing = pd.read_csv(output_path)
        print(f"Existing file has {len(df_existing)} entries - nothing to add.")