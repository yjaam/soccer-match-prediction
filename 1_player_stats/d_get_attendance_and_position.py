import pandas as pd
import numpy as np
from pathlib import Path
import unicodedata
import sys
import os

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent  # Go up from 1_player_stats to project root

# Add PROJECT_DIR to path for importing from src/
sys.path.insert(0, str(PROJECT_DIR))

# FIXED: Import from src package
from src.team_name_mapping_FINAL import soccer_teams, resolve_team_name

# FIXED: Paths now point to project-level directories
LINEUPS_FILE = PROJECT_DIR / "data" / "lineups_values_ratings.csv"
CLUBS_FILE = PROJECT_DIR / "transfermarkt_data" / "clubs.csv"
GAMES_FILE = PROJECT_DIR / "transfermarkt_data" / "games.csv"
OUTPUT_FILE = PROJECT_DIR / "data" / "lineups_values_ratings_games.csv"

# Map lineups `Competition` values to transfermarkt `domestic_competition_id`
COMPETITION_TO_TM_ID = {
    'bundesliga': 'L1',
    'premier-league': 'GB1',
    'premier league': 'GB1',
    'laliga': 'ES1',
    'la liga': 'ES1',
    'la-liga': 'ES1',
    'serie-a': 'IT1',
    'serie a': 'IT1',
    'ligue-1': 'FR1',
    'ligue 1': 'FR1',
    'ligue1': 'FR1'
}


def strip_accents(text: str) -> str:
    """Remove accents from text for better matching."""
    if text is None:
        return ''
    s = unicodedata.normalize('NFKD', str(text))
    return ''.join(ch for ch in s if not unicodedata.combining(ch))


def comp_to_tm_id(comp: str):
    """Convert competition name to transfermarkt competition ID."""
    if pd.isna(comp):
        return None
    return COMPETITION_TO_TM_ID.get(str(comp).lower().strip())


def assert_exists(path: Path, label: str):
    """Check if a file exists, raise error with details if not."""
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found at:\n{path}\n"
            f"SCRIPT_DIR:\n{SCRIPT_DIR}\n"
            f"PROJECT_DIR:\n{PROJECT_DIR}\n"
            f"CWD:\n{Path.cwd()}"
        )


def get_club_id_from_team_code(team_code: str, clubs_df: pd.DataFrame, competition: str = None) -> int:
    """
    Find club_id from a three-letter team code (like 'ARS', 'MUN', 'PSG').
    
    Strategy:
    1. Look up the team code directly in clubs.csv 'club_code' column
    2. If not found, try to find the full name from the mapping and search clubs
    3. Fall back to fuzzy matching on club names
    """
    if pd.isna(team_code) or not team_code or str(team_code).strip() == '':
        return None
    
    team_code = str(team_code).strip().upper()
    
    # Filter clubs by competition if possible
    tm_comp = comp_to_tm_id(competition)
    if tm_comp:
        clubs_subset = clubs_df[clubs_df['domestic_competition_id'] == tm_comp]
        if clubs_subset.empty:
            clubs_subset = clubs_df
    else:
        clubs_subset = clubs_df
    
    # 1. Direct match on club_code (most reliable)
    match = clubs_subset[clubs_subset['club_code'].str.upper() == team_code]
    if not match.empty:
        return match.iloc[0]['club_id']
    
    # 2. Try to find the full name from the mapping
    # The mapping goes full_name -> code, so we need to reverse lookup
    possible_full_names = []
    for full_name, code in soccer_teams.items():
        if code.upper() == team_code:
            possible_full_names.append(full_name)
    
    # Try matching full names against clubs
    for full_name in possible_full_names:
        # Try exact match first
        match = clubs_subset[clubs_subset['name'] == full_name]
        if not match.empty:
            return match.iloc[0]['club_id']
        
        # Try contains match
        clean_name = strip_accents(full_name).lower()
        for _, club_row in clubs_subset.iterrows():
            club_name_clean = strip_accents(club_row['name']).lower()
            if clean_name in club_name_clean or club_name_clean in clean_name:
                return club_row['club_id']
    
    # 3. Try to match by searching club names that contain parts of known full names
    for full_name in possible_full_names:
        # Extract key words from full name (words longer than 3 chars)
        key_words = [w for w in full_name.split() if len(w) > 3 and w.lower() not in ['club', 'football', 'sport', 'verein']]
        for word in key_words:
            clean_word = strip_accents(word).lower()
            for _, club_row in clubs_subset.iterrows():
                club_name_clean = strip_accents(club_row['name']).lower()
                if clean_word in club_name_clean:
                    return club_row['club_id']
    
    # 4. Last resort: try matching the team code against parts of club names
    # This handles cases where the code might appear in the name
    for _, club_row in clubs_subset.iterrows():
        club_name_upper = strip_accents(club_row['name']).upper()
        if team_code in club_name_upper:
            return club_row['club_id']
    
    print(f"Warning: Could not find club_id for team code: {team_code}")
    return None


def get_team_code_from_name(team_name: str) -> str:
    """
    Get the three-letter team code from a full team name.
    Uses the resolve_team_name function from the mapping module.
    """
    if pd.isna(team_name):
        return None
    
    team_name = str(team_name).strip()
    return resolve_team_name(team_name)


def main():
    print(f"[INFO] SCRIPT_DIR:  {SCRIPT_DIR}")
    print(f"[INFO] PROJECT_DIR: {PROJECT_DIR}")
    print(f"[INFO] CWD:         {Path.cwd()}")
    print(f"[INFO] LINEUPS:     {LINEUPS_FILE}")
    print(f"[INFO] CLUBS:       {CLUBS_FILE}")
    print(f"[INFO] GAMES:       {GAMES_FILE}")

    assert_exists(LINEUPS_FILE, "LINEUPS_FILE")
    assert_exists(CLUBS_FILE, "CLUBS_FILE")
    assert_exists(GAMES_FILE, "GAMES_FILE")

    lineups_df = pd.read_csv(LINEUPS_FILE)
    clubs_df = pd.read_csv(CLUBS_FILE)
    games_df = pd.read_csv(GAMES_FILE)
    
    # Print sample data for debugging
    print(f"\n[INFO] Lineups shape: {lineups_df.shape}")
    print(f"[INFO] Clubs shape: {clubs_df.shape}")
    print(f"[INFO] Games shape: {games_df.shape}")
    
    print(f"\n[INFO] Sample of Home Team values in lineups:")
    print(lineups_df['Home Team'].dropna().unique()[:10])
    
    print(f"\n[INFO] Sample of club_code values in clubs:")
    print(clubs_df['club_code'].dropna().unique()[:10])
    
    # Convert dates
    games_df['date'] = pd.to_datetime(games_df['date'], errors='coerce')
    games_df = games_df.sort_values(by='date', ascending=False)

    print("\n[INFO] Mapping club IDs from team codes...")
    # Get unique team codes for progress reporting
    unique_teams = pd.concat([
        lineups_df['Home Team'].dropna(),
        lineups_df['Away Team'].dropna()
    ]).unique()
    
    print(f"[INFO] Unique team codes to map: {len(unique_teams)}")
    
    # Map team codes to club IDs
    team_to_club_id = {}
    for team_code in unique_teams:
        club_id = get_club_id_from_team_code(team_code, clubs_df)
        team_to_club_id[team_code] = club_id
    
    # Report mapping statistics
    mapped_count = sum(1 for v in team_to_club_id.values() if v is not None)
    unmapped_count = len(team_to_club_id) - mapped_count
    print(f"[INFO] Successfully mapped: {mapped_count}/{len(team_to_club_id)}")
    print(f"[INFO] Failed to map: {unmapped_count}")
    
    if unmapped_count > 0:
        print("\n[WARNING] Unmapped team codes:")
        for code, club_id in team_to_club_id.items():
            if club_id is None:
                print(f"  - {code}")
    
    # Apply mapping to lineups
    lineups_df['home_club_id'] = lineups_df['Home Team'].map(team_to_club_id)
    lineups_df['away_club_id'] = lineups_df['Away Team'].map(team_to_club_id)

    # Extract attendance and positions
    print("\n[INFO] Extracting attendance and positions...")
    attendances = []
    home_positions = []
    away_positions = []
    
    skipped_matches = 0
    
    for idx, row in lineups_df.iterrows():
        h_id = row['home_club_id']
        a_id = row['away_club_id']
        
        if pd.isna(h_id) or pd.isna(a_id):
            attendances.append(np.nan)
            home_positions.append(np.nan)
            away_positions.append(np.nan)
            skipped_matches += 1
            continue
        
        # Find the last match between these two teams
        last_match = games_df[
            ((games_df['home_club_id'] == h_id) & (games_df['away_club_id'] == a_id)) |
            ((games_df['home_club_id'] == a_id) & (games_df['away_club_id'] == h_id))
        ]
        
        if not last_match.empty:
            attendances.append(last_match.iloc[0]['attendance'])
        else:
            attendances.append(np.nan)
        
        # Latest home team position
        home_games = games_df[
            (games_df['home_club_id'] == h_id) | 
            (games_df['away_club_id'] == h_id)
        ]
        home_games_with_pos = home_games[
            ((home_games['home_club_id'] == h_id) & home_games['home_club_position'].notna()) |
            ((home_games['away_club_id'] == h_id) & home_games['away_club_position'].notna())
        ]
        
        if not home_games_with_pos.empty:
            latest_h = home_games_with_pos.iloc[0]
            pos_h = latest_h['home_club_position'] if latest_h['home_club_id'] == h_id else latest_h['away_club_position']
            home_positions.append(pos_h)
        else:
            home_positions.append(np.nan)
        
        # Latest away team position
        away_games = games_df[
            (games_df['home_club_id'] == a_id) | 
            (games_df['away_club_id'] == a_id)
        ]
        away_games_with_pos = away_games[
            ((away_games['home_club_id'] == a_id) & away_games['home_club_position'].notna()) |
            ((away_games['away_club_id'] == a_id) & away_games['away_club_position'].notna())
        ]
        
        if not away_games_with_pos.empty:
            latest_a = away_games_with_pos.iloc[0]
            pos_a = latest_a['home_club_position'] if latest_a['home_club_id'] == a_id else latest_a['away_club_position']
            away_positions.append(pos_a)
        else:
            away_positions.append(np.nan)
        
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(lineups_df)} matches...")
    
    lineups_df['attendance'] = attendances
    lineups_df['home_club_position_before_game'] = home_positions
    lineups_df['away_club_position_before_game'] = away_positions
    
    print(f"\n[INFO] Matches skipped due to missing club IDs: {skipped_matches}")
    print(f"[INFO] Attendance found: {sum(1 for a in attendances if pd.notna(a))}")
    print(f"[INFO] Home positions found: {sum(1 for p in home_positions if pd.notna(p))}")
    print(f"[INFO] Away positions found: {sum(1 for p in away_positions if pd.notna(p))}")

    # Save results
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    lineups_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n[SUCCESS] Done! Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()