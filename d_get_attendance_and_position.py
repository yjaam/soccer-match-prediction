import pandas as pd
import numpy as np
from pathlib import Path
import unicodedata

SCRIPT_DIR = Path(__file__).resolve().parent

LINEUPS_FILE = SCRIPT_DIR / "data" / "lineups_values_ratings.csv"
CLUBS_FILE = SCRIPT_DIR / "transfermarkt_data" / "clubs.csv"
GAMES_FILE = SCRIPT_DIR / "transfermarkt_data" / "games.csv"
OUTPUT_FILE = SCRIPT_DIR / "data" / "lineups_values_ratings_games.csv"

team_mapping = {
    'FC Bayern München': 'Bayern Munich',
    'Hertha BSC': 'Hertha BSC',
    'Verein für Leibesübungen Wolfsburg': 'Wolfsburg',
    '1. Fußball- und Sportverein Mainz 05': 'Mainz 05',
    'Hamburger SV': 'Hamburger SV',
    'FC Schalke 04': 'Schalke 04',
    'TSG 1899 Hoffenheim Fußball-Spielbetriebs GmbH': 'Hoffenheim',
    'Borussia Verein für Leibesübungen 1900 Mönchengladbach': 'Mönchengladbach',
    'Sport-Club Freiburg': 'Freiburg',
    '1. Fußball-Club Köln': 'Köln',
    'Eintracht Frankfurt Fußball AG': 'Eintracht Frankfurt',
    'Bayer 04 Leverkusen Fußball': 'Bayer Leverkusen',
    'FC Augsburg 1907': 'Augsburg',
    'Verein für Bewegungsspiele Stuttgart 1893': 'Stuttgart',
    'Borussia Dortmund': 'Dortmund',
    'Sportverein Werder Bremen von 1899': 'Werder Bremen',
    'Hannover 96': 'Hannover 96',
    'RasenBallsport Leipzig': 'RB Leipzig',
    'Fortuna Düsseldorf': 'Düsseldorf',
    '1.FC Nuremberg': 'Nürnberg',
    '1. FC Union Berlin': 'Union Berlin',
    'SC Paderborn 07': 'Paderborn 07',
    'Arminia Bielefeld': 'Arminia',
    'Verein für Leibesübungen Bochum 1848 – Fußballgemeinschaft': 'Bochum',
    'SpVgg Greuther Fürth': 'Greuther Fürth'
}

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
    if text is None:
        return ''
    s = unicodedata.normalize('NFKD', str(text))
    return ''.join(ch for ch in s if not unicodedata.combining(ch))

def comp_to_tm_id(comp: str):
    if pd.isna(comp):
        return None
    return COMPETITION_TO_TM_ID.get(str(comp).lower().strip())

def assert_exists(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found at:\n{path}\n"
            f"SCRIPT_DIR:\n{SCRIPT_DIR}\n"
            f"CWD:\n{Path.cwd()}"
        )

def safe_contains(series: pd.Series, pattern: str) -> pd.Series:
    # regex=False avoids regex edge cases in team names
    return series.astype(str).str.contains(pattern, case=False, na=False, regex=False)

def get_club_id(name, clubs_df, competition=None):
    if pd.isna(name):
        return None

    name_str = str(name).strip()
    # prefer filtering clubs by domestic competition when available
    tm_comp = comp_to_tm_id(competition)
    if tm_comp:
        clubs_subset = clubs_df[clubs_df['domestic_competition_id'] == tm_comp]
        # if subset empty, fall back to full clubs_df
        if clubs_subset.empty:
            clubs_subset = clubs_df
    else:
        clubs_subset = clubs_df

    # 1) direct mapped lookup (legacy)
    mapped_name = team_mapping.get(name_str)
    if mapped_name:
        mapped_code = mapped_name.lower().replace(' ', '-')
        match = clubs_subset[
            safe_contains(clubs_subset['name'], mapped_name) |
            safe_contains(clubs_subset['club_code'], mapped_code)
        ]
        if not match.empty:
            return match.iloc[0]['club_id']

    # 2) cleaned name lookup
    clean_name = (
        name_str
        .replace('FC ', '')
        .replace('1. ', '')
        .replace(' SV', '')
        .replace(' SC', '')
        .strip()
    )

    match = clubs_subset[safe_contains(clubs_subset['name'], clean_name)]
    if not match.empty:
        return match.iloc[0]['club_id']

    # 3) token fallback (NO list comprehension, avoids your bug completely)
    parts = clean_name.split()
    for token in parts:
        token = token.strip()
        if len(token) <= 3:
            continue
        match = clubs_subset[safe_contains(clubs_subset['name'], token)]
        if not match.empty:
            return match.iloc[0]['club_id']

    print(f"Warning: Could not find ID for club: {name_str}")
    return None

def main():
    print(f"[INFO] SCRIPT_DIR: {SCRIPT_DIR}")
    print(f"[INFO] CWD:       {Path.cwd()}")
    print(f"[INFO] LINEUPS:   {LINEUPS_FILE}")
    print(f"[INFO] CLUBS:     {CLUBS_FILE}")
    print(f"[INFO] GAMES:     {GAMES_FILE}")

    assert_exists(LINEUPS_FILE, "LINEUPS_FILE")
    assert_exists(CLUBS_FILE, "CLUBS_FILE")
    assert_exists(GAMES_FILE, "GAMES_FILE")

    lineups_df = pd.read_csv(LINEUPS_FILE)
    clubs_df = pd.read_csv(CLUBS_FILE)
    games_df = pd.read_csv(GAMES_FILE)

    games_df['date'] = pd.to_datetime(games_df['date'], errors='coerce')
    games_df = games_df.sort_values(by='date', ascending=False)

    print("Mapping club IDs (league-aware)...")
    # apply with access to the Competition column so we can filter clubs by domestic league
    lineups_df['home_club_id'] = lineups_df.apply(
        lambda r: get_club_id(r['Home Team'], clubs_df, r.get('Competition') if 'Competition' in r else None),
        axis=1
    )
    lineups_df['away_club_id'] = lineups_df.apply(
        lambda r: get_club_id(r['Away Team'], clubs_df, r.get('Competition') if 'Competition' in r else None),
        axis=1
    )

    attendances = []
    home_positions = []
    away_positions = []

    print("Extracting attendance and positions...")
    for _, row in lineups_df.iterrows():
        h_id = row['home_club_id']
        a_id = row['away_club_id']

        # last H2H (latest first because sorted descending by date)
        last_match = games_df[
            ((games_df['home_club_id'] == h_id) & (games_df['away_club_id'] == a_id)) |
            ((games_df['home_club_id'] == a_id) & (games_df['away_club_id'] == h_id))
        ]
        attendances.append(last_match.iloc[0]['attendance'] if not last_match.empty else np.nan)

        # latest known home-team position
        home_games = games_df[(games_df['home_club_id'] == h_id) | (games_df['away_club_id'] == h_id)]
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

        # latest known away-team position
        away_games = games_df[(games_df['home_club_id'] == a_id) | (games_df['away_club_id'] == a_id)]
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

    lineups_df['attendance'] = attendances
    lineups_df['home_club_position_before_game'] = home_positions
    lineups_df['away_club_position_before_game'] = away_positions

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    lineups_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Done! Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()