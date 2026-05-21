import os
import re
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup
import pandas as pd

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}

# Rotowire league parameters. Empty string for Premier League.
LEAGUES = {
    "Premier League": "",
    "Bundesliga": "BUND",
    "La Liga": "LIGA",
    "Serie A": "SERI",
    "Ligue 1": "FRAN",
    "Champions League": "UCL"
}

# Standardized competition names (matching historic data and filter script)
COMPETITION_STANDARDIZATION = {
    "Premier League": "premier-league",
    "Bundesliga": "bundesliga",
    "La Liga": "laliga",
    "Serie A": "serie-a",
    "Ligue 1": "ligue-1",
    "Champions League": "champions-league"
}

def fetch(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.text

def parse_rotowire_date(date_str: str) -> datetime.date:
    """
    Converts Rotowire relative strings like "FRI 2:30 PM" or "MAY 8 9:00 AM" 
    into actual datetime.date objects for math comparison.
    """
    today = datetime.today().date()
    date_str = date_str.strip().upper()
    
    # 1) Check for Month format (e.g. "MAY 8" or "MAY 8 2:30 PM")
    month_match = re.match(r'(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+(\d+)', date_str)
    if month_match:
        month_str, day_str = month_match.groups()
        month = datetime.strptime(month_str, '%b').month
        day = int(day_str)
        
        parsed_date = datetime(today.year, month, day).date()
        if parsed_date < today - timedelta(days=30):
            parsed_date = datetime(today.year + 1, month, day).date()
        return parsed_date

    # 2) Check for Weekday format (e.g. "FRI 2:30 PM")
    weekday_match = re.match(r'(MON|TUE|WED|THU|FRI|SAT|SUN)', date_str)
    if weekday_match:
        wd_str = weekday_match.group(1)
        weekdays = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
        target_wd = weekdays.index(wd_str)
        today_wd = today.weekday()
        
        diff = target_wd - today_wd
        if diff < -3:
            diff += 7
            
        return today + timedelta(days=diff)

    return today

def parse_players(li_elements: list) -> tuple[str, list[str], list[str], list[str]]:
    """
    Maps Rotowire positions accurately to our 4 buckets, while excluding injuries.
    """
    goalkeeper = ""
    defenders = []
    midfielders = []
    attackers = []

    for li in li_elements:
        pos_el = li.select_one("div")
        name_el = li.select_one("a")
        
        if not pos_el or not name_el:
            continue
            
        pos = pos_el.get_text(strip=True).upper()
        name = name_el.get_text(strip=True)

        # Exclude injuries / suspensions (they use 1-letter codes or slashes)
        if len(pos) == 1 or "/" in pos:
            continue

        if pos == "GK":
            goalkeeper = name
        elif pos.startswith("F") or pos in ["ST", "RW", "LW", "A"]:
            attackers.append(name)
        elif pos.startswith("M") or pos.startswith("AM") or pos.startswith("DM"):
            midfielders.append(name)
        elif pos.startswith("D") and not pos.startswith("DM"):
            defenders.append(name)
        else:
            midfielders.append(name)

    return goalkeeper, defenders, midfielders, attackers

def scrape_league(league_name: str, league_code: str) -> list:
    """Scrapes a single league and returns a list of match dictionaries."""
    print(f"Scraping {league_name}...")
    
    if not league_code:
        url = "https://www.rotowire.com/soccer/lineups.php"
    else:
        url = f"https://www.rotowire.com/soccer/lineups.php?league={league_code}"
    
    try:
        html = fetch(url)
    except Exception as e:
        print(f"  -> Failed to fetch {league_name}: {e}")
        return []
        
    soup = BeautifulSoup(html, "html.parser")
    match_data = []
    
    matches = soup.select(".lineups > div.lineup")
    if not matches:
        matches = [m for m in soup.select(".lineups > div") if m.select_one(".lineup__box")]

    first_match_date = None

    for match in matches:
        # Date Parsing & Window Filtering
        time_el = match.select_one(".lineup__time")
        raw_date_str = time_el.get_text(" ", strip=True) if time_el else ""
        
        parsed_date = parse_rotowire_date(raw_date_str)
        
        if first_match_date is None:
            first_match_date = parsed_date
            
        # 3-day window constraint
        if (parsed_date - first_match_date).days > 3:
            break

        # Team Names
        home_team_el = match.select_one(".lineup__matchup .lineup__mteam.is-home")
        away_team_el = match.select_one(".lineup__matchup .lineup__mteam.is-visit")
        
        if not home_team_el or not away_team_el:
            continue

        home_team = home_team_el.get_text(strip=True)
        away_team = away_team_el.get_text(strip=True)

        # Lineups
        home_list = match.select("ul.lineup__list.is-home > li")
        away_list = match.select("ul.lineup__list.is-visit > li")

        if not home_list or not away_list:
            continue

        home_gk, home_defs, home_mids, home_atts = parse_players(home_list)
        away_gk, away_defs, away_mids, away_atts = parse_players(away_list)

        # Build dictionary with the new Competition column (using standardized names)
        current_match = {
            "Competition": COMPETITION_STANDARDIZATION.get(league_name, league_name),
            "Date": parsed_date.strftime("%Y-%m-%d"),
            "Home Team": home_team,
            "Away Team": away_team,
            "Home_Goalkeeper": home_gk,
            "Away_Goalkeeper": away_gk,
        }

        # Populate Home Outfield
        for i, d in enumerate(home_defs, 1):
            current_match[f"Home_Defender_{i}"] = d
        for i, m in enumerate(home_mids, 1):
            current_match[f"Home_Midfielder_{i}"] = m
        for i, a in enumerate(home_atts, 1):
            current_match[f"Home_Attacker_{i}"] = a

        # Populate Away Outfield
        for i, d in enumerate(away_defs, 1):
            current_match[f"Away_Defender_{i}"] = d
        for i, m in enumerate(away_mids, 1):
            current_match[f"Away_Midfielder_{i}"] = m
        for i, a in enumerate(away_atts, 1):
            current_match[f"Away_Attacker_{i}"] = a

        match_data.append(current_match)

    print(f"  -> Found {len(match_data)} matches.")
    return match_data

def main():
    all_matches = []
    
    # Iterate through all configured leagues
    for name, code in LEAGUES.items():
        league_matches = scrape_league(name, code)
        all_matches.extend(league_matches)

    df_scraped = pd.DataFrame(all_matches)
    
    # Format and reorder columns
    if not df_scraped.empty:
        # Base columns updated to include Competition
        base_cols = ["Competition", "Date", "Home Team", "Away Team"]
        # Fixed reference to df_scraped.columns
        player_cols = [c for c in df_scraped.columns if c not in base_cols]
        df_scraped = df_scraped[base_cols + player_cols]

    df_scraped = df_scraped.fillna("")

    # Set up output path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "data", "lineups.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load existing lineups if available
    if os.path.exists(output_path):
        print("\nLoading existing lineups...")
        df_existing = pd.read_csv(output_path, low_memory=False)
        print(f"Existing rows: {len(df_existing)}")
        
        # Create a match key for duplicate detection
        df_existing['match_key'] = (
            df_existing['Competition'] + '|' +
            df_existing['Date'].astype(str) + '|' +
            df_existing['Home Team'] + '|' +
            df_existing['Away Team']
        )
        
        df_scraped['match_key'] = (
            df_scraped['Competition'] + '|' +
            df_scraped['Date'].astype(str) + '|' +
            df_scraped['Home Team'] + '|' +
            df_scraped['Away Team']
        )
        
        # Find new matches (not in existing data)
        existing_keys = set(df_existing['match_key'])
        df_new = df_scraped[~df_scraped['match_key'].isin(existing_keys)].copy()
        
        # Remove the temporary key column
        df_existing = df_existing.drop(columns=['match_key'])
        df_new = df_new.drop(columns=['match_key'])
        
        if len(df_new) > 0:
            print(f"New rows found: {len(df_new)}")
            # Append new rows to existing data
            df_lineups = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            print("No new rows found (all matches already in database)")
            df_lineups = df_existing
    else:
        print("\nNo existing lineups file found. Creating new file...")
        df_lineups = df_scraped.copy()

    # Sort by date
    df_lineups['Date'] = pd.to_datetime(df_lineups['Date'])
    df_lineups = df_lineups.sort_values('Date').reset_index(drop=True)
    # Convert date back to string format
    df_lineups['Date'] = df_lineups['Date'].dt.strftime('%Y-%m-%d')

    # Save to CSV
    df_lineups.to_csv(output_path, index=False, encoding="utf-8-sig")

    pd.set_option("display.max_columns", None)
    print(f"\n--- Scraping Complete ---")
    if not df_lineups.empty:
        print(f"Total rows in database: {len(df_lineups)}")
        print(f"Saved data to: {output_path}\n")
        print("Preview of the first match:")
        print(df_lineups.head(1).to_string(index=False))
    else:
        print("No lineups found for any leagues.")

if __name__ == "__main__":
    main()