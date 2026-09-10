#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Scrape upcoming lineups from Rotowire into a local "upcoming_lineups" folder.

The script lives in its own dashboard folder and writes outputs next to itself.
It never touches the historical data/lineups.csv used for training.
"""

import os
import sys
import re
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup
import pandas as pd

# Import the resolver functions from the project src/ folder
try:
    from src.team_name_mapping_FINAL import resolve_team_name, resolve_team_name_fuzzy
except ModuleNotFoundError:
    # Add project root (two levels up from this script) to sys.path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.team_name_mapping_FINAL import resolve_team_name, resolve_team_name_fuzzy


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}

LEAGUES = {
    "Premier League": "",
    "Bundesliga": "BUND",
    "La Liga": "LIGA",
    "Serie A": "SERI",
    "Ligue 1": "FRAN",
    "Champions League": "UCL"
}

COMPETITION_STANDARDIZATION = {
    "Premier League": "premier-league",
    "Bundesliga": "bundesliga",
    "La Liga": "laliga",
    "Serie A": "serie-a",
    "Ligue 1": "ligue-1",
    "Champions League": "champions-league"
}


def resolve_team_code(name: str):
    """Resolve a scraped team name to the canonical team code."""
    if not name:
        return None
    code = resolve_team_name(name)
    if code is not None:
        return code
    return resolve_team_name_fuzzy(name)


def fetch(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.text


def parse_rotowire_date(date_str: str):
    """Parse Rotowire time labels into date objects."""
    today = datetime.today().date()
    if not date_str:
        return None

    date_str = str(date_str).strip()
    if not date_str:
        return None
    upper = date_str.upper()

    if upper.startswith("TODAY"):
        return today
    if upper.startswith("TOMORROW"):
        return today + timedelta(days=1)

    # Month format, e.g. "AUG 22" or "August 22 10:00 AM ET"
    month_match = re.match(
        r'(JAN(?:UARY)?|FEB(?:RUARY)?|MAR(?:CH)?|APR(?:IL)?|MAY|JUN(?:E)?|JUL(?:Y)?|AUG(?:UST)?|'
        r'SEP(?:TEMBER)?|OCT(?:OBER)?|NOV(?:EMBER)?|DEC(?:EMBER)?)\s+(\d{1,2})(?:\s*,?\s*(\d{4}))?',
        upper,
    )
    if month_match:
        month_token, day_str, year_str = month_match.groups()
        month = datetime.strptime(month_token[:3], '%b').month
        day = int(day_str)
        year = int(year_str) if year_str else today.year
        parsed_date = datetime(year, month, day).date()
        if parsed_date < today - timedelta(days=30):
            parsed_date = datetime(year + 1, month, day).date()
        return parsed_date

    # Weekday format, e.g. "FRI 2:30 PM"
    weekday_match = re.match(
        r'(MON(?:DAY)?|TUE(?:SDAY)?|WED(?:NESDAY)?|THU(?:RSDAY)?|FRI(?:DAY)?|SAT(?:URDAY)?|SUN(?:DAY)?)',
        upper,
    )
    if weekday_match:
        wd_str = weekday_match.group(1)[:3]
        weekdays = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
        target_wd = weekdays.index(wd_str)
        today_wd = today.weekday()
        diff = target_wd - today_wd
        if diff < -3:
            diff += 7
        return today + timedelta(days=diff)

    # Generic fallback
    cleaned = re.sub(r'\bET\b', '', date_str, flags=re.IGNORECASE).strip()
    parsed = pd.to_datetime(cleaned, errors='coerce')
    if pd.notna(parsed):
        parsed_date = parsed.date()
        if parsed_date < today - timedelta(days=30) and parsed.year == today.year:
            parsed_date = datetime(today.year + 1, parsed.month, parsed.day).date()
        return parsed_date

    return None


def parse_players(li_elements: list):
    """Map Rotowire positions to GK / DEF / MID / ATT buckets."""
    goalkeeper = ""
    defenders, midfielders, attackers = [], [], []

    for li in li_elements:
        pos_el = li.select_one("div")
        name_el = li.select_one("a")
        if not pos_el or not name_el:
            continue

        pos = pos_el.get_text(strip=True).upper()
        name = name_el.get_text(strip=True)

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
        time_el = match.select_one(".lineup__time")
        raw_date_str = time_el.get_text(" ", strip=True) if time_el else ""

        parsed_date = parse_rotowire_date(raw_date_str)
        if parsed_date is None:
            if first_match_date is None:
                continue
            parsed_date = first_match_date

        if first_match_date is None:
            first_match_date = parsed_date

        if (parsed_date - first_match_date).days > 3:
            break

        home_team_el = match.select_one(".lineup__matchup .lineup__mteam.is-home")
        away_team_el = match.select_one(".lineup__matchup .lineup__mteam.is-visit")
        if not home_team_el or not away_team_el:
            continue

        home_team = home_team_el.get_text(strip=True)
        away_team = away_team_el.get_text(strip=True)

        home_list = match.select("ul.lineup__list.is-home > li")
        away_list = match.select("ul.lineup__list.is-visit > li")
        if not home_list or not away_list:
            continue

        home_gk, home_defs, home_mids, home_atts = parse_players(home_list)
        away_gk, away_defs, away_mids, away_atts = parse_players(away_list)

        current_match = {
            "Competition": COMPETITION_STANDARDIZATION.get(league_name, league_name),
            "Date": parsed_date.strftime("%Y-%m-%d"),
            "Home Team": home_team,
            "Away Team": away_team,
            "Home_Goalkeeper": home_gk,
            "Away_Goalkeeper": away_gk,
        }

        for i, d in enumerate(home_defs, 1):
            current_match[f"Home_Defender_{i}"] = d
        for i, m in enumerate(home_mids, 1):
            current_match[f"Home_Midfielder_{i}"] = m
        for i, a in enumerate(home_atts, 1):
            current_match[f"Home_Attacker_{i}"] = a

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
    for name, code in LEAGUES.items():
        all_matches.extend(scrape_league(name, code))

    df_scraped = pd.DataFrame(all_matches)

    if df_scraped.empty:
        print("\nNo lineups found for any leagues. Nothing to save.")
        return

    # --- Resolve team names to canonical codes ---
    df_scraped["Home Team Raw"] = df_scraped["Home Team"]
    df_scraped["Away Team Raw"] = df_scraped["Away Team"]
    df_scraped["Home Team"] = df_scraped["Home Team Raw"].apply(resolve_team_code)
    df_scraped["Away Team"] = df_scraped["Away Team Raw"].apply(resolve_team_code)

    unresolved_home = df_scraped[df_scraped["Home Team"].isna()]["Home Team Raw"].dropna().unique().tolist()
    unresolved_away = df_scraped[df_scraped["Away Team"].isna()]["Away Team Raw"].dropna().unique().tolist()
    unresolved = sorted(set(unresolved_home + unresolved_away))
    if unresolved:
        print(f"\nWarning: {len(unresolved)} team names could not be resolved to IDs.")
        for team_name in unresolved:
            print(f"  - {team_name}")

    # Order columns
    base_cols = ["Competition", "Date", "Home Team", "Away Team", "Home Team Raw", "Away Team Raw"]
    player_cols = [c for c in df_scraped.columns if c not in base_cols]
    df_scraped = df_scraped[base_cols + player_cols]
    df_scraped = df_scraped.fillna("")

    # --- Write to a local "upcoming_lineups" folder next to this script ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "upcoming_lineups")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"upcoming_lineups_{timestamp}.csv")
    latest_path = os.path.join(output_dir, "upcoming_lineups_latest.csv")

    # Sort by date before saving
    df_scraped["Date"] = pd.to_datetime(df_scraped["Date"])
    df_scraped = df_scraped.sort_values("Date").reset_index(drop=True)
    df_scraped["Date"] = df_scraped["Date"].dt.strftime("%Y-%m-%d")

    df_scraped.to_csv(output_path, index=False, encoding="utf-8-sig")
    df_scraped.to_csv(latest_path, index=False, encoding="utf-8-sig")

    pd.set_option("display.max_columns", None)
    print(f"\n--- Scraping Complete ---")
    print(f"Rows scraped: {len(df_scraped)}")
    print(f"Timestamped file: {output_path}")
    print(f"Latest copy:      {latest_path}")
    print(f"\nDate range: {df_scraped['Date'].min()} to {df_scraped['Date'].max()}")
    print(f"\nPreview of the first match:")
    print(df_scraped.head(1).to_string(index=False))


if __name__ == "__main__":
    main()