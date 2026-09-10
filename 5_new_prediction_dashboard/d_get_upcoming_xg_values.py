#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Fetch xG data + compute pre-match form for UPCOMING matches (dashboard use).

Differs from the historical script:
- Fetches full history (needed to compute form)
- Matches upcoming fixtures from ./upcoming_lineups/
- Writes:
    ./prediction_data/xg_history.csv           (raw Understat history, local copy)
    ./prediction_data/upcoming_xg_form.csv     (per-match form features for predictions)
- Never touches data/big5_xg_clean.csv (training artifact)
"""

import os
import re
import sys
import unicodedata
import pandas as pd
from datetime import datetime
from understatapi import UnderstatClient

# Path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from src.team_name_mapping_FINAL import resolve_team_name_fuzzy, soccer_teams


NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")

LOCAL_ALIASES = {
    "Manchester Utd": "MUN",
    "PSG": "PSG",
    "Frankfurt": "SGE",
    "Gladbach": "BMG",
    "Hellas Verona": "VER",
    "Hertha BSC": "BSC",
    "Nottingham": "NFO",
    "Atlético Madrid": "ATM",
    "Köln": "KOE",
    "Alavés": "ALA",
    "Saint-Étienne": "STE",
    "Rayo Vallecano": "RAY",
    "Dep. La Coruña": "DEP",
    "Málaga": "MAL",
    "Cádiz": "CAD",
    "Almería": "ALM",
    "Nîmes": "NIM",
    "Leganés": "LEG",
    "Darmstadt 98": "D98",
}


def normalize_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", name)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    text = NON_ALNUM_PATTERN.sub(" ", text)
    return " ".join(text.split())


NORMALIZED_LOOKUP = {}
for team_name, code in soccer_teams.items():
    NORMALIZED_LOOKUP.setdefault(normalize_name(str(team_name)), code)


def resolve_team_code(name_value: str):
    if pd.isna(name_value):
        return None
    name_text = str(name_value).strip()
    code = resolve_team_name_fuzzy(name_text)
    if code:
        return code
    alias_code = LOCAL_ALIASES.get(name_text)
    if alias_code:
        return alias_code
    return NORMALIZED_LOOKUP.get(normalize_name(name_text))


def parse_matchday(m):
    for key in ("round", "gameweek", "week"):
        if key in m and m[key] is not None:
            return pd.to_numeric(m[key], errors="coerce")
    return pd.NA


# ============================================================================
# UNDERSTAT FETCHING (historical + current season)
# ============================================================================

def fetch_league_season_matches(league_name, seasons):
    understat = UnderstatClient()
    LEAGUE_CANDIDATES = {
        "Premier League": ["Premier League", "EPL", "premier-league", "england"],
        "La Liga": ["La Liga", "LaLiga", "La_Liga", "laliga", "spain"],
        "Serie A": [
            "Serie A", "SerieA", "Serie_A", "serie-a", "serie_a",
            "italy", "italia", "Serie A TIM", "SERIE A",
        ],
        "Ligue 1": [
            "Ligue 1", "Ligue1", "Ligue_1", "ligue-1", "ligue_1",
            "france", "Ligue 1 Uber Eats", "Ligue1UberEats", "Ligue1_UberEats",
        ],
        "Bundesliga": ["Bundesliga", "bundesliga"],
    }

    candidates = LEAGUE_CANDIDATES.get(league_name, [league_name])
    rows = []

    for s in seasons:
        season_str = str(s)
        matches = None
        used_candidate = None

        for cand in candidates:
            try:
                league = understat.league(league=cand)
                matches = league.get_match_data(season=season_str)
                used_candidate = cand
            except Exception:
                matches = None
            if matches is not None:
                break

        if matches is None:
            print(f"[WARN] {league_name} season {season_str} skipped (no valid candidate)")
            continue

        if not matches:
            print(f"[INFO] {league_name} season {season_str}: no matches (tried '{used_candidate}').")
            continue

        for m in matches:
            date_raw = m.get("datetime")
            date = pd.to_datetime(date_raw, errors="coerce")

            home_team = (m.get("h") or {}).get("title")
            away_team = (m.get("a") or {}).get("title")

            xg = m.get("xG") or {}
            home_xg = pd.to_numeric(xg.get("h"), errors="coerce")
            away_xg = pd.to_numeric(xg.get("a"), errors="coerce")

            rows.append({
                "league": league_name,
                "season": s,
                "matchday": parse_matchday(m),
                "date": date.date() if pd.notna(date) else pd.NaT,
                "home_team": home_team,
                "away_team": away_team,
                "home_xg": home_xg,
                "away_xg": away_xg,
            })

    return rows


def fetch_all_xg_history():
    current_year = datetime.today().year
    seasons = list(range(2012, current_year + 1))

    leagues = ["Premier League", "La Liga", "Serie A", "Ligue 1", "Bundesliga"]

    all_rows = []
    for lg in leagues:
        print(f"[INFO] Fetching {lg} for seasons {seasons} ...")
        all_rows.extend(fetch_league_season_matches(lg, seasons))

    if not all_rows:
        raise RuntimeError("No match data found from Understat.")

    df = pd.DataFrame(all_rows)
    df = df.dropna(subset=["date", "home_team", "away_team"]).copy()
    df["matchday"] = pd.to_numeric(df["matchday"], errors="coerce")
    df = df.drop_duplicates(subset=["league", "season", "date", "home_team", "away_team"], keep="first")
    df = df.sort_values(by=["date", "league", "season", "matchday", "home_team"],
                        na_position="last").reset_index(drop=True)
    return df


def add_canonical_team_names(df):
    """Resolve teams to canonical codes and build game_id."""
    out = df.copy()
    out["home_team_mapped"] = out["home_team"].apply(resolve_team_code)
    out["away_team_mapped"] = out["away_team"].apply(resolve_team_code)
    out["home_team_mapped"] = out["home_team_mapped"].fillna(out["home_team"])
    out["away_team_mapped"] = out["away_team_mapped"].fillna(out["away_team"])

    date_token = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y%m%d")
    out["game_id"] = (date_token + "_" +
                      out["home_team_mapped"].astype(str) + "_" +
                      out["away_team_mapped"].astype(str))
    out.loc[
        pd.to_datetime(out["date"], errors="coerce").isna()
        | out["home_team_mapped"].isna()
        | out["away_team_mapped"].isna(),
        "game_id",
    ] = pd.NA
    return out


# ============================================================================
# WEIGHTED FORM COMPUTATION
# ============================================================================

def build_team_xg_history(xg_df):
    """Build per-team xG history lists for weighted form computation."""
    team_hist = {}
    xg_df = xg_df.sort_values("date").copy()

    for _, r in xg_df.iterrows():
        ht = r["home_team_mapped"]
        at = r["away_team_mapped"]
        d = pd.to_datetime(r["date"], errors="coerce")
        hxg = r["home_xg"]
        axg = r["away_xg"]

        if pd.notna(ht) and pd.notna(hxg) and pd.notna(d):
            team_hist.setdefault(ht, []).append((d, float(hxg)))
        if pd.notna(at) and pd.notna(axg) and pd.notna(d):
            team_hist.setdefault(at, []).append((d, float(axg)))

    for team in team_hist:
        team_hist[team].sort(key=lambda x: x[0])
    return team_hist


def weighted_form(values, weights=(5, 4, 3, 2, 1)):
    """Compute weighted form from up to 5 most recent values (recent first)."""
    if len(values) < 5:
        return float("nan")
    return float(sum(v * w for v, w in zip(values[:5], weights)))


def get_last_n_before_date(history_list, current_date, n=5):
    """Return up to n values strictly before current_date, most recent first."""
    prev = [val for (d, val) in history_list if d < current_date]
    if not prev:
        return []
    return prev[::-1][:n]


# ============================================================================
# MAIN
# ============================================================================

def main():
    upcoming_dir = os.path.join(script_dir, "upcoming_lineups")
    prediction_data_dir = os.path.join(script_dir, "prediction_data")
    os.makedirs(prediction_data_dir, exist_ok=True)

    upcoming_path = os.path.join(upcoming_dir, "upcoming_lineups_latest.csv")
    history_out = os.path.join(prediction_data_dir, "xg_history.csv")
    form_out = os.path.join(prediction_data_dir, "upcoming_xg_form.csv")

    if not os.path.exists(upcoming_path):
        print(f"Error: upcoming lineups not found at {upcoming_path}")
        print("Run a_scrape_upcoming_lineups.py first.")
        return

    # ------------------------------------------------------------------
    # 1. Fetch full xG history
    # ------------------------------------------------------------------
    print("Fetching full xG history from Understat...")
    xg = fetch_all_xg_history()
    xg = add_canonical_team_names(xg)
    print(f"Fetched {len(xg)} historical xG matches")

    # Save the local copy (dashboard-scoped, not the training artifact)
    xg.to_csv(history_out, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved raw xG history to: {history_out}")

    # ------------------------------------------------------------------
    # 2. Build team xG histories
    # ------------------------------------------------------------------
    print("\nBuilding team xG histories...")
    team_hist = build_team_xg_history(xg)
    print(f"Built histories for {len(team_hist)} teams")

    # ------------------------------------------------------------------
    # 3. Load upcoming fixtures
    # ------------------------------------------------------------------
    upcoming = pd.read_csv(upcoming_path, low_memory=False)
    print(f"\nLoaded {len(upcoming)} upcoming matches")

    if "game_id" not in upcoming.columns:
        upcoming["game_id"] = (
            upcoming["Date"].astype(str).str.replace("-", "", regex=False)
            + "_" + upcoming["Home Team"].astype(str)
            + "_" + upcoming["Away Team"].astype(str)
        )

    # ------------------------------------------------------------------
    # 4. Compute pre-match weighted form for each upcoming fixture
    # ------------------------------------------------------------------
    print("\nComputing weighted xG form for upcoming fixtures...")
    rows = []
    missing_teams = set()

    for _, match in upcoming.iterrows():
        match_date = pd.to_datetime(match["Date"], errors="coerce")
        ht = match["Home Team"]
        at = match["Away Team"]

        if pd.isna(match_date):
            print(f"  ⚠ Skipping {match.get('game_id')} - invalid date")
            continue

        # Home form
        if ht in team_hist:
            h_vals = get_last_n_before_date(team_hist[ht], match_date, n=5)
            h_form = weighted_form(h_vals)
        else:
            h_form = float("nan")
            missing_teams.add(ht)

        # Away form
        if at in team_hist:
            a_vals = get_last_n_before_date(team_hist[at], match_date, n=5)
            a_form = weighted_form(a_vals)
        else:
            a_form = float("nan")
            missing_teams.add(at)

        rows.append({
            "game_id": match["game_id"],
            "Date": match["Date"],
            "Home Team": ht,
            "Away Team": at,
            "home_form": h_form,
            "away_form": a_form,
        })

    if missing_teams:
        # Filter to actual string team names (drop NaN/float artifacts)
        missing_teams_clean = {t for t in missing_teams if isinstance(t, str) and t.strip()}
        if missing_teams_clean:
            print(f"\n⚠ {len(missing_teams_clean)} teams not found in xG history:")
            for t in sorted(missing_teams_clean)[:15]:
                print(f"  - {t}")
            if len(missing_teams_clean) > 15:
                print(f"  ... and {len(missing_teams_clean) - 15} more")

    form_df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 5. Report and save
    # ------------------------------------------------------------------
    total = len(form_df)
    home_forms = form_df["home_form"].notna().sum()
    away_forms = form_df["away_form"].notna().sum()

    print(f"\n{'='*60}")
    print(f"XG FORM COMPUTATION SUMMARY (upcoming matches)")
    print(f"{'='*60}")
    print(f"Total upcoming matches:   {total}")
    print(f"Home form computed:       {home_forms}/{total}")
    print(f"Away form computed:       {away_forms}/{total}")

    if total > 0:
        sample = form_df.head(5)
        print(f"\nSample rows:")
        for _, r in sample.iterrows():
            print(f"  {r['Date']} {r['Home Team']} vs {r['Away Team']}")
            print(f"    Home form: {r['home_form']:.2f}, Away form: {r['away_form']:.2f}")

    form_df.to_csv(form_out, index=False, encoding="utf-8-sig")
    print(f"\n[OK] Saved upcoming xG form to: {form_out}")


if __name__ == "__main__":
    main()