#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# Team name mapping: lineup name -> Understat name
# Extend this dictionary if needed.
# -------------------------------------------------------------------
TEAM_NAME_MAP = {
    "Bayern Munich": "Bayern Munich",
    "FC Bayern München": "Bayern Munich",
    "Bayer 04 Leverkusen": "Bayer Leverkusen",
    "RB Leipzig": "RB Leipzig",
    "Borussia Dortmund": "Borussia Dortmund",
    "Eintracht Frankfurt": "Eintracht Frankfurt",
    "SC Freiburg": "SC Freiburg",
    "FSV Mainz 05": "Mainz 05",
    "1. FSV Mainz 05": "Mainz 05",
    "VfL Wolfsburg": "Wolfsburg",
    "Borussia Mönchengladbach": "B. Monchengladbach",
    "TSG Hoffenheim": "Hoffenheim",
    "TSG 1899 Hoffenheim": "Hoffenheim",
    "FC Augsburg": "Augsburg",
    "VfB Stuttgart": "Stuttgart",
    "Werder Bremen": "Werder Bremen",
    "SV Werder Bremen": "Werder Bremen",
    "Union Berlin": "Union Berlin",
    "1. FC Union Berlin": "Union Berlin",
    "VfL Bochum": "Bochum",
    "VfL Bochum 1848": "Bochum",
    "1. FC Köln": "FC Koln",
    "FC Köln": "FC Koln",
    "Heidenheim": "Heidenheim",
    "1. FC Heidenheim": "Heidenheim",
    "FC St. Pauli": "St. Pauli",
    "Holstein Kiel": "Holstein Kiel",
    # fallback: identity mapping handled in function
}

def map_team_name(name: str) -> str:
    if pd.isna(name):
        return name
    n = str(name).strip()
    return TEAM_NAME_MAP.get(n, n)

def weighted_form(values, weights=(5,4,3,2,1)):
    """
    values should be ordered from most recent to older.
    Computes sum(values[i] * weights[i]) for up to 5 entries.
    """
    if len(values) < 5:
        return np.nan
    return float(sum(v * w for v, w in zip(values[:5], weights)))

def build_team_histories(xg_df: pd.DataFrame):
    """
    Build per-team histories:
      team_hist[team] = list of (date, xg) sorted ascending by date
    """
    team_hist = {}

    xg_df = xg_df.sort_values("date").copy()

    for _, r in xg_df.iterrows():
        ht = r["home_team_mapped"]
        at = r["away_team_mapped"]
        d  = r["date"]
        hxg = r["home_xg"]
        axg = r["away_xg"]

        if pd.notna(ht) and pd.notna(hxg):
            team_hist.setdefault(ht, []).append((d, float(hxg)))
        if pd.notna(at) and pd.notna(axg):
            team_hist.setdefault(at, []).append((d, float(axg)))

    # Sort each list by date just in case
    for team in team_hist:
        team_hist[team].sort(key=lambda x: x[0])

    return team_hist

def get_last_n_before_date(history_list, current_date, n=5):
    """
    history_list: list of (date, value), ascending by date.
    Returns up to n most recent values strictly before current_date, in recency order.
    """
    prev = [val for (d, val) in history_list if d < current_date]
    if not prev:
        return []
    # most recent first
    prev = prev[::-1]
    return prev[:n]

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    lineup_path = os.path.join(script_dir, "data", "lineups_values_ratings_games.csv")
    xg_path = os.path.join(script_dir, "data", "bundesliga_xg_clean.csv")
    out_path = os.path.join(script_dir, "data", "data_for_pca.csv")

    # -------------------------
    # Load data
    # -------------------------
    df = pd.read_csv(lineup_path)
    xg = pd.read_csv(xg_path)

    # Ensure dates are datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    xg["date"] = pd.to_datetime(xg["date"], errors="coerce")

    # Drop invalid xg rows
    xg = xg.dropna(subset=["date", "home_team", "away_team"]).copy()

    # Map team names on both sides
    # lineup columns expected: "Home Team", "Away Team"
    df["home_team_mapped"] = df["Home Team"].apply(map_team_name)
    df["away_team_mapped"] = df["Away Team"].apply(map_team_name)

    xg["home_team_mapped"] = xg["home_team"].apply(map_team_name)
    xg["away_team_mapped"] = xg["away_team"].apply(map_team_name)

    # Build xG histories
    team_hist = build_team_histories(xg)

    # -------------------------
    # Compute forms
    # -------------------------
    home_forms = []
    away_forms = []

    # Sort matches by date to mimic temporal flow
    df = df.sort_values("Date").reset_index(drop=True)

    for _, row in df.iterrows():
        match_date = row["Date"]
        ht = row["home_team_mapped"]
        at = row["away_team_mapped"]

        # home form: last 5 matches xG (total, not just home)
        h_hist = team_hist.get(ht, [])
        h_vals = get_last_n_before_date(h_hist, match_date, n=5)
        h_form = weighted_form(h_vals)

        # away form: last 5 matches xG (total, not just away)
        a_hist = team_hist.get(at, [])
        a_vals = get_last_n_before_date(a_hist, match_date, n=5)
        a_form = weighted_form(a_vals)

        home_forms.append(h_form)
        away_forms.append(a_form)

    df["home_form"] = home_forms
    df["away_form"] = away_forms

    # Cleanup helper cols
    df = df.drop(columns=["home_team_mapped", "away_team_mapped"], errors="ignore")

    # Save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {out_path}")
    print(f"Rows: {len(df)}")
    print(f"home_form non-null: {df['home_form'].notna().sum()}")
    print(f"away_form non-null: {df['away_form'].notna().sum()}")

if __name__ == "__main__":
    main()