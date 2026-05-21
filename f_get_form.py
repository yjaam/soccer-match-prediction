#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

from team_name_resolver import TeamNameResolver

def load_canonical_team_names() -> list[str]:
    """Load the canonical team vocabulary from the xG file.

    The xG CSV already contains the previously established correct team names,
    so this script should resolve lineup names into that vocabulary instead of
    maintaining a separate Bundesliga-specific manual map.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xg_path = os.path.join(script_dir, "data", "big5_xg_clean.csv")
    xg = pd.read_csv(xg_path)

    home_names = xg["home_team_mapped"].dropna().astype(str)
    away_names = xg["away_team_mapped"].dropna().astype(str)
    return sorted(set(home_names).union(set(away_names)))

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
    xg_path = os.path.join(script_dir, "data", "big5_xg_clean.csv")
    forms_out_path = os.path.join(script_dir, "data", "lineups_values_ratings_games_forms.csv")

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

    canonical_team_names = load_canonical_team_names()
    resolver = TeamNameResolver(target_names=canonical_team_names, min_fuzzy_score=82)

    # Map lineup team names into the canonical xG vocabulary.
    df["home_team_mapped"] = df["Home Team"].apply(lambda x: resolver.resolve(x) if pd.notna(x) else x)
    df["away_team_mapped"] = df["Away Team"].apply(lambda x: resolver.resolve(x) if pd.notna(x) else x)

    # Keep the already canonical xG names as-is.
    xg["home_team_mapped"] = xg["home_team"].apply(lambda x: resolver.resolve(x) if pd.notna(x) else x)
    xg["away_team_mapped"] = xg["away_team"].apply(lambda x: resolver.resolve(x) if pd.notna(x) else x)

    # Preserve original names if resolver returns None for unexpected cases.
    df["home_team_mapped"] = df["home_team_mapped"].fillna(df["Home Team"])
    df["away_team_mapped"] = df["away_team_mapped"].fillna(df["Away Team"])
    xg["home_team_mapped"] = xg["home_team_mapped"].fillna(xg["home_team"])
    xg["away_team_mapped"] = xg["away_team_mapped"].fillna(xg["away_team"])

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

    # Remove temporary helpers before writing any pipeline outputs.
    df_full = df.drop(columns=["home_team_mapped", "away_team_mapped"], errors="ignore").copy()

    # Save the full forms dataset as the only output of this stage.
    os.makedirs(os.path.dirname(forms_out_path), exist_ok=True)
    df_full.to_csv(forms_out_path, index=False, encoding="utf-8-sig")

    print(f"Saved full forms CSV: {forms_out_path}")
    print(f"Rows: {len(df_full)}")
    print(f"home_form non-null: {df_full['home_form'].notna().sum()}")
    print(f"away_form non-null: {df_full['away_form'].notna().sum()}")

if __name__ == "__main__":
    main()