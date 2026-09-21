#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent  # Go up from 1_player_stats to project root

# FIXED: Add PROJECT_DIR to path, then import from src
sys.path.insert(0, str(PROJECT_DIR))

# Import the FINAL team name mapping
from src.team_name_mapping_FINAL import soccer_teams, resolve_team_name


def weighted_form(values, weights=(5, 4, 3, 2, 1)):
    """
    values should be ordered from most recent to older.
    Computes sum(values[i] * weights[i]) for up to 5 entries.
    """
    if len(values) < 5:
        return np.nan
    return float(sum(v * w for v, w in zip(values[:5], weights)))


def build_team_histories(xg_df: pd.DataFrame):
    """
    Build per-team histories using three-letter team codes.
    team_hist[team_code] = list of (date, xg) sorted ascending by date
    """
    team_hist = {}

    xg_df = xg_df.sort_values("date").copy()

    for _, r in xg_df.iterrows():
        ht = r["home_team_code"]
        at = r["away_team_code"]
        d = r["date"]
        hxg = r["home_xg"]
        axg = r["away_xg"]

        if pd.notna(ht) and pd.notna(hxg):
            team_hist.setdefault(ht, []).append((d, float(hxg)))
        if pd.notna(at) and pd.notna(axg):
            team_hist.setdefault(at, []).append((d, float(axg)))

    # Sort each list by date
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
    # FIXED: Use PROJECT_DIR for all file paths
    lineup_path = os.path.join(PROJECT_DIR, "data", "lineups_values_ratings_games.csv")
    xg_path = os.path.join(PROJECT_DIR, "data", "big5_xg_clean.csv")
    goals_path = os.path.join(PROJECT_DIR, "data", "goals_and_formations.csv")
    forms_out_path = os.path.join(PROJECT_DIR, "data", "player_group_before_pca.csv")

    # -------------------------
    # Load data
    # -------------------------
    print("Loading data...")
    df = pd.read_csv(lineup_path)
    xg = pd.read_csv(xg_path)

    # Ensure dates are datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    xg["date"] = pd.to_datetime(xg["date"], errors="coerce")

    # Drop invalid xg rows
    xg_before = len(xg)
    xg = xg.dropna(subset=["date", "home_team", "away_team"]).copy()
    print(f"Dropped {xg_before - len(xg)} invalid xG rows")
    print(f"xG data shape: {xg.shape}")

    # ============================================================================
    # STEP 1: Map xG team names to three-letter codes
    # ============================================================================
    print("\n--- Step 1: Mapping xG team names to three-letter codes ---")

    # Get unique xG team names
    unique_xg_teams = pd.concat([
        xg["home_team"].dropna(),
        xg["away_team"].dropna()
    ]).unique()

    print(f"Unique xG team names: {len(unique_xg_teams)}")

    # Map each xG team name to a three-letter code
    xg_name_to_code = {}
    unmapped_xg = []

    for team_name in unique_xg_teams:
        code = resolve_team_name(team_name)
        if code:
            xg_name_to_code[team_name] = code
        else:
            unmapped_xg.append(team_name)

    mapped_xg = len(xg_name_to_code)
    print(f"Mapped xG names: {mapped_xg}/{len(unique_xg_teams)}")

    if unmapped_xg:
        print(f"Warning: {len(unmapped_xg)} xG team names could not be mapped:")
        for team in unmapped_xg:
            print(f"  - '{team}'")

        # Try to add these missing mappings to help the user
        print("\nSuggested additions to team_name_mapping_FINAL.py:")
        for team in unmapped_xg:
            # Try to find a similar name already in the mapping
            team_lower = team.lower().strip()
            for existing_name, code in soccer_teams.items():
                if team_lower in existing_name.lower() or existing_name.lower() in team_lower:
                    print(f"  '{team}': '{code}',  # similar to '{existing_name}'")
                    break

    # Apply three-letter codes to xG data
    xg["home_team_code"] = xg["home_team"].map(xg_name_to_code)
    xg["away_team_code"] = xg["away_team"].map(xg_name_to_code)

    # Drop xG rows where we couldn't map the team names
    xg_valid = xg.dropna(subset=["home_team_code", "away_team_code"])
    dropped_xg = len(xg) - len(xg_valid)
    if dropped_xg > 0:
        print(f"Dropped {dropped_xg} xG rows due to unmapped team names")

    # ============================================================================
    # STEP 2: Verify lineup team codes (they should already be three-letter codes)
    # ============================================================================
    print("\n--- Step 2: Verifying lineup team codes ---")

    # Get unique lineup team codes (filter out NaN values)
    unique_lineup_teams = pd.concat([
        df["Home Team"].dropna(),
        df["Away Team"].dropna()
    ]).unique()

    # Filter out any remaining NaN or non-string values
    unique_lineup_teams = [str(t).strip().upper() for t in unique_lineup_teams
                          if pd.notna(t) and str(t).strip()]

    print(f"Unique lineup team codes: {len(unique_lineup_teams)}")

    # Check if all lineup teams are valid three-letter codes in our mapping
    valid_codes = set(soccer_teams.values())
    invalid_lineup_codes = []

    for code in unique_lineup_teams:
        if code.upper() not in valid_codes:
            invalid_lineup_codes.append(code)

    if invalid_lineup_codes:
        print(f"Warning: {len(invalid_lineup_codes)} lineup codes not found in mapping:")
        for code in invalid_lineup_codes[:10]:
            print(f"  - '{code}'")
        if len(invalid_lineup_codes) > 10:
            print(f"  ... and {len(invalid_lineup_codes) - 10} more")
    else:
        print("All lineup team codes are valid!")

    # Store the three-letter codes directly (they're already codes)
    df["home_team_code"] = df["Home Team"].apply(
        lambda x: str(x).strip().upper() if pd.notna(x) else None
    )
    df["away_team_code"] = df["Away Team"].apply(
        lambda x: str(x).strip().upper() if pd.notna(x) else None
    )

    # ============================================================================
    # STEP 3: Build xG histories using three-letter codes
    # ============================================================================
    print("\n--- Step 3: Building team xG histories ---")
    team_hist = build_team_histories(xg_valid)
    print(f"Built histories for {len(team_hist)} unique team codes")

    # Show which teams have histories
    teams_with_history = set(team_hist.keys())

    # Get lineup codes (filtering out None/NaN)
    lineup_codes = set()
    for code in df["home_team_code"].dropna():
        if code:
            lineup_codes.add(code)
    for code in df["away_team_code"].dropna():
        if code:
            lineup_codes.add(code)

    lineup_with_history = lineup_codes & teams_with_history
    lineup_without_history = lineup_codes - teams_with_history

    print(f"Lineup teams with xG history: {len(lineup_with_history)}/{len(lineup_codes)}")
    if lineup_without_history:
        # Filter out any non-string values before sorting
        str_codes = [c for c in lineup_without_history if isinstance(c, str)]
        print(f"Teams without xG history ({len(str_codes)}):")
        for code in sorted(str_codes)[:10]:
            print(f"  - {code}")
        if len(str_codes) > 10:
            print(f"  ... and {len(str_codes) - 10} more")

    # ============================================================================
    # STEP 4: Compute forms
    # ============================================================================
    print("\n--- Step 4: Computing weighted forms ---")
    home_forms = []
    away_forms = []

    # Sort matches by date to mimic temporal flow
    df = df.sort_values("Date").reset_index(drop=True)

    form_computed = 0
    form_missing = 0

    for idx, row in df.iterrows():
        match_date = row["Date"]
        ht = row["home_team_code"]
        at = row["away_team_code"]

        # Home form: last 5 matches xG (total, not just home)
        if pd.notna(ht) and ht in team_hist:
            h_vals = get_last_n_before_date(team_hist[ht], match_date, n=5)
            h_form = weighted_form(h_vals)
        else:
            h_form = np.nan

        # Away form: last 5 matches xG (total, not just away)
        if pd.notna(at) and at in team_hist:
            a_vals = get_last_n_before_date(team_hist[at], match_date, n=5)
            a_form = weighted_form(a_vals)
        else:
            a_form = np.nan

        home_forms.append(h_form)
        away_forms.append(a_form)

        if pd.notna(h_form):
            form_computed += 1
        else:
            form_missing += 1

        if pd.notna(a_form):
            form_computed += 1
        else:
            form_missing += 1

        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(df)} matches...")

    df["home_form"] = home_forms
    df["away_form"] = away_forms

    # ============================================================================
    # STEP 5: Print statistics and save
    # ============================================================================
    total_forms = len(home_forms) + len(away_forms)

    print(f"\n{'='*60}")
    print(f"FORM COMPUTATION STATISTICS")
    print(f"{'='*60}")
    print(f"xG data rows: {len(xg)} (valid: {len(xg_valid)})")
    print(f"Unique xG team names: {len(unique_xg_teams)}")
    print(f"  Successfully mapped to codes: {mapped_xg} ({mapped_xg/len(unique_xg_teams)*100:.1f}%)")
    print(f"  Failed to map: {len(unmapped_xg)}")
    print(f"")
    print(f"Lineup matches processed: {len(df)}")
    print(f"Unique lineup team codes: {len(unique_lineup_teams)}")
    print(f"  Valid codes: {len(unique_lineup_teams) - len(invalid_lineup_codes)}")
    print(f"  Invalid codes: {len(invalid_lineup_codes)}")
    print(f"")
    print(f"Teams with xG history: {len(teams_with_history)}")
    print(f"Lineup teams with history: {len(lineup_with_history)}/{len(lineup_codes)}")
    print(f"")
    print(f"Form values computed: {form_computed}/{total_forms} ({form_computed/total_forms*100:.1f}%)")
    print(f"Form values missing:  {form_missing}/{total_forms} ({form_missing/total_forms*100:.1f}%)")
    print(f"  home_form non-null: {df['home_form'].notna().sum()}")
    print(f"  away_form non-null: {df['away_form'].notna().sum()}")

    # Show sample of computed forms
    if form_computed > 0:
        valid_forms = df[df['home_form'].notna()]
        if len(valid_forms) > 0:
            print(f"\nSample of computed forms (first 5):")
            sample = valid_forms.head(5)
            for _, row in sample.iterrows():
                print(f"  {row['Date'].date()} - {row['home_team_code']} vs {row['away_team_code']}")
                print(f"    Home form: {row['home_form']:.2f}, Away form: {row['away_form']:.2f}")

    # add game_id column as defined before (e.g., "20230812_MUN_BAR")
    print(f"\nAdding game_id column...")
    df['game_id'] = df['Date'].dt.strftime('%Y%m%d') + '_' + df['home_team_code'] + '_' + df['away_team_code']
    cols = df.columns.tolist()
    cols = ['game_id'] + [col for col in cols if col != 'game_id']
    df = df[cols]

    # ------------------------------------------------------------------
    # Merge in HG / AG from goals_and_formations.csv so that downstream
    # scripts (e.g. the PCA step and the LSTM training pipeline) can
    # resolve the ground-truth scoreline via game_id.
    # ------------------------------------------------------------------
    print(f"\nMerging goals from {goals_path}...")
    if os.path.exists(goals_path):
        goals = pd.read_csv(goals_path)

        # Locate the home/away goal columns robustly — different pipeline
        # scripts use different names for the same thing.
        home_goal_candidates = ["HG", "FTHG", "home_goals", "Home_goals", "home_score"]
        away_goal_candidates = ["AG", "FTAG", "away_goals", "Away_goals", "away_score"]

        hg_col = next((c for c in home_goal_candidates if c in goals.columns), None)
        ag_col = next((c for c in away_goal_candidates if c in goals.columns), None)

        if "game_id" not in goals.columns or hg_col is None or ag_col is None:
            print(f"  ⚠ Could not locate game_id + goal columns in {goals_path}.")
            print(f"    Available columns: {list(goals.columns)}")
            print(f"    HG column found:  {hg_col}")
            print(f"    AG column found:  {ag_col}")
            print(f"    → Skipping goal merge; downstream will see missing HG/AG.")
            df["HG"] = np.nan
            df["AG"] = np.nan
        else:
            goals_lookup = (
                goals[["game_id", hg_col, ag_col]]
                .rename(columns={hg_col: "HG", ag_col: "AG"})
                .drop_duplicates(subset=["game_id"], keep="first")
            )
            goals_lookup["HG"] = pd.to_numeric(goals_lookup["HG"], errors="coerce")
            goals_lookup["AG"] = pd.to_numeric(goals_lookup["AG"], errors="coerce")

            before = len(df)
            df = df.merge(goals_lookup, on="game_id", how="left")
            matched = df["HG"].notna().sum()
            print(f"  Matched goals for {matched}/{before} rows "
                  f"({matched / before * 100:.1f}%)")

            if matched < before:
                missing = before - matched
                print(f"  ⚠ {missing} row(s) had no matching game_id in goals file.")
                sample_missing = df[df["HG"].isna()]["game_id"].head(5).tolist()
                if sample_missing:
                    print(f"    Example game_ids without goals: {sample_missing}")

            # Sanity check: are the codes overlapping at all?
            if matched == 0:
                print("  ⚠ Zero overlaps — checking format compatibility...")
                example_df_id = df["game_id"].dropna().iloc[0] if df["game_id"].notna().any() else None
                example_goals_id = goals_lookup["game_id"].dropna().iloc[0] if goals_lookup["game_id"].notna().any() else None
                print(f"    Example lineup game_id:  {example_df_id!r}")
                print(f"    Example goals game_id:   {example_goals_id!r}")
    else:
        print(f"  ⚠ {goals_path} not found — HG/AG will be missing.")
        print(f"    (Downstream scripts that need ground truth will fail.)")
        df["HG"] = np.nan
        df["AG"] = np.nan

    # ------------------------------------------------------------------
    # Final column ordering: game_id first, HG/AG right after so they're
    # easy to eyeball, then everything else.
    # ------------------------------------------------------------------
    preferred_front = ["game_id", "HG", "AG"]
    front = [c for c in preferred_front if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]

    # Save the full forms dataset
    os.makedirs(os.path.dirname(forms_out_path), exist_ok=True)
    df.to_csv(forms_out_path, index=False, encoding="utf-8-sig")

    print(f"\nFinal shape: {df.shape}")
    print(f"  HG non-null: {df['HG'].notna().sum() if 'HG' in df.columns else 0}")
    print(f"  AG non-null: {df['AG'].notna().sum() if 'AG' in df.columns else 0}")

    print(f"\nSaved full forms CSV: {forms_out_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()