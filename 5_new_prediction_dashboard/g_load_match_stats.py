#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Match-statistics features for UPCOMING fixtures (dashboard use).

Pipeline:
1. Load the training feature list (final_data/nn_input_match_stats_TRAIN.csv header)
   so the dashboard produces EXACTLY the same columns as the model expects.
   This automatically excludes features dropped by the training-time coverage filter.
2. Load historical match stats from data/big5_matches_fixed.csv (READ-ONLY).
3. Map historical team names → 3-letter codes via src/team_name_mapping_FINAL.
4. Build per-team, per-side chronological histories (home side / away side).
5. For each upcoming fixture, compute weighted rolling means (weights 5,4,3,2,1)
   over the team's last 5 side-specific matches strictly before the kickoff date.
6. Column naming matches training: Home_<feature> from the home team's home
   history, Away_<feature> from the away team's away history.

Writes only to:
    ./prediction_data/upcoming_match_stats_features.csv
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

HISTORICAL_MATCH_PATH = PROJECT_DIR / "data" / "big5_matches_fixed.csv"
TRAINING_INPUT_PATH = PROJECT_DIR / "final_data" / "nn_input_match_stats_TRAIN.csv"
UPCOMING_LINEUPS_PATH = SCRIPT_DIR / "upcoming_lineups" / "upcoming_lineups_latest.csv"
PREDICTION_DATA_DIR = SCRIPT_DIR / "prediction_data"
OUTPUT_PATH = PREDICTION_DATA_DIR / "upcoming_match_stats_features.csv"

PREDICTION_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Import the project-wide team-name resolver
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.team_name_mapping_FINAL import (  # noqa: E402
    resolve_team_name,
    resolve_team_name_fuzzy,
    soccer_teams,
)


# ============================================================================
# TEAM NAME RESOLUTION
# ============================================================================

NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")


def normalize_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", name)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    text = NON_ALNUM_PATTERN.sub(" ", text)
    return " ".join(text.split())


NORMALIZED_LOOKUP = {}
for team_name, code in soccer_teams.items():
    NORMALIZED_LOOKUP.setdefault(normalize_name(str(team_name)), code)


def resolve_code(name_value) -> str | None:
    """Map any historical team name to a canonical 3-letter code."""
    if pd.isna(name_value):
        return None
    name_text = str(name_value).strip()
    if not name_text:
        return None

    code = resolve_team_name(name_text)
    if code:
        return code

    code = resolve_team_name_fuzzy(name_text)
    if code:
        return code

    return NORMALIZED_LOOKUP.get(normalize_name(name_text))


# ============================================================================
# CONFIGURATION (same as training)
# ============================================================================

K_PAST_MATCHES = 5
WEIGHTS = np.array([5, 4, 3, 2, 1], dtype=float)


# ============================================================================
# HELPERS
# ============================================================================

def load_training_feature_columns() -> list[str]:
    """
    Read the header of nn_input_match_stats_TRAIN.csv to determine which
    feature columns the model expects.

    Excludes metadata columns (game_id, HG, AG, match_id).
    """
    if not TRAINING_INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Training input not found at {TRAINING_INPUT_PATH}. "
            f"Run the training pipeline (a_prepare_data.py) first."
        )

    header = pd.read_csv(TRAINING_INPUT_PATH, nrows=0).columns.tolist()
    metadata = {"game_id", "HG", "AG", "match_id", "Date",
                "Home Team", "Away Team", "league", "season", "url",
                "home_team", "away_team"}
    feature_cols = [c for c in header if c not in metadata]
    return feature_cols


def parse_game_date_from_id(game_id) -> pd.Timestamp:
    parts = str(game_id).split("_", 2)
    if len(parts) < 3:
        return pd.NaT
    return pd.to_datetime(parts[0], format="%Y%m%d", errors="coerce")


def weighted_mean(values: np.ndarray) -> float:
    """Weighted mean with most recent value weighted highest (5,4,3,2,1)."""
    if len(values) == 0:
        return np.nan
    ordered = values[::-1]
    weights = WEIGHTS[: len(ordered)]
    mask = ~np.isnan(ordered)
    if not mask.any():
        return np.nan
    return float(np.sum(ordered[mask] * weights[mask]) / np.sum(weights[mask]))


def build_side_histories(historical: pd.DataFrame, feature_columns: list[str]):
    """
    side_hist[side][team_code][feature] = [(date, value), ...] sorted asc by date.
    Historical long team names are resolved to 3-letter codes first.
    """
    print("\nResolving historical team names to codes...")
    historical = historical.copy()
    historical["home_code"] = historical["home_team"].map(resolve_code)
    historical["away_code"] = historical["away_team"].map(resolve_code)

    n = len(historical)
    h_res = historical["home_code"].notna().sum()
    a_res = historical["away_code"].notna().sum()
    print(f"  Home resolution: {h_res}/{n} ({h_res/n*100:.1f}%)")
    print(f"  Away resolution: {a_res}/{n} ({a_res/n*100:.1f}%)")

    unresolved = set()
    unresolved.update(historical.loc[historical["home_code"].isna(), "home_team"].dropna().unique())
    unresolved.update(historical.loc[historical["away_code"].isna(), "away_team"].dropna().unique())
    unresolved = sorted(str(x) for x in unresolved if str(x).strip())
    if unresolved:
        print(f"  Unresolved names: {len(unresolved)}")
        for name in unresolved[:10]:
            print(f"    - {name}")
        if len(unresolved) > 10:
            print(f"    ... and {len(unresolved) - 10} more")

    historical["_match_date"] = historical["game_id"].apply(parse_game_date_from_id)
    historical = historical.sort_values(["_match_date", "game_id"]).reset_index(drop=True)

    side_hist = {"home": {}, "away": {}}

    for side in ("home", "away"):
        code_col = "home_code" if side == "home" else "away_code"
        for team, group in historical.groupby(code_col, sort=False):
            if pd.isna(team) or str(team).strip() == "":
                continue
            team_dict = {}
            for feat in feature_columns:
                if feat not in group.columns:
                    continue
                team_dict[feat] = [
                    (d, pd.to_numeric(v, errors="coerce"))
                    for d, v in zip(group["_match_date"], group[feat])
                    if pd.notna(d)
                ]
            side_hist[side][team] = team_dict

    return side_hist


def weighted_form_for(team: str, side: str, match_date: pd.Timestamp,
                      side_hist: dict, feature_columns: list[str]) -> dict:
    """Weighted means for one (team, side, before match_date)."""
    out = {feat: np.nan for feat in feature_columns}
    if not team or side not in side_hist or team not in side_hist[side]:
        return out
    th = side_hist[side][team]
    for feat in feature_columns:
        if feat not in th:
            continue
        vals = [v for (d, v) in th[feat] if pd.notna(d) and d < match_date]
        if not vals:
            continue
        window = vals[-K_PAST_MATCHES:]
        out[feat] = weighted_mean(np.array(window, dtype=float))
    return out


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 78)
    print("DASHBOARD: Match-statistics features for upcoming fixtures")
    print("=" * 78)

    # ------------------------------------------------------------------
    # Determine the model's expected feature columns from training input
    # ------------------------------------------------------------------
    print("\nLoading expected feature list from training input header...")
    try:
        training_feature_cols = load_training_feature_columns()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print(f"  Training input defines {len(training_feature_cols)} feature columns")

    home_features = [c for c in training_feature_cols if c.startswith("Home_")]
    away_features = [c for c in training_feature_cols if c.startswith("Away_")]
    other_features = [c for c in training_feature_cols
                      if c not in home_features + away_features]

    print(f"  Layout: {len(home_features)} Home_*, {len(away_features)} Away_*, "
          f"{len(other_features)} other")
    if other_features:
        print(f"  Other columns (grouped with away side, matching training):")
        for c in other_features:
            print(f"    - {c}")

    # The training pipeline groups any non-Home_/Away_ column with the away side
    away_features = away_features + other_features

    all_features = home_features + away_features

    # ------------------------------------------------------------------
    # Load historical data (read-only)
    # ------------------------------------------------------------------
    if not HISTORICAL_MATCH_PATH.exists():
        print(f"\nError: historical file not found at {HISTORICAL_MATCH_PATH}")
        return

    historical = pd.read_csv(HISTORICAL_MATCH_PATH, low_memory=False)
    print(f"\nLoaded historical data: {len(historical)} rows, "
          f"{historical.shape[1]} columns")

    for required in ("game_id", "home_team", "away_team"):
        if required not in historical.columns:
            raise KeyError(f"Historical data must have a '{required}' column.")

    # Check which expected features are actually present historically
    missing_historical = [c for c in all_features if c not in historical.columns]
    if missing_historical:
        print(f"\n⚠ {len(missing_historical)} expected features are NOT in the "
              f"historical file:")
        for c in missing_historical[:15]:
            print(f"    - {c}")
        if len(missing_historical) > 15:
            print(f"    ... and {len(missing_historical) - 15} more")

    # ------------------------------------------------------------------
    # Load upcoming fixtures
    # ------------------------------------------------------------------
    if not UPCOMING_LINEUPS_PATH.exists():
        print(f"\nError: upcoming fixtures not found at {UPCOMING_LINEUPS_PATH}")
        return

    upcoming = pd.read_csv(UPCOMING_LINEUPS_PATH, low_memory=False)
    print(f"\nLoaded upcoming fixtures: {len(upcoming)} rows")

    if "game_id" not in upcoming.columns:
        upcoming = upcoming.copy()
        upcoming["game_id"] = (
            upcoming["Date"].astype(str).str.replace("-", "", regex=False)
            + "_" + upcoming["Home Team"].astype(str)
            + "_" + upcoming["Away Team"].astype(str)
        )

    # ------------------------------------------------------------------
    # Build histories (only for expected features, if present)
    # ------------------------------------------------------------------
    feature_columns_for_history = [c for c in all_features
                                    if c in historical.columns]
    side_hist = build_side_histories(historical, feature_columns_for_history)
    print(f"  Histories built: {len(side_hist['home'])} home teams, "
          f"{len(side_hist['away'])} away teams")

    # ------------------------------------------------------------------
    # Compute per-fixture features
    # ------------------------------------------------------------------
    print("\nComputing weighted features for upcoming fixtures...")
    rows = []
    skipped_invalid = 0
    no_home_history = 0
    no_away_history = 0

    for _, match in upcoming.iterrows():
        match_date = pd.to_datetime(match.get("Date"), errors="coerce")
        ht = match.get("Home Team")
        at = match.get("Away Team")

        if pd.isna(match_date) or pd.isna(ht) or pd.isna(at):
            skipped_invalid += 1
            continue

        ht = str(ht).strip()
        at = str(at).strip()
        if not ht or not at or ht.lower() == "nan" or at.lower() == "nan":
            skipped_invalid += 1
            continue

        home_vals = weighted_form_for(ht, "home", match_date, side_hist, home_features)
        away_vals = weighted_form_for(at, "away", match_date, side_hist, away_features)

        if all(pd.isna(v) for v in home_vals.values()):
            no_home_history += 1
        if all(pd.isna(v) for v in away_vals.values()):
            no_away_history += 1

        row = {
            "game_id": match["game_id"],
            "Date": match["Date"],
            "Home Team": ht,
            "Away Team": at,
        }
        for feat in home_features:
            row[feat] = home_vals.get(feat, np.nan)
        for feat in away_features:
            row[feat] = away_vals.get(feat, np.nan)
        rows.append(row)

    if not rows:
        print("\n⚠ No valid upcoming fixtures produced features.")
        return

    out = pd.DataFrame(rows)

    # Ensure output has exactly the same columns as training (in the same order),
    # plus the metadata columns that downstream scripts rely on
    metadata_cols = ["game_id", "Date", "Home Team", "Away Team"]
    expected_output_cols = metadata_cols + all_features
    for col in expected_output_cols:
        if col not in out.columns:
            out[col] = np.nan
    out = out[expected_output_cols]

    feature_out_cols = all_features
    total_cells = len(out) * len(feature_out_cols)
    nan_cells = int(out[feature_out_cols].isna().sum().sum())

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Upcoming fixtures read:               {len(upcoming)}")
    print(f"Skipped (invalid date/team):          {skipped_invalid}")
    print(f"Fixtures w/ NO home history:          {no_home_history}")
    print(f"Fixtures w/ NO away history:          {no_away_history}")
    print(f"Feature rows produced:                {len(out)}")
    print(f"Feature columns:                      {len(feature_out_cols)} "
          f"(same as training input: {len(training_feature_cols)})")
    print(f"Missing feature cells:                {nan_cells}/{total_cells} "
          f"({nan_cells/total_cells*100:.1f}%)")

    # Column alignment check
    if set(feature_out_cols) == set(training_feature_cols):
        print(f"\n✅ Feature columns exactly match training input")
    else:
        only_in_training = sorted(set(training_feature_cols) - set(feature_out_cols))
        only_in_dashboard = sorted(set(feature_out_cols) - set(training_feature_cols))
        if only_in_training:
            print(f"\n⚠ {len(only_in_training)} columns in training but not in "
                  f"dashboard output:")
            for c in only_in_training[:10]:
                print(f"    - {c}")
        if only_in_dashboard:
            print(f"\n⚠ {len(only_in_dashboard)} columns in dashboard but not in "
                  f"training:")
            for c in only_in_dashboard[:10]:
                print(f"    - {c}")

    # Show sample
    out["_nan"] = out[feature_out_cols].isna().sum(axis=1)
    print(f"\nSample fixtures (NaN count per row):")
    for _, r in out.head(5).iterrows():
        print(f"  {r['Date']} {r['Home Team']} vs {r['Away Team']} — "
              f"{r['_nan']}/{len(feature_out_cols)} NaN")
    out = out.drop(columns=["_nan"])

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n[OK] Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()