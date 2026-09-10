#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compute skill matchup features for UPCOMING matches (dashboard use).

The training pipeline computes skill matchups from the *most recent
same-fixture match*. For upcoming fixtures, we do the same thing:
1. For each upcoming home_team vs away_team pair, find the last historical
   meeting between those two teams.
2. Reuse the skill matchup features (log-ratios) from that prior meeting.
3. If no prior meeting exists, fall back to the most recent meeting of
   the same fixture pair (via any side) — or, failing that, drop the row.

Reads:
    ./upcoming_lineups/upcoming_lineups_latest.csv
    ./prediction_data/upcoming_lineups_values_ratings.csv
    ../data/player_group_before_pca.csv (historical reference, read-only)

Writes:
    ./prediction_data/upcoming_skill_matchups.csv
"""

import os
import sys
import numpy as np
import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

upcoming_dir = os.path.join(script_dir, "upcoming_lineups")
prediction_data_dir = os.path.join(script_dir, "prediction_data")
os.makedirs(prediction_data_dir, exist_ok=True)

upcoming_path = os.path.join(upcoming_dir, "upcoming_lineups_latest.csv")
historical_path = os.path.join(project_dir, "data", "player_group_before_pca.csv")
output_path = os.path.join(prediction_data_dir, "upcoming_skill_matchups.csv")


# ============================================================================
# FEATURE COMPUTATION (identical to training)
# ============================================================================

def safe_log_ratio(numerator, denominator, eps=1e-6):
    numerator = np.clip(pd.to_numeric(numerator, errors="coerce"), eps, None)
    denominator = np.clip(pd.to_numeric(denominator, errors="coerce"), eps, None)
    return np.log(numerator / denominator)


def build_feature(df, numerator_col, denominator_col):
    return safe_log_ratio(df[numerator_col], df[denominator_col])


REQUIRED_COLUMNS = [
    "HomeAttacker_shooting_Avg",
    "HomeAttacker_pace_Avg",
    "HomeAttacker_dribbling_Avg",
    "HomeMidfielder_overall_Avg",
    "HomeMidfielder_passing_Avg",
    "AwayAttacker_shooting_Avg",
    "AwayAttacker_pace_Avg",
    "AwayAttacker_dribbling_Avg",
    "AwayMidfielder_overall_Avg",
    "AwayMidfielder_passing_Avg",
    "HomeDefender_defending_Avg",
    "HomeDefender_pace_Avg",
    "HomeDefender_physic_Avg",
    "AwayDefender_defending_Avg",
    "AwayDefender_pace_Avg",
    "AwayDefender_physic_Avg",
    "HomeGoalkeeper_goalkeeping_reflexes_Avg",
    "AwayGoalkeeper_goalkeeping_reflexes_Avg",
]


def build_skill_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """Given a frame with required *_Avg columns, produce the 12 log-ratio features."""
    feature_df = pd.DataFrame(index=df.index)

    feature_df["home_attack_vs_away_defense"] = build_feature(
        df, "HomeAttacker_shooting_Avg", "AwayDefender_defending_Avg"
    )
    feature_df["away_attack_vs_home_defense"] = build_feature(
        df, "AwayAttacker_shooting_Avg", "HomeDefender_defending_Avg"
    )
    feature_df["home_attack_vs_away_goalkeeper"] = build_feature(
        df, "HomeAttacker_shooting_Avg", "AwayGoalkeeper_goalkeeping_reflexes_Avg"
    )
    feature_df["away_attack_vs_home_goalkeeper"] = build_feature(
        df, "AwayAttacker_shooting_Avg", "HomeGoalkeeper_goalkeeping_reflexes_Avg"
    )
    feature_df["home_midfield_vs_away_midfield"] = build_feature(
        df, "HomeMidfielder_overall_Avg", "AwayMidfielder_overall_Avg"
    )
    feature_df["away_midfield_vs_home_midfield"] = build_feature(
        df, "AwayMidfielder_overall_Avg", "HomeMidfielder_overall_Avg"
    )
    feature_df["home_midfield_vs_away_defense"] = build_feature(
        df, "HomeMidfielder_passing_Avg", "AwayDefender_defending_Avg"
    )
    feature_df["away_midfield_vs_home_defense"] = build_feature(
        df, "AwayMidfielder_passing_Avg", "HomeDefender_defending_Avg"
    )
    feature_df["home_attack_pace_vs_away_defense_pace"] = build_feature(
        df, "HomeAttacker_pace_Avg", "AwayDefender_pace_Avg"
    )
    feature_df["away_attack_pace_vs_home_defense_pace"] = build_feature(
        df, "AwayAttacker_pace_Avg", "HomeDefender_pace_Avg"
    )
    feature_df["home_attack_dribbling_vs_away_defense_physicality"] = build_feature(
        df, "HomeAttacker_dribbling_Avg", "AwayDefender_physic_Avg"
    )
    feature_df["away_attack_dribbling_vs_home_defense_physicality"] = build_feature(
        df, "AwayAttacker_dribbling_Avg", "HomeDefender_physic_Avg"
    )

    return feature_df


# ============================================================================
# MAIN
# ============================================================================

def main():
    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------
    if not os.path.exists(historical_path):
        print(f"Error: historical file not found at {historical_path}")
        return
    if not os.path.exists(upcoming_path):
        print(f"Error: upcoming lineups not found at {upcoming_path}")
        print("Run a_scrape_upcoming_lineups.py first.")
        return

    historical = pd.read_csv(historical_path, low_memory=False)
    upcoming = pd.read_csv(upcoming_path, low_memory=False)

    print(f"Loaded {len(historical)} historical rows and {len(upcoming)} upcoming fixtures")

    # ------------------------------------------------------------------
    # Validate historical columns
    # ------------------------------------------------------------------
    missing_hist = [c for c in REQUIRED_COLUMNS if c not in historical.columns]
    if missing_hist:
        print(f"Error: historical data missing required columns: {missing_hist}")
        return

    if "Date" not in historical.columns:
        raise KeyError("Historical data must have a 'Date' column.")
    if "game_id" not in historical.columns:
        raise KeyError("Historical data must have a 'game_id' column.")

    # Ensure team columns are present
    if not {"Home Team", "Away Team"}.issubset(historical.columns):
        raise KeyError("Historical data must have 'Home Team' and 'Away Team' columns.")

    historical = historical.copy()
    historical["Date"] = pd.to_datetime(historical["Date"], errors="coerce")

    # ------------------------------------------------------------------
    # Ensure game_id exists on upcoming
    # ------------------------------------------------------------------
    if "game_id" not in upcoming.columns:
        upcoming["game_id"] = (
            upcoming["Date"].astype(str).str.replace("-", "", regex=False)
            + "_" + upcoming["Home Team"].astype(str)
            + "_" + upcoming["Away Team"].astype(str)
        )
    upcoming = upcoming.copy()
    upcoming["Date"] = pd.to_datetime(upcoming["Date"], errors="coerce")

    # ------------------------------------------------------------------
    # For each upcoming fixture, find the last historical meeting
    # ------------------------------------------------------------------
    print("\nLooking up last meeting for each upcoming fixture...")
    rows = []
    no_history = 0

    for _, match in upcoming.iterrows():
        ht = match["Home Team"]
        at = match["Away Team"]

        # Find last historical meeting between these two teams (either direction)
        candidates = historical[
            ((historical["Home Team"] == ht) & (historical["Away Team"] == at)) |
            ((historical["Home Team"] == at) & (historical["Away Team"] == ht))
        ]

        if candidates.empty:
            no_history += 1
            print(f"  ⚠ No prior meeting: {ht} vs {at}")
            continue

        last = candidates.sort_values("Date").iloc[-1]
        last_date = last["Date"]

        # Build the required columns frame so we can reuse build_skill_matchup_features.
        # To ensure the sign of the log-ratios corresponds to the *upcoming* fixture,
        # we flip home/away columns if the historical meeting had reversed sides.
        historical_was_reversed = (last["Home Team"] == at)

        if historical_was_reversed:
            # Swap home/away feature columns so ratios are computed from
            # the perspective of the CURRENT upcoming home team.
            swapped = {}
            for col in REQUIRED_COLUMNS:
                if col.startswith("Home"):
                    swapped_col = "Away" + col[len("Home"):]
                elif col.startswith("Away"):
                    swapped_col = "Home" + col[len("Away"):]
                else:
                    swapped_col = col
                swapped[col] = last[swapped_col]

            feature_source = pd.DataFrame([swapped])
        else:
            feature_source = pd.DataFrame([last[REQUIRED_COLUMNS].to_dict()])

        # Compute log-ratio features
        features = build_skill_matchup_features(feature_source).iloc[0].to_dict()

        row = {
            "game_id": match["game_id"],
            "Date": match["Date"],
            "Home Team": ht,
            "Away Team": at,
            "_source_last_meeting_date": last_date,
            "_source_reversed": bool(historical_was_reversed),
        }
        row.update(features)
        rows.append(row)

    if not rows:
        print("\n⚠ No upcoming fixtures had a prior meeting. Nothing to save.")
        return

    out = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"SKILL MATCHUP FEATURES (upcoming fixtures)")
    print(f"{'='*60}")
    print(f"Upcoming fixtures total:         {len(upcoming)}")
    print(f"Fixtures with prior meeting:     {len(out)}")
    print(f"Fixtures without prior meeting:  {no_history}")

    # Show how recent the reference meetings were
    if "_source_last_meeting_date" in out.columns:
        out["_source_last_meeting_date"] = pd.to_datetime(
            out["_source_last_meeting_date"], errors="coerce"
        )
        recent = out["_source_last_meeting_date"].max()
        oldest = out["_source_last_meeting_date"].min()
        print(f"Reference meeting date range:    {oldest.date()} → {recent.date()}")

    # Show a sample
    print(f"\nSample of upcoming fixtures with resolved matchups:")
    preview_cols = ["Date", "Home Team", "Away Team", "_source_last_meeting_date"]
    preview = out[preview_cols].head(5)
    print(preview.to_string(index=False))

    # ------------------------------------------------------------------
    # Save (drop internal columns for cleanliness, keep them optional)
    # ------------------------------------------------------------------
    out_clean = out.drop(columns=[c for c in out.columns if c.startswith("_")])
    out_clean.to_csv(output_path, index=False, encoding="utf-8-sig")

    # Also save a "with metadata" version for audit/debug purposes
    meta_path = os.path.join(prediction_data_dir, "upcoming_skill_matchups_with_meta.csv")
    out.to_csv(meta_path, index=False, encoding="utf-8-sig")

    print(f"\n[OK] Saved clean upcoming skill matchups to: {output_path}")
    print(f"[OK] Saved augmented version (with source meeting info) to: {meta_path}")
    print(f"     Columns: {list(out_clean.columns)}")


if __name__ == "__main__":
    main()