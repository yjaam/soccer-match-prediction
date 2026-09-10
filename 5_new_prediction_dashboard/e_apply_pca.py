#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Apply the training-time player-statistics PCA to UPCOMING matches.

Uses the exact PCA transformation weights that were fit during training
(misc/player_statistics_pca_weights.json) so the new match projections are
in the same latent space as the training data.

Pipeline:
1. Load upcoming lineups_values_ratings (from c_ script)
2. Apply the same LOCF imputation as training (per-team-side, using local history)
3. Apply the training PCA weights per position group
4. Save the resulting per-group PC1 scores for the fusion model.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# Training-time artifacts (read-only)
MISC_DIR = PROJECT_DIR / "misc"
WEIGHTS_PATH = MISC_DIR / "player_statistics_pca_weights.json"

# Dashboard-local data
PREDICTION_DATA_DIR = SCRIPT_DIR / "prediction_data"
INPUT_PATH = PREDICTION_DATA_DIR / "upcoming_lineups_values_ratings.csv"
OUTPUT_PATH = PREDICTION_DATA_DIR / "upcoming_player_statistics_pca.csv"


def load_weights():
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"PCA weights not found at {WEIGHTS_PATH}.\n"
            f"Re-run the training player-statistics PCA script to regenerate them."
        )
    return json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))


def resolve_date_column(frame):
    for candidate in ("Date_x", "Date", "date"):
        if candidate in frame.columns:
            return candidate
    raise KeyError("No date column found.")


def resolve_team_columns(frame):
    home_candidates = ["home_team_code", "Home Team", "home_club_id", "HomeTeam"]
    away_candidates = ["away_team_code", "Away Team", "away_club_id", "AwayTeam"]
    home_column = next((col for col in home_candidates if col in frame.columns), None)
    away_column = next((col for col in away_candidates if col in frame.columns), None)
    if home_column is None or away_column is None:
        raise KeyError("Could not resolve home/away team columns.")
    return {"home": home_column, "away": away_column}


def locf_impute_upcoming(
    frame,
    feature_columns,
    team_columns,
    date_column,
    fallback_means,
):
    """Forward-fill within team groups (chronological). Then fill with training means.

    For upcoming matches alone, LOCF won't do anything within a single row.
    So we rely on the training fallback means for any missing values.
    """
    imputed = frame.copy()
    imputed[date_column] = pd.to_datetime(imputed[date_column], errors="coerce")
    imputed = imputed.sort_values([date_column, "game_id"]).reset_index(drop=True)

    for column in feature_columns:
        imputed[column] = pd.to_numeric(imputed[column], errors="coerce")
        fallback = fallback_means.get(column, 0.0)
        imputed[column] = imputed[column].fillna(fallback)
        if imputed[column].isna().any():
            imputed[column] = imputed[column].fillna(0.0)

    return imputed


def project_group(frame, group_data):
    """Project a group's features through the saved PCA component."""
    cols = group_data["feature_columns"]
    means = pd.Series(group_data["means"])
    stds = pd.Series(group_data["stds"])
    component = np.array(group_data["component"], dtype=float)

    # Align to saved feature order
    matrix = frame[cols].copy()
    for c in cols:
        matrix[c] = pd.to_numeric(matrix[c], errors="coerce")

    scaled = ((matrix - means) / stds).fillna(0.0).to_numpy(dtype=float)
    scores = scaled @ component.T
    return scores


def main():
    if not INPUT_PATH.exists():
        print(f"Error: input not found at {INPUT_PATH}")
        print("Run c_get_upcoming_fifa_ratings.py first.")
        return

    weights = load_weights()
    print(f"Loaded PCA weights from {WEIGHTS_PATH}")

    df = pd.read_csv(INPUT_PATH, low_memory=False)
    print(f"Loaded {len(df)} upcoming matches")

    if "game_id" not in df.columns:
        raise KeyError("game_id column is required in upcoming data.")

    date_column = resolve_date_column(df)
    team_columns = resolve_team_columns(df)

    # Collect all features referenced by any group
    all_features = sorted({
        c for g in weights["groups"].values() for c in g["feature_columns"]
    })

    # Apply the same fallback strategy used during training
    fallback_means = weights.get("locf_fallback_means", {})
    df = locf_impute_upcoming(df, all_features, team_columns, date_column, fallback_means)

    # Project each group through the training PCA
    output = pd.DataFrame({"game_id": df["game_id"].values})

    print("\nProjecting each position group through training PCA:")
    for group_key, group_data in weights["groups"].items():
        scores = project_group(df, group_data)
        output[group_key] = scores
        print(f"  {group_key}: {len(group_data['feature_columns'])} features → 1 PC")

    # Keep metadata for downstream merging
    output["Date"] = df[date_column].values
    output["Home Team"] = df[team_columns["home"]].values
    output["Away Team"] = df[team_columns["away"]].values

    output.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\n{'='*60}")
    print(f"PLAYER STATISTICS PCA (upcoming)")
    print(f"{'='*60}")
    print(f"Matches processed:     {len(output)}")
    print(f"Groups projected:      {len(weights['groups'])}")
    print(f"Output columns:        {list(output.columns)}")
    print(f"Saved to:              {OUTPUT_PATH}")

    # Show a quick preview
    print(f"\nPreview:")
    print(output.head(3).to_string(index=False))


if __name__ == "__main__":
    main()