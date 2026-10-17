#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Prepare per-module TRAIN/TEST inputs for the DASHBOARD neural network.

This is the dashboard variant of the original training-time data-prep script.

Difference from training-time script:
- The "TRAIN" side contains ALL historical matches (every module's full
  *TRAIN.csv and *TEST.csv are concatenated, so the model always learns from
  everything that has been scraped up to now). No fixed season cutoff is
  used anywhere — if a new module's TRAIN file grows, the historical side
  grows with it automatically.
- The "TEST" side contains the current UPCOMING fixtures, sourced from the
  dashboard's ./prediction_data/upcoming_*.csv files.
- Standardization statistics are fit on the full historical data.

Goals are read from data/big5_matches_fixed.csv (the legacy
target_variables.csv has been removed).

Outputs (all inside the dashboard folder):
- nn_input_data/nn_input_<module>_TRAIN.csv, nn_input_data/nn_input_<module>_TEST.csv
  for every module in MODULE_SPECS that has usable data.
- nn_input_data/nn_module_metadata.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent                 # 5_new_prediction_dashboard
PROJECT_DIR = SCRIPT_DIR.parent                              # project root

# Historical artifacts (read-only, produced by the training pipeline)
TRAINING_FINAL_DATA_DIR = PROJECT_DIR / "final_data"
TRAINING_DATA_DIR = PROJECT_DIR / "data"

# Goals source — big5_matches_fixed.csv replaces the removed target_variables.csv
TRAINING_GOALS_SOURCE_PATH = TRAINING_DATA_DIR / "big5_matches_fixed.csv"

# Dashboard-local outputs
NN_INPUT_DATA_DIR = SCRIPT_DIR / "nn_input_data"
META_OUTPUT_PATH = NN_INPUT_DATA_DIR / "nn_module_metadata.json"

# Dashboard-local upcoming inputs (produced by the b/c/d/e/f scripts)
PREDICTION_DATA_DIR = SCRIPT_DIR / "prediction_data"

# Written by the training scripts h (player_stats) and 2c (match_stats).
PCA_STATUS_PATH = PROJECT_DIR / "misc" / "pca_status.json"


# ============================================================================
# MODULE SPECS
#
# Each module has one or more "historical" sources. Every historical source
# that exists is concatenated and deduplicated on game_id to form the TRAIN
# side. Any single source may be missing without breaking the script.
#
# - The convention *TRAIN.csv + *TEST.csv comes from the original training
#   pipeline (where TEST was a holdout). Both are still useful as historical
#   data because the season they held out is now in the past.
#
# - The `match_stats` module reads from big5_matches_fixed.csv-derived
#   features. Because that file grows as new matches are scraped, the
#   historical side grows automatically with it.
# ============================================================================

MODULE_SPECS = {
    "lineup": {
        "historical": [
            TRAINING_FINAL_DATA_DIR / "lineup_embeddings_TRAIN.csv",
            TRAINING_FINAL_DATA_DIR / "lineup_embeddings_TEST.csv",
        ],
        "upcoming": PREDICTION_DATA_DIR / "upcoming_lineup_embeddings.csv",
        "key": "game_id",
        "label_source": TRAINING_GOALS_SOURCE_PATH,
        "label_key": "game_id",
    },
    "player_stats": {
        "historical": [
            TRAINING_FINAL_DATA_DIR / "player_statistics_pca_TRAIN.csv",
            TRAINING_FINAL_DATA_DIR / "player_statistics_pca_TEST.csv",
        ],
        "upcoming": PREDICTION_DATA_DIR / "upcoming_player_statistics_pca.csv",
        "key": "game_id",
        "label_source": None,
        "label_key": None,
    },
    "skill_matchups": {
        "historical": [
            TRAINING_FINAL_DATA_DIR / "skill_matchups_TRAIN.csv",
            TRAINING_FINAL_DATA_DIR / "skill_matchups_TEST.csv",
        ],
        "upcoming": PREDICTION_DATA_DIR / "upcoming_skill_matchups.csv",
        "key": "game_id",
        "label_source": None,
        "label_key": None,
    },
    "match_stats": {
        # match_statistics_pca_TRAIN / TEST are the two halves of the original
        # historical split. They are concatenated here so the model always
        # sees the full history. When a new run of the pipeline regenerates
        # them, this script picks up the new rows automatically.
        "historical": [
            TRAINING_FINAL_DATA_DIR / "match_statistics_pca_TRAIN.csv",
            TRAINING_FINAL_DATA_DIR / "match_statistics_pca_TEST.csv",
        ],
        "upcoming": PREDICTION_DATA_DIR / "upcoming_match_stats_features.csv",
        "key": "game_id",
        "label_source": None,
        "label_key": None,
    },
    "league": {
        "historical": [
            TRAINING_FINAL_DATA_DIR / "league_embeddings_TRAIN.csv",
            TRAINING_FINAL_DATA_DIR / "league_embeddings_TEST.csv",
        ],
        "upcoming": PREDICTION_DATA_DIR / "upcoming_league_embeddings.csv",
        "key": "game_id",
        "label_source": None,
        "label_key": None,
    },
    "season": {
        "historical": [
            TRAINING_FINAL_DATA_DIR / "season_embeddings_TRAIN.csv",
            TRAINING_FINAL_DATA_DIR / "season_embeddings_TEST.csv",
        ],
        "upcoming": PREDICTION_DATA_DIR / "upcoming_season_embeddings.csv",
        "key": "game_id",
        "label_source": None,
        "label_key": None,
    },
    "matchup": {
        "historical": [
            TRAINING_FINAL_DATA_DIR / "matchup_embeddings_TRAIN.csv",
            TRAINING_FINAL_DATA_DIR / "matchup_embeddings_TEST.csv",
        ],
        "upcoming": PREDICTION_DATA_DIR / "upcoming_matchup_embeddings.csv",
        "key": "game_id",
        "label_source": None,
        "label_key": None,
    },
}


# ============================================================================
# LOADING
# ============================================================================

def _load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path, low_memory=False)


def _load_optional(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, low_memory=False)


def _load_historical_concat(hist_paths: list[Path], key_col: str) -> pd.DataFrame | None:
    """
    Load every historical source that exists, concatenate them, and dedup on
    the key column. Returns None if none of the paths exist.

    Rows from earlier-listed paths win when a game_id appears in more than
    one file (matches the previous behavior where *TRAIN came first).
    """
    frames = []
    found = []
    for p in hist_paths:
        df = _load_optional(p)
        if df is None:
            continue
        if key_col not in df.columns:
            continue
        frames.append(df)
        found.append((p.name, len(df)))

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=[key_col], keep="first")
    return combined, found


def _add_labels_if_needed(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """
    Ensure HG / AG labels are present. Pulls from label_source if needed.

    Auto-detects the home/away goal columns in the source file so it works
    with big5_matches_fixed.csv (Home_goals / Away_goals) as well as any
    legacy file that used HG / AG.
    """
    label_source = spec.get("label_source")
    label_key = spec.get("label_key")
    key_col = spec["key"]

    if label_source is None or not label_source.exists():
        return df

    labels = _load_frame(label_source)

    home_candidates = ["HG", "FTHG", "Home_goals", "home_goals"]
    away_candidates = ["AG", "FTAG", "Away_goals", "away_goals"]

    hg_col = next((c for c in home_candidates if c in labels.columns), None)
    ag_col = next((c for c in away_candidates if c in labels.columns), None)

    if label_key not in labels.columns or hg_col is None or ag_col is None:
        raise KeyError(
            f"Label source {label_source} missing required columns. "
            f"Need '{label_key}' + a home-goal column and an away-goal column. "
            f"Available columns: {list(labels.columns)}"
        )

    labels_norm = (
        labels[[label_key, hg_col, ag_col]]
        .rename(columns={hg_col: "HG", ag_col: "AG"})
    )

    out = df.copy()
    for col in ["HG", "AG"]:
        if col in out.columns:
            out = out.drop(columns=[col])

    out = out.merge(labels_norm, left_on=key_col, right_on=label_key, how="left")
    if label_key != key_col and label_key in out.columns:
        out = out.drop(columns=[label_key])

    return out


def _coerce_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["HG"] = pd.to_numeric(out["HG"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    out["AG"] = pd.to_numeric(out["AG"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    return out


# ============================================================================
# CLEANING + STANDARDIZATION
# ============================================================================

def _clean_module(
    module_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    key_col: str,
    max_goal_value: int,
    fit_standardization_on_train_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict]:
    """
    Clean both TRAIN and TEST frames, then standardize.

    train_df:   full historical data (everything scraped up to now)
    test_df:    upcoming fixtures (no HG/AG labels expected — filled with NaN)
    """
    required = {key_col}
    if not required.issubset(train_df.columns):
        raise KeyError(f"{module_name} TRAIN missing required columns: {sorted(required - set(train_df.columns))}")

    if key_col not in test_df.columns:
        raise KeyError(f"{module_name} TEST missing required key column '{key_col}'.")

    for col in ["HG", "AG"]:
        if col not in train_df.columns:
            train_df = train_df.copy()
            train_df[col] = np.nan
        if col not in test_df.columns:
            test_df = test_df.copy()
            test_df[col] = np.nan

    train = _coerce_labels(train_df)
    test = _coerce_labels(test_df)

    before_train = len(train)
    before_test = len(test)

    # Historical: keep only rows with valid labels in the goal range
    train = train.dropna(subset=["HG", "AG"]).copy()
    train = train.loc[train["HG"].between(0, max_goal_value) &
                      train["AG"].between(0, max_goal_value)].copy()

    # Upcoming: no label filtering — just require a valid key
    test = test.dropna(subset=[key_col]).copy()

    feature_candidates = [c for c in train.columns if c not in {key_col, "HG", "AG"}]

    id_like_cols = {
        c for c in feature_candidates
        if c == "match_id" or c.endswith("_id") or c.endswith("Id") or c.endswith("ID")
    }
    feature_candidates = [c for c in feature_candidates if c not in id_like_cols]

    feature_cols = []
    for col in feature_candidates:
        train[col] = pd.to_numeric(train[col], errors="coerce")
        if col in test.columns:
            test[col] = pd.to_numeric(test[col], errors="coerce")
        else:
            test[col] = np.nan
        if train[col].notna().sum() > 0:
            feature_cols.append(col)

    if not feature_cols:
        raise ValueError(
            f"No usable numeric features found for module '{module_name}'.\n"
            f"    Historical shape (post-cleaning): {train.shape}\n"
            f"    Historical rows kept after label drop: {len(train)}\n"
            f"    Historical columns (first 15): {list(train.columns)[:15]}\n"
            f"    Hint: if rows kept = 0, the module's HG/AG labels could not "
            f"be resolved — check label_source and game_id format."
        )

    medians = train[feature_cols].median(numeric_only=True)
    train[feature_cols] = train[feature_cols].fillna(medians)
    test[feature_cols] = test[feature_cols].fillna(medians)

    train[feature_cols] = train[feature_cols].replace([np.inf, -np.inf], np.nan)
    test[feature_cols] = test[feature_cols].replace([np.inf, -np.inf], np.nan)
    train[feature_cols] = train[feature_cols].fillna(medians)
    test[feature_cols] = test[feature_cols].fillna(medians)

    if fit_standardization_on_train_only:
        feature_means = train[feature_cols].mean(numeric_only=True)
        feature_stds = train[feature_cols].std(numeric_only=True).replace(0, 1.0).fillna(1.0)
    else:
        combined = pd.concat([train[feature_cols], test[feature_cols]], ignore_index=True)
        feature_means = combined.mean(numeric_only=True)
        feature_stds = combined.std(numeric_only=True).replace(0, 1.0).fillna(1.0)

    train[feature_cols] = (train[feature_cols] - feature_means) / feature_stds
    test[feature_cols] = (test[feature_cols] - feature_means) / feature_stds

    train = train[[key_col, "HG", "AG"] + feature_cols].copy()
    test = test[[key_col, "HG", "AG"] + feature_cols].copy()

    train = train.drop_duplicates(subset=[key_col], keep="first").sort_values(key_col).reset_index(drop=True)
    test = test.drop_duplicates(subset=[key_col], keep="first").sort_values(key_col).reset_index(drop=True)

    stats = {
        "rows_train_before": int(before_train),
        "rows_test_before": int(before_test),
        "rows_train_after": int(len(train)),
        "rows_test_after": int(len(test)),
        "feature_count": int(len(feature_cols)),
        "standardized": True,
        "standardization_fit_on": "train_only" if fit_standardization_on_train_only else "train_and_test",
    }
    return train, test, feature_cols, stats


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    NN_INPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("PREPARING DASHBOARD NN INPUTS")
    print("=" * 78)
    print(f"Historical reference: {TRAINING_FINAL_DATA_DIR}")
    print(f"Goals source:         {TRAINING_GOALS_SOURCE_PATH}")
    print(f"Upcoming inputs:      {PREDICTION_DATA_DIR}")
    print(f"Output directory:     {NN_INPUT_DATA_DIR}")
    print()

    meta = {"modules": {}}

    pca_status = {}
    if PCA_STATUS_PATH.exists():
        try:
            pca_status = json.loads(PCA_STATUS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pca_status = {}

    for module_name, spec in MODULE_SPECS.items():
        print(f"\n[{module_name}]")

        hist_paths = spec.get("historical", [])
        loaded = _load_historical_concat(hist_paths, spec["key"])
        if loaded is None:
            print(f"  ⚠ Skipping (no historical source files exist).")
            continue
        hist_full, found = loaded

        for name, n in found:
            print(f"  Historical source: {name:50s} {n:>7} rows")
        print(f"  Historical rows combined (deduped): {len(hist_full)}")

        hist_full = _add_labels_if_needed(hist_full, spec)

        upcoming = _load_optional(spec["upcoming"])
        if upcoming is None:
            print(f"  ⚠ Upcoming file not found: {spec['upcoming']} — skipping module.")
            continue
        print(f"  Upcoming rows:   {len(upcoming)}")

        if spec["key"] not in upcoming.columns:
            print(f"  ⚠ Upcoming file missing key column '{spec['key']}' — skipping module.")
            continue

        try:
            train_clean, test_clean, feature_cols, stats = _clean_module(
                module_name=module_name,
                train_df=hist_full,
                test_df=upcoming,
                key_col=spec["key"],
                max_goal_value=20,
                fit_standardization_on_train_only=True,
            )
        except Exception as e:
            print(f"  ❌ Error during cleaning: {e}")
            continue

        train_out = NN_INPUT_DATA_DIR / f"nn_input_{module_name}_TRAIN.csv"
        test_out = NN_INPUT_DATA_DIR / f"nn_input_{module_name}_TEST.csv"
        train_clean.to_csv(train_out, index=False)
        test_clean.to_csv(test_out, index=False)

        meta["modules"][module_name] = {
            "key_col": spec["key"],
            "feature_cols": feature_cols,
            # None = unknown; k_train_lstm.py only uses PCsInit when this is False.
            "pca_applied": pca_status.get(module_name, {}).get("pca_applied"),
            "train_file": str(train_out),
            "test_file": str(test_out),
            **stats,
        }

        print(f"  Features:        {len(feature_cols)}")
        print(f"  TRAIN shape:     {train_clean.shape}")
        print(f"  TEST shape:      {test_clean.shape}")
        print(f"  Saved:           {train_out.name}, {test_out.name}")

    META_OUTPUT_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n{'='*78}")
    print(f"Saved module metadata to {META_OUTPUT_PATH}")
    print(f"Total modules prepared: {len(meta['modules'])}")
    print(f"{'='*78}")


if __name__ == "__main__":
    main()