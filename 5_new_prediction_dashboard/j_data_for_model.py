#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Prepare per-module TRAIN/TEST inputs for the DASHBOARD neural network.

This is the dashboard variant of the original training-time data-prep script.

Difference from training-time script:
- The "TRAIN" side contains ALL historical matches (train + test split from
  the original training run are concatenated). This lets the model learn from
  the full history up to and including the 2025-2026 season.
- The "TEST" side contains the current UPCOMING fixtures, sourced from the
  dashboard's ./prediction_data/upcoming_*.csv files.
- Standardization statistics are fit on the full historical data.

Outputs (all inside the dashboard folder):
- nn_input_data/nn_input_lineup_TRAIN.csv,   nn_input_data/nn_input_lineup_TEST.csv
- nn_input_data/nn_input_player_stats_TRAIN.csv, nn_input_data/nn_input_player_stats_TEST.csv
- nn_input_data/nn_input_skill_matchups_TRAIN.csv, nn_input_data/nn_input_skill_matchups_TEST.csv
- nn_input_data/nn_input_match_stats_TRAIN.csv, nn_input_data/nn_input_match_stats_TEST.csv
- nn_input_data/nn_input_league_TRAIN.csv,   nn_input_data/nn_input_league_TEST.csv
- nn_input_data/nn_input_season_TRAIN.csv,   nn_input_data/nn_input_season_TEST.csv
- nn_input_data/nn_input_matchup_TRAIN.csv,  nn_input_data/nn_input_matchup_TEST.csv
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
TRAINING_TARGET_PATH = TRAINING_DATA_DIR / "target_variables.csv"

# Dashboard-local outputs
NN_INPUT_DATA_DIR = SCRIPT_DIR / "nn_input_data"
META_OUTPUT_PATH = NN_INPUT_DATA_DIR / "nn_module_metadata.json"

# Dashboard-local upcoming inputs (produced by the b/c/d/e/f scripts)
PREDICTION_DATA_DIR = SCRIPT_DIR / "prediction_data"


# ============================================================================
# MODULE SPECS
#
# Each module spec declares:
#   - historical train / test files (from the training-time pipeline)
#   - upcoming file (from the dashboard)
#   - key column name
#   - optional label source
# ============================================================================

MODULE_SPECS = {
    "lineup": {
        "train": TRAINING_FINAL_DATA_DIR / "lineup_embeddings_TRAIN.csv",
        "test": TRAINING_FINAL_DATA_DIR / "lineup_embeddings_TEST.csv",
        "upcoming": PREDICTION_DATA_DIR / "upcoming_lineup_embeddings.csv",
        "key": "game_id",
        "label_source": TRAINING_TARGET_PATH,
        "label_key": "game_id",
    },
    "player_stats": {
        "train": TRAINING_FINAL_DATA_DIR / "player_statistics_pca_TRAIN.csv",
        "test": TRAINING_FINAL_DATA_DIR / "player_statistics_pca_TEST.csv",
        "upcoming": PREDICTION_DATA_DIR / "upcoming_player_statistics_pca.csv",
        "key": "game_id",
        "label_source": None,
        "label_key": None,
    },
    "skill_matchups": {
        "train": TRAINING_FINAL_DATA_DIR / "skill_matchups_TRAIN.csv",
        "test": TRAINING_FINAL_DATA_DIR / "skill_matchups_TEST.csv",
        "upcoming": PREDICTION_DATA_DIR / "upcoming_skill_matchups.csv",
        "key": "game_id",
        "label_source": None,
        "label_key": None,
    },
    "match_stats": {
        "train": TRAINING_FINAL_DATA_DIR / "match_statistics_pca_TRAIN.csv",
        "test": TRAINING_FINAL_DATA_DIR / "match_statistics_pca_TEST.csv",
        "upcoming": PREDICTION_DATA_DIR / "upcoming_match_stats_features.csv",
        "key": "game_id",
        "label_source": None,
        "label_key": None,
    },
    "league": {
        "train": TRAINING_FINAL_DATA_DIR / "league_embeddings_TRAIN.csv",
        "test": TRAINING_FINAL_DATA_DIR / "league_embeddings_TEST.csv",
        "upcoming": PREDICTION_DATA_DIR / "upcoming_league_embeddings.csv",
        "key": "game_id",
        "label_source": None,
        "label_key": None,
    },
    "season": {
        "train": TRAINING_FINAL_DATA_DIR / "season_embeddings_TRAIN.csv",
        "test": TRAINING_FINAL_DATA_DIR / "season_embeddings_TEST.csv",
        "upcoming": PREDICTION_DATA_DIR / "upcoming_season_embeddings.csv",
        "key": "game_id",
        "label_source": None,
        "label_key": None,
    },
    "matchup": {
        "train": TRAINING_FINAL_DATA_DIR / "matchup_embeddings_TRAIN.csv",
        "test": TRAINING_FINAL_DATA_DIR / "matchup_embeddings_TEST.csv",
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


def _add_labels_if_needed(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Ensure HG / AG labels are present. Pulls from label_source if needed."""
    label_source = spec.get("label_source")
    label_key = spec.get("label_key")
    key_col = spec["key"]

    if label_source is None or not label_source.exists():
        return df

    labels = _load_frame(label_source)
    required = {label_key, "HG", "AG"}
    if not required.issubset(labels.columns):
        raise KeyError(f"Label source {label_source} missing required columns: {required}")

    out = df.copy()
    for col in ["HG", "AG"]:
        if col in out.columns:
            out = out.drop(columns=[col])

    out = out.merge(labels[[label_key, "HG", "AG"]],
                    left_on=key_col, right_on=label_key, how="left")
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

    train_df:   full historical data (all seasons up to and including 2025-2026)
    test_df:    upcoming fixtures (no HG/AG labels expected — they'll be filled with NaN)
    """
    required = {key_col}
    if not required.issubset(train_df.columns):
        raise KeyError(f"{module_name} TRAIN missing required columns: {sorted(required - set(train_df.columns))}")

    # Make sure the test frame has the key column
    if key_col not in test_df.columns:
        raise KeyError(f"{module_name} TEST missing required key column '{key_col}'.")

    # Ensure HG/AG columns exist in both frames (test may have none)
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

    # Determine feature columns from the intersection of numeric columns
    feature_candidates = [c for c in train.columns if c not in {key_col, "HG", "AG"}]

    # Drop obvious identifier columns
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
        # Keep the feature only if it's populated in at least one training row
        if train[col].notna().sum() > 0:
            feature_cols.append(col)

    if not feature_cols:
        raise ValueError(f"No usable numeric features found for module '{module_name}'.")

    # Median imputation from training
    medians = train[feature_cols].median(numeric_only=True)
    train[feature_cols] = train[feature_cols].fillna(medians)
    test[feature_cols] = test[feature_cols].fillna(medians)

    # Replace infinities
    train[feature_cols] = train[feature_cols].replace([np.inf, -np.inf], np.nan)
    test[feature_cols] = test[feature_cols].replace([np.inf, -np.inf], np.nan)
    train[feature_cols] = train[feature_cols].fillna(medians)
    test[feature_cols] = test[feature_cols].fillna(medians)

    # Standardize
    if fit_standardization_on_train_only:
        feature_means = train[feature_cols].mean(numeric_only=True)
        feature_stds = train[feature_cols].std(numeric_only=True).replace(0, 1.0).fillna(1.0)
    else:
        combined = pd.concat([train[feature_cols], test[feature_cols]], ignore_index=True)
        feature_means = combined.mean(numeric_only=True)
        feature_stds = combined.std(numeric_only=True).replace(0, 1.0).fillna(1.0)

    train[feature_cols] = (train[feature_cols] - feature_means) / feature_stds
    test[feature_cols] = (test[feature_cols] - feature_means) / feature_stds

    # Reorder and subset columns
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
    print(f"Upcoming inputs:      {PREDICTION_DATA_DIR}")
    print(f"Output directory:     {NN_INPUT_DATA_DIR}")
    print()

    meta = {"modules": {}}

    for module_name, spec in MODULE_SPECS.items():
        print(f"\n[{module_name}]")

        # --- Historical side: concatenate original TRAIN + TEST (all history) ---
        try:
            hist_train = _load_frame(spec["train"])
        except FileNotFoundError as e:
            print(f"  ⚠ Skipping (historical train missing): {e}")
            continue

        hist_test = _load_optional(spec["test"])
        if hist_test is not None:
            hist_full = pd.concat([hist_train, hist_test], ignore_index=True)
            hist_full = hist_full.drop_duplicates(subset=[spec["key"]], keep="first")
            print(f"  Historical rows: {len(hist_train)} (TRAIN) + {len(hist_test)} (TEST) "
                  f"→ {len(hist_full)} combined")
        else:
            hist_full = hist_train
            print(f"  Historical rows: {len(hist_train)} (TRAIN only)")

        # Attach labels if needed
        hist_full = _add_labels_if_needed(hist_full, spec)

        # --- Upcoming side: dashboard-local file ---
        upcoming = _load_optional(spec["upcoming"])
        if upcoming is None:
            print(f"  ⚠ Upcoming file not found: {spec['upcoming']} — skipping module.")
            continue
        print(f"  Upcoming rows:   {len(upcoming)}")

        # Upcoming frames don't have HG/AG — that's fine, they'll be NaN.
        # But we do want the same key column.
        if spec["key"] not in upcoming.columns:
            print(f"  ⚠ Upcoming file missing key column '{spec['key']}' — skipping module.")
            continue

        # --- Clean + standardize ---
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