#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Prepare per-module TRAIN/TEST inputs for independent neural training.

Outputs:
- final_data/nn_input_lineup_TRAIN.csv, final_data/nn_input_lineup_TEST.csv
- final_data/nn_input_player_stats_TRAIN.csv, final_data/nn_input_player_stats_TEST.csv
- final_data/nn_input_skill_matchups_TRAIN.csv, final_data/nn_input_skill_matchups_TEST.csv
- misc/nn_module_metadata.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
FINAL_DATA_DIR = PROJECT_DIR / "final_data"
DATA_DIR = PROJECT_DIR / "data"
MISC_DIR = PROJECT_DIR / "misc"

META_OUTPUT_PATH = MISC_DIR / "nn_module_metadata.json"

MODULE_SPECS = {
    "lineup": {
        "train": FINAL_DATA_DIR / "lineup_embeddings_TRAIN.csv",
        "test": FINAL_DATA_DIR / "lineup_embeddings_TEST.csv",
        "key": "game_id",
        "label_source": DATA_DIR / "target_variables.csv",
        "label_key": "game_id",
    },
    "player_stats": {
        "train": FINAL_DATA_DIR / "player_statistics_pca_TRAIN.csv",
        "test": FINAL_DATA_DIR / "player_statistics_pca_TEST.csv",
        "key": "game_id",
        "label_source": None,
        "label_key": None,
    },
    "skill_matchups": {
        "train": FINAL_DATA_DIR / "skill_matchups_TRAIN.csv",
        "test": FINAL_DATA_DIR / "skill_matchups_TEST.csv",
        "key": "game_id",
        "label_source": None,
        "label_key": None,
    },
    "match_stats": {
        "train": FINAL_DATA_DIR / "match_statistics_pca_TRAIN.csv",
        "test": FINAL_DATA_DIR / "match_statistics_pca_TEST.csv",
        "key": "game_id",
        "label_source": None,
        "label_key": None,
    },
}


def _load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path, low_memory=False)


def _add_labels_if_needed(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    label_source = spec.get("label_source")
    label_key = spec.get("label_key")
    key_col = spec["key"]

    if label_source is None:
        return df

    labels = _load_frame(label_source)
    required = {label_key, "HG", "AG"}
    if not required.issubset(labels.columns):
        raise KeyError(f"Label source {label_source} missing required columns: {required}")

    out = df.copy()
    for col in ["HG", "AG"]:
        if col in out.columns:
            out = out.drop(columns=[col])

    out = out.merge(labels[[label_key, "HG", "AG"]], left_on=key_col, right_on=label_key, how="left")
    if label_key != key_col and label_key in out.columns:
        out = out.drop(columns=[label_key])

    return out


def _coerce_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["HG"] = pd.to_numeric(out["HG"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    out["AG"] = pd.to_numeric(out["AG"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    return out


def _clean_module(
    module_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    key_col: str,
    max_goal_value: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict]:
    required = {key_col, "HG", "AG"}
    if not required.issubset(train_df.columns) or not required.issubset(test_df.columns):
        missing_train = sorted(required.difference(train_df.columns))
        missing_test = sorted(required.difference(test_df.columns))
        raise KeyError(
            f"{module_name} missing required columns. train missing={missing_train}, test missing={missing_test}"
        )

    train = _coerce_labels(train_df)
    test = _coerce_labels(test_df)

    before_train = len(train)
    before_test = len(test)

    train = train.dropna(subset=["HG", "AG"]).copy()
    test = test.dropna(subset=["HG", "AG"]).copy()

    train = train.loc[train["HG"].between(0, max_goal_value) & train["AG"].between(0, max_goal_value)].copy()
    test = test.loc[test["HG"].between(0, max_goal_value) & test["AG"].between(0, max_goal_value)].copy()

    feature_candidates = [c for c in train.columns if c not in {key_col, "HG", "AG"}]

    # Prevent identifier leakage/corruption from entering model features.
    id_like_cols = {
        c
        for c in feature_candidates
        if c == "match_id" or c.endswith("_id") or c.endswith("Id") or c.endswith("ID")
    }
    feature_candidates = [c for c in feature_candidates if c not in id_like_cols]
    feature_cols = []
    for col in feature_candidates:
        train[col] = pd.to_numeric(train[col], errors="coerce")
        test[col] = pd.to_numeric(test[col], errors="coerce")
        if train[col].notna().sum() > 0:
            feature_cols.append(col)

    if not feature_cols:
        raise ValueError(f"No usable numeric features found for module '{module_name}'.")

    medians = train[feature_cols].median(numeric_only=True)
    train[feature_cols] = train[feature_cols].fillna(medians)
    test[feature_cols] = test[feature_cols].fillna(medians)

    train = train.replace([np.inf, -np.inf], np.nan)
    test = test.replace([np.inf, -np.inf], np.nan)
    train[feature_cols] = train[feature_cols].fillna(medians)
    test[feature_cols] = test[feature_cols].fillna(medians)

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
    }

    return train, test, feature_cols, stats


def main() -> None:
    FINAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MISC_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PREPARING PER-MODULE INPUTS")
    print("=" * 70)

    meta = {"modules": {}}

    for module_name, spec in MODULE_SPECS.items():
        key_col = spec["key"]
        train_raw = _load_frame(spec["train"])
        test_raw = _load_frame(spec["test"])

        train_raw = _add_labels_if_needed(train_raw, spec)
        test_raw = _add_labels_if_needed(test_raw, spec)

        train_clean, test_clean, feature_cols, stats = _clean_module(
            module_name=module_name,
            train_df=train_raw,
            test_df=test_raw,
            key_col=key_col,
            max_goal_value=20,
        )

        train_out = FINAL_DATA_DIR / f"nn_input_{module_name}_TRAIN.csv"
        test_out = FINAL_DATA_DIR / f"nn_input_{module_name}_TEST.csv"
        train_clean.to_csv(train_out, index=False)
        test_clean.to_csv(test_out, index=False)

        meta["modules"][module_name] = {
            "key_col": key_col,
            "feature_cols": feature_cols,
            "train_file": str(train_out),
            "test_file": str(test_out),
            **stats,
        }

        print(f"[{module_name}] TRAIN shape: {train_clean.shape} | TEST shape: {test_clean.shape}")
        print(f"[{module_name}] Saved: {train_out.name}, {test_out.name}")

    META_OUTPUT_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved module metadata to {META_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
