#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Derive longitudinal (sequence) inputs from the per-module nn_input files.

Run after a_prepare_data.py. For every module and every game the script builds
history sequences from strictly earlier games (game_id = YYYYMMDD_HOME_AWAY):

  home_form    last K games of the home team (all seasons)
  away_form    last K games of the away team (all seasons)
  home_season  last K games of the home team in the current season
  away_season  last K games of the away team in the current season
  h2h          last K meetings of the two teams (all seasons, either venue)

Each step is the module's feature row of that earlier game plus 5 context dims:
is_home (team of the series was home), goals_for, goals_against, days_ago, valid.
Sequences are ordered oldest -> newest and left-padded with zeros (valid = 0).
The test split may use train games as history, never the other way round in time.

Outputs (same layout as the nn_input files: game_id, HG, AG, features):
- final_data/lstm_input_<module>_TRAIN.csv / _TEST.csv
- misc/lstm_module_metadata.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
FINAL_DATA_DIR = PROJECT_DIR / "final_data"
MISC_DIR = PROJECT_DIR / "misc"
SRC_META_PATH = MISC_DIR / "nn_module_metadata.json"
META_OUTPUT_PATH = MISC_DIR / "lstm_module_metadata.json"

SEQ_LEN = 5
SERIES = ["home_form", "away_form", "home_season", "away_season", "h2h"]
CONTEXT_DIMS = ["is_home", "goals_for", "goals_against", "days_ago", "valid"]


def _parse_ids(game_ids: pd.Series):
    parts = game_ids.astype(str).str.split("_", n=2, expand=True)
    dates = pd.to_datetime(parts[0], format="%Y%m%d")
    return dates, parts[1].to_numpy(), parts[2].to_numpy()


def _season_of(dates: pd.Series) -> np.ndarray:
    return (dates.dt.year - (dates.dt.month < 8).astype(int)).to_numpy()


def build_sequences(df: pd.DataFrame, feature_cols: list[str], seq_len: int) -> np.ndarray:
    """Return array (N, n_series, seq_len, len(feature_cols) + 5) aligned with df rows."""
    n = len(df)
    n_feat = len(feature_cols)
    step_dim = n_feat + len(CONTEXT_DIMS)

    dates, home, away = _parse_ids(df["game_id"])
    day = (dates.astype("int64") // 86_400_000_000_000).to_numpy()
    season = _season_of(dates)
    hg = df["HG"].to_numpy(dtype=np.float32)
    ag = df["AG"].to_numpy(dtype=np.float32)
    feats = df[feature_cols].to_numpy(dtype=np.float32)

    # Chronological row indices per team and per team pair.
    order = np.lexsort((df["game_id"].to_numpy(), day))
    team_idx: dict[str, list[int]] = {}
    pair_idx: dict[tuple, list[int]] = {}
    for i in order:
        team_idx.setdefault(home[i], []).append(i)
        team_idx.setdefault(away[i], []).append(i)
        pair_idx.setdefault(tuple(sorted((home[i], away[i]))), []).append(i)
    team_idx = {k: np.array(v) for k, v in team_idx.items()}
    pair_idx = {k: np.array(v) for k, v in pair_idx.items()}

    out = np.zeros((n, len(SERIES), seq_len, step_dim), dtype=np.float32)

    def fill(i, s, team, rows):
        rows = rows[-seq_len:]
        if len(rows) == 0:
            return
        k0 = seq_len - len(rows)
        block = out[i, s, k0:]
        block[:, :n_feat] = feats[rows]
        is_home = (home[rows] == team)
        block[:, n_feat] = is_home
        block[:, n_feat + 1] = np.where(is_home, hg[rows], ag[rows])
        block[:, n_feat + 2] = np.where(is_home, ag[rows], hg[rows])
        block[:, n_feat + 3] = np.log1p(day[i] - day[rows]) / 8.0
        block[:, n_feat + 4] = 1.0

    for i in range(n):
        for s, team in ((0, home[i]), (1, away[i])):
            idx = team_idx[team]
            prior = idx[day[idx] < day[i]]
            fill(i, s, team, prior)
            fill(i, s + 2, team, prior[season[prior] == season[i]])
        pidx = pair_idx[tuple(sorted((home[i], away[i])))]
        fill(i, 4, home[i], pidx[day[pidx] < day[i]])
    return out


def main() -> None:
    meta_src = json.loads(SRC_META_PATH.read_text(encoding="utf-8"))["modules"]
    meta_out = {"modules": {}, "seq_len": SEQ_LEN, "series": SERIES, "context_dims": CONTEXT_DIMS}

    for module_name, m in meta_src.items():
        feature_cols = m["feature_cols"]
        train = pd.read_csv(FINAL_DATA_DIR / f"nn_input_{module_name}_TRAIN.csv", low_memory=False)
        test = pd.read_csv(FINAL_DATA_DIR / f"nn_input_{module_name}_TEST.csv", low_memory=False)
        full = pd.concat([train, test], ignore_index=True)

        seq = build_sequences(full, feature_cols, SEQ_LEN)
        n_train = len(train)
        step_dim = seq.shape[-1]
        seq_cols = [f"{s}_t{k}_d{j}" for s in SERIES for k in range(SEQ_LEN) for j in range(step_dim)]
        all_cols = feature_cols + seq_cols

        for split, sl in (("TRAIN", slice(0, n_train)), ("TEST", slice(n_train, None))):
            part = full.iloc[sl]
            flat = np.concatenate(
                [part[feature_cols].to_numpy(dtype=np.float32), seq[sl].reshape(len(part), -1)], axis=1
            )
            out = pd.DataFrame(flat, columns=all_cols)
            out.insert(0, "AG", part["AG"].to_numpy())
            out.insert(0, "HG", part["HG"].to_numpy())
            out.insert(0, "game_id", part["game_id"].to_numpy())
            path = FINAL_DATA_DIR / f"lstm_input_{module_name}_{split}.csv"
            out.to_csv(path, index=False, float_format="%.4g")
            print(f"[{module_name}] {split}: {out.shape} -> {path.name}")

        meta_out["modules"][module_name] = {
            "key_col": m["key_col"],
            "feature_cols": all_cols,
            "current_cols": feature_cols,
            "n_current": len(feature_cols),
            "n_series": len(SERIES),
            "seq_len": SEQ_LEN,
            "step_dim": step_dim,
            "pca_applied": None,
            "train_file": str(FINAL_DATA_DIR / f"lstm_input_{module_name}_TRAIN.csv"),
            "test_file": str(FINAL_DATA_DIR / f"lstm_input_{module_name}_TEST.csv"),
        }

    META_OUTPUT_PATH.write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
    print(f"Saved {META_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
