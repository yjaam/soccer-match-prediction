#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Per-module MLP encoders + LSTM fusion, wired for the dashboard.

Structure
---------
1. Train one small MLP per input module. Each maps that module's features
   to (lambda_home, lambda_away). Modules are trained independently, so
   modules with different row coverage don't force a global inner-join at
   training time.
2. Assemble each module's predictions into an ordered sequence
       [(lh_1, la_1, diff_1), ..., (lh_N, la_N, diff_N)]
   and train an LSTM fusion head that outputs the final
   (lambda_home, lambda_away) plus a 3-class outcome logit vector.
3. Save predictions in the format l_prepare_shiny_data.py expects, plus the
   trained state dicts and a JSON summary.

Reads
-----
    ./nn_input_data/nn_input_{module}_{TRAIN,TEST}.csv
    ./nn_input_data/nn_module_metadata.json
    ./lstm_config.yaml  (optional; falls back to lstm_config.json)

Writes
------
    ./weekly_results/lstm_predictions.csv
    ./weekly_results/lstm_model.pt
    ./weekly_results/lstm_summary.json

Prediction coding (also used by the naive baseline)
---------------------------------------------------
    +1  -> HOME_WIN
     0  -> DRAW
    -1  -> AWAY_WIN
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


# ============================================================================
# PATHS
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent          # 5_new_prediction_dashboard
PROJECT_DIR = SCRIPT_DIR.parent

NN_INPUT_DATA_DIR = SCRIPT_DIR / "nn_input_data"
WEEKLY_RESULTS_DIR = SCRIPT_DIR / "weekly_results"

META_PATH = NN_INPUT_DATA_DIR / "nn_module_metadata.json"

MODEL_PATH = WEEKLY_RESULTS_DIR / "lstm_model.pt"
PRED_PATH = WEEKLY_RESULTS_DIR / "lstm_predictions.csv"
SUMMARY_PATH = WEEKLY_RESULTS_DIR / "lstm_summary.json"

CONFIG_YAML_PATH = SCRIPT_DIR / "lstm_config.yaml"
CONFIG_JSON_PATH = SCRIPT_DIR / "lstm_config.json"

DEFAULT_MODULES = ["lineup", "player_stats", "skill_matchups", "match_stats",
                   "league", "season", "matchup"]

# Internal outcome labels (0/1/2). Used only inside this script.
OUTCOME_NAMES = {0: "HOME", 1: "DRAW", 2: "AWAY"}
OUTCOME_LABEL_INTERNAL = {0: "HOME_WIN", 1: "DRAW", 2: "AWAY_WIN"}

# External prediction coding used in the saved CSV, matching the naive
# baseline. l_prepare_shiny_data.py imports this and uses it to map
# predicted_result to a human-readable outcome label.
OUTCOME_LABEL = {-1: "AWAY_WIN", 0: "DRAW", 1: "HOME_WIN"}
INT_TO_SIGNED = np.array([1, 0, -1])   # internal 0/1/2 -> +1/0/-1

ACTUAL_COLS = ["actual_home_goals", "actual_away_goals", "actual_outcome"]


# ============================================================================
# SHARED HELPERS (also imported by l_prepare_shiny_data.py)
# ============================================================================

LATEST_PRED_PATH = WEEKLY_RESULTS_DIR / "weekly_predictions_latest.csv"
HISTORICAL_MATCH_PATH = PROJECT_DIR / "data" / "big5_matches_fixed.csv"
UPCOMING_LINEUPS_PATH = SCRIPT_DIR / "upcoming_lineups" / "upcoming_lineups_latest.csv"

LEAGUE_TAG_MAP = {
    "England Premier League": "premier-league",
    "France Ligue 1": "ligue-1",
    "Germany Bundesliga": "bundesliga",
    "Italy Serie A": "serie-a",
    "Spain La Liga": "laliga",
    "premier-league": "premier-league",
    "ligue-1": "ligue-1",
    "bundesliga": "bundesliga",
    "serie-a": "serie-a",
    "laliga": "laliga",
}

LEAGUE_DISPLAY = {
    "premier-league": "Premier League",
    "bundesliga": "Bundesliga",
    "laliga": "La Liga",
    "serie-a": "Serie A",
    "ligue-1": "Ligue 1",
}


def _normalize_name(name):
    if pd.isna(name):
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def extract_team_codes_from_game_id(game_id):
    parts = str(game_id).split("_")
    if len(parts) >= 3:
        return parts[1], parts[2]
    return None, None


def build_team_to_league_from_history():
    if not HISTORICAL_MATCH_PATH.exists():
        return {}
    try:
        from src.team_name_mapping_FINAL import (
            resolve_team_name, resolve_team_name_fuzzy, soccer_teams,
        )
    except ImportError:
        return {}

    normalized_lookup = {}
    for name, code in soccer_teams.items():
        normalized_lookup.setdefault(_normalize_name(name), code)

    def resolve(name):
        if pd.isna(name):
            return None
        text = str(name).strip()
        code = resolve_team_name(text) or resolve_team_name_fuzzy(text)
        if code:
            return code
        return normalized_lookup.get(_normalize_name(text))

    hist = pd.read_csv(HISTORICAL_MATCH_PATH, low_memory=False)
    if "league" not in hist.columns:
        return {}

    team_to_league = {}
    for _, row in hist.iterrows():
        league_raw = row.get("league")
        if pd.isna(league_raw):
            continue
        league = LEAGUE_TAG_MAP.get(str(league_raw).strip(),
                                    str(league_raw).strip())
        for side in ("home_team", "away_team"):
            code = resolve(row.get(side))
            if code and code not in team_to_league:
                team_to_league[code] = league
    return team_to_league


def load_upcoming_context():
    if not UPCOMING_LINEUPS_PATH.exists():
        return {}, {}
    df = pd.read_csv(UPCOMING_LINEUPS_PATH, low_memory=False)
    team_lookup = {}
    for _, row in df.iterrows():
        for side in ("Home", "Away"):
            code = row.get(f"{side} Team")
            raw = row.get(f"{side} Team Raw")
            if (isinstance(code, str) and isinstance(raw, str)
                    and code.strip() and raw.strip()):
                team_lookup[code.strip()] = raw.strip()
    league_lookup = {}
    if "game_id" in df.columns and "Competition" in df.columns:
        for _, row in df.iterrows():
            gid = row.get("game_id")
            comp = row.get("Competition")
            if (isinstance(gid, str) and isinstance(comp, str)
                    and gid.strip() and comp.strip()):
                league_lookup[gid.strip()] = comp.strip()
    return team_lookup, league_lookup


def resolve_league_for_game(game_id, league_lookup, team_to_league):
    league = league_lookup.get(game_id)
    if league:
        return league
    home_code, away_code = extract_team_codes_from_game_id(game_id)
    for code in (home_code, away_code):
        if code and code in team_to_league:
            return team_to_league[code]
    try:
        from src.game_id_context import infer_league_and_season
        ctx = infer_league_and_season(game_id)
        if isinstance(ctx, dict):
            league = ctx.get("league")
            if league and league.lower() != "unknown":
                return league
    except Exception:
        pass
    return "unknown"


def load_actual_results_from_history(path: Path = HISTORICAL_MATCH_PATH):
    if not path.exists():
        print(f"⚠ {path} not found — cannot populate actual goals.")
        return {}
    hist = pd.read_csv(path, low_memory=False)
    home_candidates = ["HG", "FTHG", "Home_goals", "home_goals"]
    away_candidates = ["AG", "FTAG", "Away_goals", "away_goals"]
    hg_col = next((c for c in home_candidates if c in hist.columns), None)
    ag_col = next((c for c in away_candidates if c in hist.columns), None)
    if "game_id" not in hist.columns or hg_col is None or ag_col is None:
        print(f"⚠ {path} is missing game_id or goal columns.")
        return {}
    lookup = {}
    for _, row in hist.iterrows():
        gid = row.get("game_id")
        if not isinstance(gid, str) or not gid.strip():
            continue
        hg = pd.to_numeric(row.get(hg_col), errors="coerce")
        ag = pd.to_numeric(row.get(ag_col), errors="coerce")
        if pd.isna(hg) or pd.isna(ag):
            continue
        lookup[gid.strip()] = (int(hg), int(ag))
    print(f"  Loaded {len(lookup)} historical results from {path.name}")
    return lookup


def update_actuals_from_results(dated_path: Path,
                                results: dict,
                                overwrite: bool = False) -> int:
    if not dated_path.exists():
        return 0
    try:
        df = pd.read_csv(dated_path)
    except Exception as e:
        print(f"⚠ Could not read {dated_path.name}: {e}")
        return 0
    if df.empty or "game_id" not in df.columns:
        return 0

    if "actual_home_goals" not in df.columns:
        df["actual_home_goals"] = pd.Series([pd.NA] * len(df), dtype="Int64")
    else:
        df["actual_home_goals"] = pd.to_numeric(
            df["actual_home_goals"], errors="coerce"
        ).astype("Int64")
    if "actual_away_goals" not in df.columns:
        df["actual_away_goals"] = pd.Series([pd.NA] * len(df), dtype="Int64")
    else:
        df["actual_away_goals"] = pd.to_numeric(
            df["actual_away_goals"], errors="coerce"
        ).astype("Int64")
    if "actual_outcome" not in df.columns:
        df["actual_outcome"] = pd.Series([pd.NA] * len(df), dtype="string")
    else:
        df["actual_outcome"] = df["actual_outcome"].astype("string")

    already_filled = df["actual_home_goals"].notna()
    updated = 0
    for gid, (hg, ag) in results.items():
        mask = df["game_id"].astype(str) == str(gid)
        if not mask.any():
            continue
        if not overwrite:
            mask = mask & ~already_filled
        if not mask.any():
            continue
        df.loc[mask, "actual_home_goals"] = int(hg)
        df.loc[mask, "actual_away_goals"] = int(ag)
        if hg > ag:
            outcome = "HOME_WIN"
        elif hg == ag:
            outcome = "DRAW"
        else:
            outcome = "AWAY_WIN"
        df.loc[mask, "actual_outcome"] = outcome
        updated += int(mask.sum())
    if updated > 0:
        df.to_csv(dated_path, index=False, na_rep="")
    return updated


def dated_output_paths(base_dir: Path, game_ids=None, run_date=None):
    from datetime import datetime as _dt, timedelta as _td
    if run_date is None:
        run_date = _dt.now()
    dates = []
    for gid in (game_ids or []):
        if not isinstance(gid, str) or len(gid) < 8:
            continue
        try:
            dates.append(_dt.strptime(gid[:8], "%Y%m%d"))
        except ValueError:
            continue
    anchor = min(dates) if dates else run_date
    anchor = anchor - _td(days=anchor.weekday())
    stamp = anchor.strftime("%Y-%m-%d")
    return (
        base_dir / f"weekly_predictions_{stamp}.csv",
        base_dir / "weekly_predictions_latest.csv",
    )


# ============================================================================
# CONFIG
# ============================================================================

DEFAULT_CONFIG = {
    "modules": list(DEFAULT_MODULES),
    "val_fraction": 0.15,

    # Per-module MLP
    "module_hidden1": 0,           # 0 = auto
    "module_hidden2": 0,           # 0 = auto
    "module_hidden_cap": 64,
    "module_dropout": 0.3,
    "module_epochs": 100,
    "module_lr": 5e-4,
    "module_weight_decay": 1e-3,
    "module_batch_size": 512,
    "module_patience": 15,
    "module_val_ratio": 0.1,

    # LSTM fusion
    "lstm_hidden": 32,
    "lstm_layers": 1,
    "lstm_dropout": 0.3,
    "lstm_epochs": 100,
    "lstm_lr": 5e-4,
    "lstm_weight_decay": 1e-3,
    "lstm_batch_size": 512,
    "lstm_patience": 15,
    "lstm_outcome_head_hidden": 32,

    # Loss
    "outcome_weight": 1.0,
    "use_full_class_weights": True,
    "home_weight_scale": 1.0,
    "draw_weight_scale": 1.0,
    "away_weight_scale": 1.0,
    "spread_weight": 0.0,
    "target_std": 0.5,

    # Decision
    "decision_source": "auto",     # auto | rate | classifier | blend
    "draw_margin_grid": {"start": 0.0, "stop": 2.0, "num": 201},
    "blend_sweep": {"start": 0.0, "stop": 1.0, "num": 21},
    "sweep_objective": "balanced_accuracy",

    # Runtime
    "device": "auto",
    "verbose": False,
    "seed": 42,
}


def _deep_merge(base, override):
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _expand_grid(spec):
    if isinstance(spec, dict):
        return tuple(np.round(
            np.linspace(spec["start"], spec["stop"], spec["num"]), 3
        ))
    if isinstance(spec, (list, tuple)):
        return tuple(float(v) for v in spec)
    raise ValueError(f"Invalid grid spec: {spec!r}")


def load_config():
    if CONFIG_YAML_PATH.exists() and _HAS_YAML:
        try:
            user = yaml.safe_load(CONFIG_YAML_PATH.read_text(encoding="utf-8")) or {}
            print(f"[config] Loaded {CONFIG_YAML_PATH.name}")
        except Exception as e:
            print(f"[config] Failed to parse {CONFIG_YAML_PATH.name}: {e}")
            user = {}
    elif CONFIG_JSON_PATH.exists():
        try:
            user = json.loads(CONFIG_JSON_PATH.read_text(encoding="utf-8"))
            print(f"[config] Loaded {CONFIG_JSON_PATH.name}")
        except Exception as e:
            print(f"[config] Failed to parse {CONFIG_JSON_PATH.name}: {e}")
            user = {}
    else:
        user = {}
        print("[config] No config file found; using defaults.")

    merged = _deep_merge(DEFAULT_CONFIG, user)
    merged["draw_margin_grid"] = _expand_grid(merged["draw_margin_grid"])
    merged["blend_sweep"] = _expand_grid(merged["blend_sweep"])
    return merged


class _Cfg:
    """Attribute-access wrapper around the merged config dict."""
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)

    def __repr__(self):
        return f"Cfg({self.__dict__})"


CFG = _Cfg(load_config())


# ============================================================================
# DATA LOADING
# ============================================================================

def load_module_meta(module_name):
    if not META_PATH.exists():
        raise FileNotFoundError(f"Missing metadata file {META_PATH}.")
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    if module_name not in meta.get("modules", {}):
        raise KeyError(f"Module '{module_name}' not found in metadata.")
    return meta["modules"][module_name]


def load_module_frames(module_name):
    """Return (train_df, test_df, key_col, feature_cols) for one module."""
    m = load_module_meta(module_name)
    key_col = m["key_col"]
    feature_cols = m["feature_cols"]

    train_path = NN_INPUT_DATA_DIR / f"nn_input_{module_name}_TRAIN.csv"
    test_path = NN_INPUT_DATA_DIR / f"nn_input_{module_name}_TEST.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing nn_input files for module '{module_name}'.")

    train_df = pd.read_csv(train_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)

    for df in (train_df, test_df):
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    return train_df, test_df, key_col, feature_cols


# ============================================================================
# DATASETS
# ============================================================================

class SimpleDataset(Dataset):
    def __init__(self, frame, feature_cols):
        self.x = torch.tensor(frame[feature_cols].astype(float).values,
                              dtype=torch.float32)
        self.hg = torch.tensor(frame["HG"].astype(float).values, dtype=torch.float32)
        self.ag = torch.tensor(frame["AG"].astype(float).values, dtype=torch.float32)
        self.game_id = frame["game_id"].astype(str).values

    def __len__(self):
        return len(self.game_id)

    def __getitem__(self, idx):
        return {
            "game_id": self.game_id[idx],
            "x": self.x[idx],
            "HG": self.hg[idx],
            "AG": self.ag[idx],
        }


class FusionDataset(Dataset):
    def __init__(self, frame, feature_cols):
        self.x = torch.tensor(frame[feature_cols].astype(float).values,
                              dtype=torch.float32)
        self.hg = torch.tensor(frame["HG"].astype(float).values, dtype=torch.float32)
        self.ag = torch.tensor(frame["AG"].astype(float).values, dtype=torch.float32)
        self.game_id = frame["game_id"].astype(str).values

    def __len__(self):
        return len(self.game_id)

    def __getitem__(self, idx):
        return {
            "game_id": self.game_id[idx],
            "x": self.x[idx],
            "HG": self.hg[idx],
            "AG": self.ag[idx],
        }


def collate(batch):
    return {
        "game_id": [b["game_id"] for b in batch],
        "x": torch.stack([b["x"] for b in batch]),
        "HG": torch.stack([b["HG"] for b in batch]),
        "AG": torch.stack([b["AG"] for b in batch]),
    }


# ============================================================================
# PER-MODULE MLP
# ============================================================================

class MLPRateModel(nn.Module):
    """log(lambda) = log(prior) + head(trunk(x)).

    Head is init'd small so the model starts near the goal prior while
    gradients still flow to earlier layers.
    """
    def __init__(self, input_dim, home_prior=1.35, away_prior=1.10,
                 hidden1=64, hidden2=32, dropout=0.2):
        super().__init__()
        self.register_buffer(
            "log_home_prior", torch.tensor(float(np.log(home_prior)))
        )
        self.register_buffer(
            "log_away_prior", torch.tensor(float(np.log(away_prior)))
        )
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden2, 2)
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        h = self.trunk(x)
        logits = self.head(h)
        return (torch.exp(self.log_home_prior + logits[:, 0]),
                torch.exp(self.log_away_prior + logits[:, 1]))


def train_single_module(module_name, cfg, device):
    """Train one MLP. Returns a dict with predictions on train and test."""
    train_df_full, test_df, key_col, feature_cols = load_module_frames(module_name)

    # Inner split for early stopping
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(len(train_df_full))
    n_val = int(len(train_df_full) * cfg.module_val_ratio)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    train_df = train_df_full.iloc[train_idx].reset_index(drop=True)
    val_df = train_df_full.iloc[val_idx].reset_index(drop=True)

    # Standardize on train subset only
    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std().replace(0, 1.0)

    for df in (train_df, val_df, test_df):
        df[feature_cols] = (df[feature_cols] - mean) / std

    train_full_scaled = train_df_full.copy()
    train_full_scaled[feature_cols] = (
        (train_full_scaled[feature_cols] - mean) / std
    )

    home_prior = float(train_df["HG"].mean())
    away_prior = float(train_df["AG"].mean())

    train_ds = SimpleDataset(train_df, feature_cols)
    val_ds = SimpleDataset(val_df, feature_cols)

    train_loader = DataLoader(train_ds, batch_size=cfg.module_batch_size,
                              shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=cfg.module_batch_size,
                            shuffle=False, collate_fn=collate)

    # Auto-size hidden dims
    h1 = cfg.module_hidden1 or min(cfg.module_hidden_cap, max(16, 2 * len(feature_cols)))
    h2 = cfg.module_hidden2 or max(8, h1 // 2)

    model = MLPRateModel(
        len(feature_cols), home_prior, away_prior,
        hidden1=h1, hidden2=h2, dropout=cfg.module_dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  [{module_name}] MLP: {len(feature_cols)}->{h1}->{h2}->2, "
          f"params={n_params:,}, rows={len(train_ds)}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.module_lr,
                            weight_decay=cfg.module_weight_decay)

    best_val, best_state, best_epoch, no_imp = float("inf"), None, 0, 0
    for epoch in range(1, cfg.module_epochs + 1):
        model.train()
        for b in train_loader:
            x = b["x"].to(device)
            hg = b["HG"].to(device)
            ag = b["AG"].to(device)
            opt.zero_grad()
            lh, la = model(x)
            loss = (F.poisson_nll_loss(lh, hg, log_input=False, full=True)
                    + F.poisson_nll_loss(la, ag, log_input=False, full=True))
            loss.mean().backward()
            opt.step()

        model.eval()
        v = 0.0
        n = 0
        with torch.no_grad():
            for b in val_loader:
                x = b["x"].to(device)
                hg = b["HG"].to(device)
                ag = b["AG"].to(device)
                lh, la = model(x)
                v += float((F.poisson_nll_loss(lh, hg, log_input=False, full=True)
                            + F.poisson_nll_loss(la, ag, log_input=False, full=True)).sum().item())
                n += len(x)
        v /= max(1, n)

        if v < best_val - 1e-5:
            best_val, best_epoch, no_imp = v, epoch, 0
            best_state = {k: t.detach().cpu().clone()
                          for k, t in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= cfg.module_patience:
                break

        if cfg.verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"    ep {epoch:03d} val_nll={v:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"  [{module_name}] best_epoch={best_epoch}  val_nll={best_val:.4f}")

    # Predict on full train + test
    def _predict(df):
        ds = SimpleDataset(df, feature_cols)
        loader = DataLoader(ds, batch_size=cfg.module_batch_size,
                            shuffle=False, collate_fn=collate)
        rows = []
        model.eval()
        with torch.no_grad():
            for b in loader:
                x = b["x"].to(device)
                lh, la = model(x)
                for i, gid in enumerate(b["game_id"]):
                    rows.append({
                        "game_id": gid,
                        "HG": float(b["HG"][i]),
                        "AG": float(b["AG"][i]),
                        "lambda_home": float(lh[i]),
                        "lambda_away": float(la[i]),
                    })
        return pd.DataFrame(rows)

    return {
        "model_state": model.state_dict(),
        "feature_cols": feature_cols,
        "train_pred": _predict(train_full_scaled),
        "test_pred": _predict(test_df),
    }


# ============================================================================
# LSTM FUSION
# ============================================================================

class LSTMFusion(nn.Module):
    """LSTM over the ordered sequence of module predictions.

    Each module contributes one timestep of shape (3,):
        (lambda_home, lambda_away, diff)
    where diff = lambda_home - lambda_away.
    """
    def __init__(self, n_modules, home_prior=1.35, away_prior=1.10,
                 hidden=32, num_layers=1, dropout=0.3,
                 outcome_head_hidden=32):
        super().__init__()
        self.register_buffer(
            "log_home_prior", torch.tensor(float(np.log(home_prior)))
        )
        self.register_buffer(
            "log_away_prior", torch.tensor(float(np.log(away_prior)))
        )
        self.n_modules = n_modules

        self.lstm = nn.LSTM(
            input_size=3,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.input_dropout = nn.Dropout(dropout)

        self.rate_head = nn.Linear(hidden, 2)
        nn.init.normal_(self.rate_head.weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.rate_head.bias)

        self.outcome_head = nn.Sequential(
            nn.Linear(hidden, outcome_head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(outcome_head_hidden, 3),
        )

    def forward(self, x):
        B = x.shape[0]
        seq = x.view(B, self.n_modules, 3)
        seq = self.input_dropout(seq)
        _, (h_n, _) = self.lstm(seq)
        h = h_n[-1]
        rate_logits = self.rate_head(h)
        lam_h = torch.exp(self.log_home_prior + rate_logits[:, 0])
        lam_a = torch.exp(self.log_away_prior + rate_logits[:, 1])
        outcome_logits = self.outcome_head(h)
        return lam_h, lam_a, outcome_logits


def build_fusion_frame(split_preds, modules):
    """Merge per-module predictions into a fusion frame. Order matters."""
    parts = []
    for mn in modules:
        df = split_preds[mn][["game_id", "HG", "AG",
                              "lambda_home", "lambda_away"]].copy()
        df = df.rename(columns={
            "lambda_home": f"{mn}_lh",
            "lambda_away": f"{mn}_la",
        })
        df[f"{mn}_diff"] = df[f"{mn}_lh"] - df[f"{mn}_la"]
        parts.append(df[["game_id", "HG", "AG",
                         f"{mn}_lh", f"{mn}_la", f"{mn}_diff"]])

    merged = parts[0]
    for df in parts[1:]:
        merged = merged.merge(df.drop(columns=["HG", "AG"]),
                              on="game_id", how="inner")

    feature_cols = []
    for mn in modules:
        feature_cols += [f"{mn}_lh", f"{mn}_la", f"{mn}_diff"]

    return merged, feature_cols


# ============================================================================
# LOSS & METRICS
# ============================================================================

def outcome_labels(hg, ag):
    return np.where(hg > ag, 0, np.where(hg == ag, 1, 2))


def compute_class_weights(outcome, use_full=True):
    counts = np.bincount(outcome, minlength=3).astype(float)
    counts[counts == 0] = 1.0
    if use_full:
        w = counts.sum() / (3 * counts)
    else:
        w = np.sqrt(counts.sum() / (3 * counts))
    return torch.tensor(w, dtype=torch.float32)


def combined_loss(lam_h, lam_a, logits, hg, ag, outcome,
                  outcome_weight, class_weights, spread_weight=0.0,
                  target_std=0.5):
    poisson = F.poisson_nll_loss(lam_h, hg, log_input=False, full=True) + \
              F.poisson_nll_loss(lam_a, ag, log_input=False, full=True)

    if class_weights is not None and class_weights[1].item() == 0.0:
        mask = outcome != 1
        if mask.any():
            ce = F.cross_entropy(logits[mask], outcome[mask],
                                 weight=class_weights)
        else:
            ce = torch.tensor(0.0, device=logits.device)
    else:
        ce = F.cross_entropy(logits, outcome, weight=class_weights)

    if spread_weight > 0.0:
        diff = lam_h - lam_a
        std_actual = diff.std() + 1e-6
        spread_penalty = spread_weight * F.relu(
            torch.tensor(target_std, device=lam_h.device) - std_actual
        )
    else:
        spread_penalty = torch.tensor(0.0, device=lam_h.device)

    return poisson + outcome_weight * ce + spread_penalty


# ============================================================================
# DECISION RULES
# ============================================================================

def rates_to_probs(lam_h, lam_a, margin):
    diff = lam_h - lam_a
    k = 2.0
    exp_h = np.exp(k * (diff - margin))
    exp_a = np.exp(k * (-diff - margin))
    exp_d = np.ones_like(diff)
    total = exp_h + exp_a + exp_d
    probs = np.zeros((len(lam_h), 3))
    probs[:, 0] = exp_h / total
    probs[:, 2] = exp_a / total
    probs[:, 1] = exp_d / total
    return probs


def rate_based_predictions(lam_h, lam_a, margin):
    diff = lam_h - lam_a
    pred = np.full(len(diff), 1, dtype=int)
    pred[diff > margin] = 0
    pred[diff < -margin] = 2
    return pred


def tune_draw_margin(lam_h, lam_a, hg, ag, margins):
    true = outcome_labels(hg, ag)
    best_margin, best_bal = 0.0, -1.0
    for margin in margins:
        pred = rate_based_predictions(lam_h, lam_a, margin)
        bal = balanced_accuracy_score(true, pred)
        if bal > best_bal:
            best_bal = bal
            best_margin = float(margin)
    return best_margin, best_bal


def prior_correct_probs(probs, train_priors, target_priors):
    probs = np.clip(probs, 1e-8, None)
    correction = target_priors / np.clip(train_priors, 1e-8, None)
    corrected = probs * correction
    return corrected / corrected.sum(axis=1, keepdims=True)


def sweep_blend_weights(clf_probs, rate_probs, y_true, weights, objective):
    results = []
    for w in weights:
        pred = np.argmax(w * clf_probs + (1 - w) * rate_probs, axis=1)
        acc = accuracy_score(y_true, pred)
        bal = balanced_accuracy_score(y_true, pred)
        mf1 = f1_score(y_true, pred, average="macro", zero_division=0)
        if objective == "balanced_accuracy":
            score = bal
        elif objective == "macro_f1":
            score = mf1
        else:
            score = acc
        results.append({
            "weight_clf": float(w),
            "accuracy": float(acc),
            "balanced_accuracy": float(bal),
            "macro_f1": float(mf1),
            "score": float(score),
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


# ============================================================================
# MAIN TRAINING
# ============================================================================

def resolve_device(spec):
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_dashboard():
    device = resolve_device(CFG.device)
    torch.manual_seed(CFG.seed)
    np.random.seed(CFG.seed)

    print(f"Device: {device}")
    print(f"Modules: {list(CFG.modules)}")

    # ------------------------------------------------------------------
    # Phase 1: per-module MLPs
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 1: PER-MODULE MLPs")
    print("=" * 78)

    module_results = {}
    for mn in CFG.modules:
        try:
            r = train_single_module(mn, CFG, device)
            module_results[mn] = r
        except Exception as e:
            print(f"  ❌ Error in module '{mn}': {e}")
            raise

    # ------------------------------------------------------------------
    # Phase 2: build fusion frames
    # ------------------------------------------------------------------
    train_pred_map = {mn: r["train_pred"] for mn, r in module_results.items()}
    test_pred_map = {mn: r["test_pred"] for mn, r in module_results.items()}

    fusion_train_full, feature_cols = build_fusion_frame(train_pred_map, CFG.modules)
    fusion_test, _ = build_fusion_frame(test_pred_map, CFG.modules)

    print(f"\n  Fusion frame: train={len(fusion_train_full)}, "
          f"test={len(fusion_test)}")

    if fusion_train_full.empty or fusion_test.empty:
        raise ValueError("Empty fusion frame — check module coverage.")

    # ------------------------------------------------------------------
    # Phase 3: train LSTM fusion
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 3: LSTM FUSION")
    print("=" * 78)

    # Split fusion train / val
    rng = np.random.default_rng(CFG.seed)
    idx = rng.permutation(len(fusion_train_full))
    n_val = int(len(fusion_train_full) * CFG.val_fraction)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    fusion_train = fusion_train_full.iloc[train_idx].reset_index(drop=True)
    fusion_val = fusion_train_full.iloc[val_idx].reset_index(drop=True)

    home_prior = float(fusion_train["HG"].mean())
    away_prior = float(fusion_train["AG"].mean())

    train_ds = FusionDataset(fusion_train, feature_cols)
    val_ds = FusionDataset(fusion_val, feature_cols)
    test_ds = FusionDataset(fusion_test, feature_cols)

    train_loader = DataLoader(train_ds, batch_size=CFG.lstm_batch_size,
                              shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=CFG.lstm_batch_size,
                            shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=CFG.lstm_batch_size,
                             shuffle=False, collate_fn=collate)

    n_modules = len(CFG.modules)
    model = LSTMFusion(
        n_modules, home_prior, away_prior,
        hidden=CFG.lstm_hidden,
        num_layers=CFG.lstm_layers,
        dropout=CFG.lstm_dropout,
        outcome_head_hidden=CFG.lstm_outcome_head_hidden,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Sequence length: {n_modules}  |  Params: {n_params:,}")
    print(f"  Rows: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    # Class weights
    train_outcomes = outcome_labels(fusion_train["HG"].values,
                                    fusion_train["AG"].values)
    val_outcomes = outcome_labels(fusion_val["HG"].values,
                                  fusion_val["AG"].values)

    class_weights = compute_class_weights(
        train_outcomes, use_full=CFG.use_full_class_weights
    ).to(device)
    class_weights = class_weights * torch.tensor(
        [CFG.home_weight_scale, CFG.draw_weight_scale, CFG.away_weight_scale],
        dtype=torch.float32, device=device,
    )
    draw_disabled = class_weights[1].item() == 0.0
    print(f"  Class weights: {[round(w, 3) for w in class_weights.tolist()]}")

    decision_source = CFG.decision_source
    if decision_source == "auto":
        decision_source = "rate" if draw_disabled else "classifier"
    print(f"  Decision source: {decision_source}")

    opt = torch.optim.AdamW(model.parameters(), lr=CFG.lstm_lr,
                            weight_decay=CFG.lstm_weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5, min_lr=1e-6,
    )

    best_val_metric, best_state, best_epoch, no_imp = -1.0, None, 0, 0

    for epoch in range(1, CFG.lstm_epochs + 1):
        model.train()
        for b in train_loader:
            x = b["x"].to(device)
            hg = b["HG"].to(device)
            ag = b["AG"].to(device)
            oc = outcome_labels(hg.cpu().numpy(), ag.cpu().numpy())
            oc_t = torch.tensor(oc, dtype=torch.long, device=device)
            opt.zero_grad()
            lh, la, logits = model(x)
            loss = combined_loss(lh, la, logits, hg, ag, oc_t,
                                 CFG.outcome_weight, class_weights,
                                 spread_weight=CFG.spread_weight,
                                 target_std=CFG.target_std)
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        # Validation
        model.eval()
        val_lh, val_la, val_logits, val_true = [], [], [], []
        with torch.no_grad():
            for b in val_loader:
                x = b["x"].to(device)
                lh, la, logits = model(x)
                val_lh.extend(lh.cpu().numpy().tolist())
                val_la.extend(la.cpu().numpy().tolist())
                val_logits.append(F.softmax(logits, dim=1).cpu().numpy())
                val_true.extend(outcome_labels(
                    b["HG"].numpy(), b["AG"].numpy()
                ).tolist())

        val_lh = np.array(val_lh)
        val_la = np.array(val_la)
        val_logits = np.vstack(val_logits)
        val_true = np.array(val_true)

        margin_now, val_bal_rate = tune_draw_margin(
            val_lh, val_la, fusion_val["HG"].values, fusion_val["AG"].values,
            CFG.draw_margin_grid,
        )
        clf_pred = np.argmax(val_logits, axis=1)
        val_bal_clf = balanced_accuracy_score(val_true, clf_pred)

        if decision_source == "rate":
            val_metric = val_bal_rate
        else:
            val_metric = val_bal_clf

        sched.step(-val_metric)

        if val_metric > best_val_metric + 1e-4:
            best_val_metric = val_metric
            best_epoch = epoch
            no_imp = 0
            best_state = {k: t.detach().cpu().clone()
                          for k, t in model.state_dict().items()}
        else:
            no_imp += 1

        if CFG.verbose or epoch % 10 == 0 or epoch == 1:
            print(f"  ep {epoch:03d} | val_bal_rate={val_bal_rate:.4f} "
                  f"val_bal_clf={val_bal_clf:.4f} margin={margin_now:.3f}")

        if no_imp >= CFG.lstm_patience:
            print(f"  ⏹ Early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  ✓ Best model from epoch {best_epoch}")

    # ------------------------------------------------------------------
    # Phase 4: final evaluation on validation and test
    # ------------------------------------------------------------------
    model.eval()
    val_lh, val_la, val_probs, val_true = [], [], [], []
    with torch.no_grad():
        for b in val_loader:
            x = b["x"].to(device)
            lh, la, logits = model(x)
            val_lh.extend(lh.cpu().numpy().tolist())
            val_la.extend(la.cpu().numpy().tolist())
            val_probs.append(F.softmax(logits, dim=1).cpu().numpy())
            val_true.extend(outcome_labels(
                b["HG"].numpy(), b["AG"].numpy()
            ).tolist())
    val_lh = np.array(val_lh)
    val_la = np.array(val_la)
    val_probs = np.vstack(val_probs)
    val_true = np.array(val_true)

    train_priors = np.bincount(train_outcomes, minlength=3).astype(float)
    train_priors /= train_priors.sum()
    val_probs_corr = prior_correct_probs(val_probs, train_priors, train_priors)

    draw_margin, val_bal = tune_draw_margin(
        val_lh, val_la, fusion_val["HG"].values, fusion_val["AG"].values,
        CFG.draw_margin_grid,
    )
    print(f"\n  Best draw margin: {draw_margin:.3f} "
          f"(val balanced acc {val_bal:.4f})")

    val_rate_probs = rates_to_probs(val_lh, val_la, draw_margin)
    val_rate_pred = rate_based_predictions(val_lh, val_la, draw_margin)
    val_clf_pred = np.argmax(val_probs_corr, axis=1)

    def _summary(y_true, y_pred):
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                       zero_division=0)),
        }

    val_rate_m = _summary(val_true, val_rate_pred)
    val_clf_m = _summary(val_true, val_clf_pred)

    sweep_results = sweep_blend_weights(
        val_probs_corr, val_rate_probs, val_true,
        CFG.blend_sweep, CFG.sweep_objective,
    )
    best_blend = sweep_results[0]
    val_blend_pred = np.argmax(
        best_blend["weight_clf"] * val_probs_corr
        + (1 - best_blend["weight_clf"]) * val_rate_probs, axis=1
    )
    val_blend_m = _summary(val_true, val_blend_pred)

    print("\n  Validation metrics:")
    print(f"    {'rate':>12}  acc={val_rate_m['accuracy']:.4f}  "
          f"bal={val_rate_m['balanced_accuracy']:.4f}")
    print(f"    {'classifier':>12}  acc={val_clf_m['accuracy']:.4f}  "
          f"bal={val_clf_m['balanced_accuracy']:.4f}")
    print(f"    {'blend':>12}  acc={val_blend_m['accuracy']:.4f}  "
          f"bal={val_blend_m['balanced_accuracy']:.4f} "
          f"(w_clf={best_blend['weight_clf']:.2f})")

    # ---- Test predictions -------------------------------------------------
    test_lh, test_la, test_probs, test_keys = [], [], [], []
    test_hg, test_ag = [], []
    with torch.no_grad():
        for b in test_loader:
            x = b["x"].to(device)
            lh, la, logits = model(x)
            test_lh.extend(lh.cpu().numpy().tolist())
            test_la.extend(la.cpu().numpy().tolist())
            test_probs.append(F.softmax(logits, dim=1).cpu().numpy())
            test_keys.extend(b["game_id"])
            test_hg.extend(b["HG"].numpy().tolist())
            test_ag.extend(b["AG"].numpy().tolist())

    test_lh = np.array(test_lh)
    test_la = np.array(test_la)
    test_probs = np.vstack(test_probs)

    test_probs_corr = prior_correct_probs(test_probs, train_priors, train_priors)
    test_rate_probs = rates_to_probs(test_lh, test_la, draw_margin)
    test_rate_pred = rate_based_predictions(test_lh, test_la, draw_margin)

    if decision_source == "rate":
        final_probs = test_rate_probs
        final_pred_internal = test_rate_pred
    elif decision_source == "blend":
        w = best_blend["weight_clf"]
        final_probs = w * test_probs_corr + (1 - w) * test_rate_probs
        final_pred_internal = np.argmax(final_probs, axis=1)
    else:  # classifier
        final_probs = test_probs_corr
        final_pred_internal = np.argmax(final_probs, axis=1)

    final_pred_signed = INT_TO_SIGNED[final_pred_internal]

    print(f"\n  Final prediction source: {decision_source}")
    print(f"  Test prediction distribution (signed):")
    for code, name in [(1, "HOME"), (0, "DRAW"), (-1, "AWAY")]:
        c = int((final_pred_signed == code).sum())
        pct = 100.0 * c / len(final_pred_signed) if len(final_pred_signed) else 0.0
        print(f"    {name:>4}: {c:>4d} ({pct:>5.1f}%)")

    # ------------------------------------------------------------------
    # Phase 5: save outputs
    # ------------------------------------------------------------------
    WEEKLY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # lstm_predictions.csv — matches the naive baseline format (4 key cols)
    # plus the probability columns the dashboard reads.
    pred_frame = pd.DataFrame({
        "game_id": test_keys,
        "HG": test_hg,
        "AG": test_ag,
        "lambda_home": test_lh,
        "lambda_away": test_la,
        "expected_goal_diff": test_lh - test_la,
        "prob_home": final_probs[:, 0],
        "prob_draw": final_probs[:, 1],
        "prob_away": final_probs[:, 2],
        "predicted_result": final_pred_signed,
    })
    pred_frame.to_csv(PRED_PATH, index=False)

    # lstm_model.pt — state dicts for per-module MLPs plus the fusion.
    save_dict = {
        "fusion_state": model.state_dict(),
        "fusion_modules": list(CFG.modules),
        "fusion_feature_cols": feature_cols,
        "home_prior": home_prior,
        "away_prior": away_prior,
        "draw_margin": draw_margin,
        "blend_weight": best_blend["weight_clf"],
        "train_priors": train_priors.tolist(),
        "decision_source": decision_source,
        "config": {k: v for k, v in CFG.__dict__.items()
                   if not isinstance(v, np.ndarray)},
        "module_states": {
            mn: r["model_state"] for mn, r in module_results.items()
        },
        "module_feature_cols": {
            mn: r["feature_cols"] for mn, r in module_results.items()
        },
    }
    torch.save(save_dict, MODEL_PATH)

    # lstm_summary.json
    summary = {
        "run_date": pd.Timestamp.now().isoformat(timespec="seconds"),
        "architecture": "per_module_mlp + lstm_fusion",
        "modules": list(CFG.modules),
        "params": {"fusion_total_trainable": n_params},
        "rows": {
            "fusion_train": len(fusion_train),
            "fusion_val": len(fusion_val),
            "upcoming": len(fusion_test),
        },
        "priors": {"train": train_priors.tolist()},
        "decision_source": decision_source,
        "draw_disabled_in_classifier": bool(draw_disabled),
        "validation": {
            "draw_margin": draw_margin,
            "rate_based": val_rate_m,
            "classifier_prior_corrected": val_clf_m,
            "best_blend": val_blend_m,
            "blend_weight_clf": best_blend["weight_clf"],
        },
        "blend_sweep_top5": sweep_results[:5],
        "best_epoch": best_epoch,
        "result_coding": {"HOME_WIN": 1, "DRAW": 0, "AWAY_WIN": -1},
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n  Saved predictions: {PRED_PATH}")
    print(f"  Saved model:       {MODEL_PATH}")
    print(f"  Saved summary:     {SUMMARY_PATH}")

    return summary


def main():
    train_dashboard()


if __name__ == "__main__":
    main()