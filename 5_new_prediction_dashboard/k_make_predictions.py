#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train the LSTM on the DASHBOARD data and produce weekly predictions.

Reads:
    ./nn_input_data/nn_input_*_TRAIN.csv   (full history)
    ./nn_input_data/nn_input_*_TEST.csv    (current upcoming fixtures)
    ./nn_input_data/nn_module_metadata.json
    ./upcoming_lineups/upcoming_lineups_latest.csv
    ../data/big5_matches_fixed.csv         (fallback for league resolution)

Writes (weekly_results/ — append-only history):
    lstm_model.pt                              (always the latest model)
    lstm_summary.json                          (always the latest summary)
    lstm_predictions.csv                       (always the latest raw predictions)
    weekly_predictions_latest.csv              (always the newest clean summary)
    weekly_predictions_YYYY-MM-DD.csv          (one per matchday/run)
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
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
    precision_recall_fscore_support,
)


# ============================================================================
# PATHS
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

NN_INPUT_DATA_DIR = SCRIPT_DIR / "nn_input_data"
UPCOMING_LINEUPS_PATH = SCRIPT_DIR / "upcoming_lineups" / "upcoming_lineups_latest.csv"
HISTORICAL_MATCH_PATH = PROJECT_DIR / "data" / "big5_matches_fixed.csv"
WEEKLY_RESULTS_DIR = SCRIPT_DIR / "weekly_results"

META_PATH = NN_INPUT_DATA_DIR / "nn_module_metadata.json"
SUMMARY_PATH = WEEKLY_RESULTS_DIR / "lstm_summary.json"
MODEL_PATH = WEEKLY_RESULTS_DIR / "lstm_model.pt"
PRED_PATH = WEEKLY_RESULTS_DIR / "lstm_predictions.csv"
WEEKLY_PRED_PATH = WEEKLY_RESULTS_DIR / "weekly_predictions_latest.csv"


DEFAULT_MODULES = ["lineup", "player_stats", "skill_matchups", "match_stats",
                   "league", "season", "matchup"]

OUTCOME_LABEL = {0: "HOME_WIN", 1: "DRAW", 2: "AWAY_WIN"}

# Historical league label → standardized tag (used by scraper)
LEAGUE_TAG_MAP = {
    "England Premier League": "premier-league",
    "France Ligue 1": "ligue-1",
    "Germany Bundesliga": "bundesliga",
    "Italy Serie A": "serie-a",
    "Spain La Liga": "laliga",
    # Already-standardized fallbacks
    "premier-league": "premier-league",
    "ligue-1": "ligue-1",
    "bundesliga": "bundesliga",
    "serie-a": "serie-a",
    "laliga": "laliga",
}

# Human-readable display names for the weekly CSV
LEAGUE_DISPLAY = {
    "premier-league": "Premier League",
    "bundesliga": "Bundesliga",
    "laliga": "La Liga",
    "serie-a": "Serie A",
    "ligue-1": "Ligue 1",
}


@dataclass
class LSTMConfig:
    modules: list[str]
    val_fraction: float = 0.15
    hidden_dim: int = 48
    num_lstm_layers: int = 1
    rate_head_hidden: int = 24
    outcome_head_hidden: int = 32
    dropout: float = 0.3
    input_dropout: float = 0.15
    epochs: int = 80
    batch_size: int = 128
    learning_rate: float = 5e-4
    weight_decay: float = 1e-3
    early_stopping_patience: int = 10
    lr_patience: int = 5
    lr_factor: float = 0.5
    min_lr: float = 1e-6
    outcome_weight: float = 1.0
    use_full_class_weights: bool = True
    draw_weight_scale: float = 0.80
    away_weight_scale: float = 0.95
    home_weight_scale: float = 1.00
    draw_margin_grid: tuple[float, ...] = field(
        default_factory=lambda: tuple(np.round(np.linspace(0.0, 0.8, 81), 3))
    )
    blend_sweep: tuple[float, ...] = field(
        default_factory=lambda: tuple(np.round(np.linspace(0.0, 1.0, 21), 2))
    )
    seed: int = 42


# ============================================================================
# HELPERS
# ============================================================================

def dated_output_paths(base_dir: Path, run_date: datetime | None = None):
    """
    Return (dated_path, latest_path) for the weekly predictions.

        dated_path  = base_dir / f"weekly_predictions_{YYYY-MM-DD}.csv"
        latest_path = base_dir / "weekly_predictions_latest.csv"
    """
    if run_date is None:
        run_date = datetime.now()
    stamp = run_date.strftime("%Y-%m-%d")
    return (
        base_dir / f"weekly_predictions_{stamp}.csv",
        base_dir / "weekly_predictions_latest.csv",
    )


# ============================================================================
# DATA LOADING
# ============================================================================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_module_meta():
    if not META_PATH.exists():
        raise FileNotFoundError(f"Missing metadata file {META_PATH}.")
    return json.loads(META_PATH.read_text(encoding="utf-8"))


def _load_split_frame(module_name, split):
    path = NN_INPUT_DATA_DIR / f"nn_input_{module_name}_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing prepared input file: {path}")
    return pd.read_csv(path, low_memory=False)


def _feature_columns(frame):
    exclude = {"game_id", "HG", "AG", "match_id"}
    return [c for c in frame.columns if c not in exclude]


def load_merged_split(split, modules):
    merged = None
    for mn in modules:
        frame = _load_split_frame(mn, split)
        frame = frame.copy().drop_duplicates(subset=["game_id"], keep="first")
        cols = _feature_columns(frame)
        keep_cols = ["game_id"] + cols
        if "HG" in frame.columns and "AG" in frame.columns and split == "TRAIN":
            keep_cols = ["game_id", "HG", "AG"] + cols
        frame = frame[keep_cols]

        if merged is None:
            merged = frame
            continue
        merged = merged.merge(frame, on="game_id", how="inner",
                              suffixes=("", f"_{mn}"))

    if merged is None or merged.empty:
        raise ValueError(f"No merged rows available for split={split}.")

    for col in ["HG", "AG"]:
        if col not in merged.columns:
            merged[col] = np.nan

    merged = merged.replace([np.inf, -np.inf], np.nan)
    if split == "TRAIN":
        merged = merged.dropna(subset=["HG", "AG"]).copy()

    merged = merged.sort_values("game_id").reset_index(drop=True)
    return merged


def split_train_val(frame, val_fraction):
    frame = frame.copy().sort_values("game_id").reset_index(drop=True)
    split_idx = int(len(frame) * (1 - val_fraction))
    split_idx = min(max(split_idx, 1), len(frame) - 1)
    return frame.iloc[:split_idx].copy(), frame.iloc[split_idx:].copy()


def outcome_labels(hg, ag):
    return np.where(hg > ag, 0, np.where(hg == ag, 1, 2))


def compute_class_weights(outcome, num_classes=3, use_full=True):
    counts = np.bincount(outcome, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    if use_full:
        weights = counts.sum() / (num_classes * counts)
    else:
        weights = np.sqrt(counts.sum() / (num_classes * counts))
    return torch.tensor(weights, dtype=torch.float32)


def split_home_away(feature_cols):
    home, away, other = [], [], []
    for c in feature_cols:
        cl = c.lower()
        if cl.startswith("home_"):
            home.append(c)
        elif cl.startswith("away_"):
            away.append(c)
        else:
            other.append(c)
    return home, away, other


# ============================================================================
# CONTEXT LOOKUPS (team names + league)
# ============================================================================

def load_upcoming_context():
    """
    Build:
      - team_lookup: code → raw-name
      - league_lookup: game_id → Competition (scraper tag)
    from upcoming_lineups_latest.csv.
    """
    if not UPCOMING_LINEUPS_PATH.exists():
        return {}, {}

    df = pd.read_csv(UPCOMING_LINEUPS_PATH, low_memory=False)

    team_lookup = {}
    for _, row in df.iterrows():
        for side in ("Home", "Away"):
            code = row.get(f"{side} Team")
            raw = row.get(f"{side} Team Raw")
            if isinstance(code, str) and isinstance(raw, str) and code.strip() and raw.strip():
                team_lookup[code.strip()] = raw.strip()

    league_lookup = {}
    if "game_id" in df.columns and "Competition" in df.columns:
        for _, row in df.iterrows():
            gid = row.get("game_id")
            comp = row.get("Competition")
            if isinstance(gid, str) and isinstance(comp, str) and gid.strip() and comp.strip():
                league_lookup[gid.strip()] = comp.strip()

    return team_lookup, league_lookup


def _normalize_name(name):
    if pd.isna(name):
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def build_team_to_league_from_history():
    """
    Build code → standardized league tag by scanning historical matches.
    Uses the same resolver as the rest of the project so codes match game_ids.
    """
    if not HISTORICAL_MATCH_PATH.exists():
        return {}

    try:
        from src.team_name_mapping_FINAL import (
            resolve_team_name,
            resolve_team_name_fuzzy,
            soccer_teams,
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
        league = LEAGUE_TAG_MAP.get(str(league_raw).strip(), str(league_raw).strip())

        for side in ("home_team", "away_team"):
            code = resolve(row.get(side))
            if code and code not in team_to_league:
                team_to_league[code] = league

    return team_to_league


def extract_team_codes_from_game_id(game_id):
    """game_id format: YYYYMMDD_HOME_AWAY"""
    parts = str(game_id).split("_")
    if len(parts) >= 3:
        return parts[1], parts[2]
    return None, None


def resolve_league_for_game(game_id, league_lookup, team_to_league):
    """
    Resolve league for a game_id with multiple fallbacks:
      1. Scraper's Competition tag (game_id → league)
      2. Either team code → league from historical team→league map
      3. infer_league_and_season (legacy)
      4. "unknown"
    """
    # 1. Scraper tag
    league = league_lookup.get(game_id)
    if league:
        return league

    # 2. Team code lookup
    home_code, away_code = extract_team_codes_from_game_id(game_id)
    for code in (home_code, away_code):
        if code and code in team_to_league:
            return team_to_league[code]

    # 3. Legacy fallback
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


# ============================================================================
# DATASET
# ============================================================================

class MatchDataset(Dataset):
    def __init__(self, frame, home_cols, away_cols, other_cols):
        self.keys = frame["game_id"].astype(str).values
        self.x_home = torch.tensor(frame[home_cols].astype(float).values, dtype=torch.float32)
        self.x_away = torch.tensor(frame[away_cols].astype(float).values, dtype=torch.float32)
        if other_cols:
            self.x_other = torch.tensor(frame[other_cols].astype(float).values, dtype=torch.float32)
        else:
            self.x_other = torch.zeros(len(frame), 0, dtype=torch.float32)
        self.hg = torch.tensor(frame["HG"].astype(float).values, dtype=torch.float32)
        self.ag = torch.tensor(frame["AG"].astype(float).values, dtype=torch.float32)

        hg_np = frame["HG"].values
        ag_np = frame["AG"].values
        outcomes = np.where(
            pd.isna(hg_np) | pd.isna(ag_np), -1,
            np.where(hg_np > ag_np, 0, np.where(hg_np == ag_np, 1, 2))
        )
        self.outcome = torch.tensor(outcomes, dtype=torch.long)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        return {
            "game_id": self.keys[idx],
            "x_home": self.x_home[idx],
            "x_away": self.x_away[idx],
            "x_other": self.x_other[idx],
            "HG": self.hg[idx],
            "AG": self.ag[idx],
            "outcome": self.outcome[idx],
        }


def collate(batch):
    return {
        "game_id": [b["game_id"] for b in batch],
        "x_home": torch.stack([b["x_home"] for b in batch]),
        "x_away": torch.stack([b["x_away"] for b in batch]),
        "x_other": torch.stack([b["x_other"] for b in batch]),
        "HG": torch.stack([b["HG"] for b in batch]),
        "AG": torch.stack([b["AG"] for b in batch]),
        "outcome": torch.stack([b["outcome"] for b in batch]),
    }


# ============================================================================
# MODEL
# ============================================================================

class LSTMMatchModel(nn.Module):
    def __init__(self, home_dim, away_dim, other_dim, config: LSTMConfig,
                 home_prior=1.5, away_prior=1.2):
        super().__init__()
        self.register_buffer("home_prior", torch.tensor(float(home_prior)))
        self.register_buffer("away_prior", torch.tensor(float(away_prior)))

        self.home_proj = nn.Linear(home_dim, config.hidden_dim)
        self.away_proj = nn.Linear(away_dim, config.hidden_dim)
        self.input_dropout = nn.Dropout(config.input_dropout)

        self.lstm = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_lstm_layers,
            batch_first=True,
            dropout=config.dropout if config.num_lstm_layers > 1 else 0.0,
        )

        combined_dim = config.hidden_dim + other_dim
        self.shared_post = nn.Sequential(
            nn.Linear(combined_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        self.home_rate_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.rate_head_hidden),
            nn.ReLU(),
            nn.Linear(config.rate_head_hidden, 1),
        )
        self.away_rate_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.rate_head_hidden),
            nn.ReLU(),
            nn.Linear(config.rate_head_hidden, 1),
        )
        self.outcome_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.outcome_head_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.outcome_head_hidden, 3),
        )

    def forward(self, x_home, x_away, x_other):
        h_home = self.input_dropout(F.relu(self.home_proj(x_home)))
        h_away = self.input_dropout(F.relu(self.away_proj(x_away)))
        seq = torch.stack([h_home, h_away], dim=1)
        _, (h_n, _) = self.lstm(seq)
        lstm_out = h_n[-1]

        if x_other.shape[1] > 0:
            combined = torch.cat([lstm_out, x_other], dim=-1)
        else:
            combined = lstm_out

        shared = self.shared_post(combined)

        h_logit = self.home_rate_head(shared).squeeze(-1)
        a_logit = self.away_rate_head(shared).squeeze(-1)
        mult_home = 0.2 + 2.8 * torch.sigmoid(h_logit)
        mult_away = 0.2 + 2.8 * torch.sigmoid(a_logit)
        lam_home = self.home_prior * mult_home
        lam_away = self.away_prior * mult_away

        outcome_logits = self.outcome_head(shared)
        return lam_home, lam_away, outcome_logits


# ============================================================================
# LOSS & METRICS
# ============================================================================

def combined_loss(lam_h, lam_a, outcome_logits, hg, ag, outcome,
                  outcome_weight=1.0, class_weights=None):
    poisson = F.poisson_nll_loss(lam_h, hg, log_input=False, full=True) + \
              F.poisson_nll_loss(lam_a, ag, log_input=False, full=True)
    ce = F.cross_entropy(outcome_logits, outcome, weight=class_weights)
    return poisson + outcome_weight * ce


def compute_metrics(model, loader, device, class_weights=None):
    model.eval()
    losses = []
    pred_outcomes, true_outcomes = [], []
    home_rates, away_rates = [], []

    with torch.no_grad():
        for batch in loader:
            if (batch["outcome"] < 0).any():
                continue

            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            lam_h, lam_a, outcome_logits = model(batch["x_home"], batch["x_away"], batch["x_other"])
            loss = combined_loss(lam_h, lam_a, outcome_logits,
                                  batch["HG"], batch["AG"], batch["outcome"],
                                  class_weights=class_weights)
            losses.append(float(loss.item()))
            pred = torch.argmax(outcome_logits, dim=1)
            pred_outcomes.extend(pred.cpu().numpy().tolist())
            true_outcomes.extend(batch["outcome"].cpu().numpy().tolist())
            home_rates.append(float(lam_h.mean().item()))
            away_rates.append(float(lam_a.mean().item()))

    if not losses:
        return {"loss": float("nan"), "accuracy": float("nan"),
                "balanced_accuracy": float("nan"), "macro_f1": float("nan"),
                "mean_home_rate": float("nan"), "mean_away_rate": float("nan"),
                "confusion_matrix": [[0]*3]*3}

    acc = float((np.array(pred_outcomes) == np.array(true_outcomes)).mean())
    bal = float(balanced_accuracy_score(true_outcomes, pred_outcomes))
    macro_f1 = float(f1_score(true_outcomes, pred_outcomes, average="macro", zero_division=0))
    cm = confusion_matrix(true_outcomes, pred_outcomes, labels=[0, 1, 2]).tolist()

    return {
        "loss": float(np.mean(losses)),
        "accuracy": acc,
        "balanced_accuracy": bal,
        "macro_f1": macro_f1,
        "mean_home_rate": float(np.mean(home_rates)),
        "mean_away_rate": float(np.mean(away_rates)),
        "confusion_matrix": cm,
    }


def prior_correct_probs(probs, train_priors, target_priors):
    probs = np.clip(probs, 1e-8, None)
    correction = target_priors / np.clip(train_priors, 1e-8, None)
    corrected = probs * correction
    corrected = corrected / corrected.sum(axis=1, keepdims=True)
    return corrected


def rates_to_probs(lam_h, lam_a, margin):
    probs = np.zeros((len(lam_h), 3))
    diff = lam_h - lam_a
    probs[:, 0] = (diff > margin).astype(float)
    probs[:, 2] = (diff < -margin).astype(float)
    probs[:, 1] = 1.0 - probs[:, 0] - probs[:, 2]
    return probs


def tune_draw_margin(lam_h, lam_a, hg, ag, margins):
    true = outcome_labels(hg, ag)
    best_margin, best_bal = 0.0, -1.0
    for margin in margins:
        pred = np.where(lam_h > lam_a + margin, 0,
                        np.where(lam_a > lam_h + margin, 2, 1))
        bal = balanced_accuracy_score(true, pred)
        if bal > best_bal:
            best_bal = bal
            best_margin = float(margin)
    return best_margin, best_bal


def sweep_blend_weights(clf_probs, rate_probs, y_true, weights, objective="balanced_accuracy"):
    results = []
    for w in weights:
        pred = np.argmax(w * clf_probs + (1 - w) * rate_probs, axis=1)
        acc = accuracy_score(y_true, pred)
        bal = balanced_accuracy_score(y_true, pred)
        mf1 = f1_score(y_true, pred, average="macro", zero_division=0)
        score = bal if objective == "balanced_accuracy" else (
            mf1 if objective == "macro_f1" else acc)
        results.append({
            "weight_clf": float(w), "accuracy": float(acc),
            "balanced_accuracy": float(bal), "macro_f1": float(mf1),
            "score": float(score),
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


# ============================================================================
# TRAINING
# ============================================================================

def train_lstm(config: LSTMConfig, device):
    print("=" * 78)
    print("TRAINING LSTM ON DASHBOARD DATA")
    print("=" * 78)

    # ------------------------------------------------------------------
    # Ensure weekly_results/ exists (do NOT wipe — it's append-only history)
    # ------------------------------------------------------------------
    WEEKLY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Using weekly_results/ as append-only history: {WEEKLY_RESULTS_DIR}")

    # ------------------------------------------------------------------
    # Load merged frames
    # ------------------------------------------------------------------
    train_full = load_merged_split("TRAIN", config.modules)
    test_df = load_merged_split("TEST", config.modules)
    train_df, val_df = split_train_val(train_full, config.val_fraction)

    all_feature_cols = [c for c in train_df.columns if c not in {"game_id", "HG", "AG"}]
    test_feature_cols = set(test_df.columns)
    all_feature_cols = [c for c in all_feature_cols if c in test_feature_cols]

    home_cols, away_cols, other_cols = split_home_away(all_feature_cols)

    print(f"\nFeature layout:")
    print(f"  home_* : {len(home_cols)}")
    print(f"  away_* : {len(away_cols)}")
    print(f"  other  : {len(other_cols)}")
    print(f"\nRows: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    home_prior = float(train_df["HG"].mean())
    away_prior = float(train_df["AG"].mean())

    train_outcomes = outcome_labels(train_df["HG"].values, train_df["AG"].values)
    val_outcomes = outcome_labels(val_df["HG"].values, val_df["AG"].values)

    class_weights = compute_class_weights(train_outcomes,
                                           use_full=config.use_full_class_weights).to(device)
    class_weights = class_weights * torch.tensor(
        [config.home_weight_scale, config.draw_weight_scale, config.away_weight_scale],
        dtype=torch.float32, device=device
    )
    print(f"\nClass weights (scaled): {[round(w, 3) for w in class_weights.tolist()]}")

    print("\nTraining outcome distribution:")
    for name, outcomes in [("Train", train_outcomes), ("Val", val_outcomes)]:
        counts = np.bincount(outcomes, minlength=3)
        pcts = 100.0 * counts / counts.sum()
        print(f"  {name:>5}: HOME {counts[0]:>5} ({pcts[0]:>4.1f}%)  "
              f"DRAW {counts[1]:>5} ({pcts[1]:>4.1f}%)  "
              f"AWAY {counts[2]:>5} ({pcts[2]:>4.1f}%)")

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    train_ds = MatchDataset(train_df, home_cols, away_cols, other_cols)
    val_ds = MatchDataset(val_df, home_cols, away_cols, other_cols)
    test_ds = MatchDataset(test_df, home_cols, away_cols, other_cols)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = LSTMMatchModel(
        home_dim=len(home_cols), away_dim=len(away_cols),
        other_dim=len(other_cols), config=config,
        home_prior=home_prior, away_prior=away_prior,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=config.learning_rate,
                                   weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config.lr_factor,
        patience=config.lr_patience, min_lr=config.min_lr,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_bal = -1.0
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0

    print("\n" + "-" * 78)
    print("TRAINING")
    print("-" * 78)

    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch in train_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            optimizer.zero_grad()
            lam_h, lam_a, outcome_logits = model(batch["x_home"], batch["x_away"], batch["x_other"])
            loss = combined_loss(lam_h, lam_a, outcome_logits,
                                  batch["HG"], batch["AG"], batch["outcome"],
                                  config.outcome_weight, class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()

        val_m = compute_metrics(model, val_loader, device, class_weights)
        scheduler.step(val_m["loss"])

        if val_m["balanced_accuracy"] > best_val_bal + 1e-4:
            best_val_bal = val_m["balanced_accuracy"]
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        lr_now = optimizer.param_groups[0]['lr']
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Ep {epoch:03d}/{config.epochs} | "
                  f"ValAcc {val_m['accuracy']:.3f} ValBal {val_m['balanced_accuracy']:.3f} "
                  f"ValF1 {val_m['macro_f1']:.3f} | "
                  f"λh {val_m['mean_home_rate']:.2f} λa {val_m['mean_away_rate']:.2f} | "
                  f"LR {lr_now:.1e}")

        if epochs_no_improve >= config.early_stopping_patience:
            print(f"  ⏹ Early stop at epoch {epoch} (best val balanced acc {best_val_bal:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  ✓ Restored best model from epoch {best_epoch}")

    # ------------------------------------------------------------------
    # Test predictions
    # ------------------------------------------------------------------
    model.eval()
    test_probs_list, test_keys = [], []
    test_home_pred, test_away_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            lam_h, lam_a, outcome_logits = model(batch["x_home"], batch["x_away"], batch["x_other"])
            probs = F.softmax(outcome_logits, dim=1)
            test_probs_list.append(probs.cpu().numpy())
            test_home_pred.extend(lam_h.cpu().numpy().tolist())
            test_away_pred.extend(lam_a.cpu().numpy().tolist())
            test_keys.extend(batch["game_id"])

    test_probs_raw = np.vstack(test_probs_list)
    test_home_pred = np.array(test_home_pred)
    test_away_pred = np.array(test_away_pred)

    train_priors = np.bincount(train_outcomes, minlength=3).astype(float)
    train_priors /= train_priors.sum()
    target_priors = train_priors
    test_probs_corrected = prior_correct_probs(test_probs_raw, train_priors, target_priors)

    # Draw margin tuning on validation
    val_home_pred, val_away_pred = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            lam_h, lam_a, _ = model(batch["x_home"], batch["x_away"], batch["x_other"])
            val_home_pred.extend(lam_h.cpu().numpy().tolist())
            val_away_pred.extend(lam_a.cpu().numpy().tolist())
    val_home_pred = np.array(val_home_pred)
    val_away_pred = np.array(val_away_pred)

    draw_margin, val_bal = tune_draw_margin(
        val_home_pred, val_away_pred,
        val_df["HG"].values, val_df["AG"].values,
        config.draw_margin_grid,
    )
    print(f"\nBest draw margin (val): {draw_margin:.3f} (val balanced acc {val_bal:.4f})")

    test_rate_probs = rates_to_probs(test_home_pred, test_away_pred, draw_margin)
    test_rate_pred = np.argmax(test_rate_probs, axis=1)
    test_clf_pred = np.argmax(test_probs_raw, axis=1)
    test_clf_corrected_pred = np.argmax(test_probs_corrected, axis=1)

    # Blend weight selected on validation
    val_probs_list = []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            _, _, ol = model(batch["x_home"], batch["x_away"], batch["x_other"])
            val_probs_list.append(F.softmax(ol, dim=1).cpu().numpy())
    val_probs_raw = np.vstack(val_probs_list)
    val_probs_corrected = prior_correct_probs(val_probs_raw, train_priors, target_priors)
    val_rate_probs = rates_to_probs(val_home_pred, val_away_pred, draw_margin)

    sweep_results = sweep_blend_weights(
        val_probs_corrected, val_rate_probs, val_outcomes,
        config.blend_sweep, objective="balanced_accuracy",
    )
    best = sweep_results[0]
    best_weight = best["weight_clf"]
    print(f"\nBest blend weight (from val): {best_weight:.2f}  "
          f"(val balanced acc {best['balanced_accuracy']:.4f})")

    test_blend_probs = best_weight * test_probs_corrected + (1 - best_weight) * test_rate_probs
    test_blend_pred = np.argmax(test_blend_probs, axis=1)

    # ------------------------------------------------------------------
    # Save raw predictions (always "latest")
    # ------------------------------------------------------------------
    pred_frame = pd.DataFrame({
        "game_id": test_keys,
        "HG": test_df["HG"].values,
        "AG": test_df["AG"].values,
        "lambda_home": test_home_pred,
        "lambda_away": test_away_pred,
        "expected_goal_diff": test_home_pred - test_away_pred,
        "prob_home": test_blend_probs[:, 0],
        "prob_draw": test_blend_probs[:, 1],
        "prob_away": test_blend_probs[:, 2],
        "predicted_result": test_blend_pred,
    })
    pred_frame.to_csv(PRED_PATH, index=False)

    # ------------------------------------------------------------------
    # Build the weekly summary table (with league resolution)
    # ------------------------------------------------------------------
    team_lookup, league_lookup = load_upcoming_context()
    team_to_league = build_team_to_league_from_history()

    print(f"\nLeague resolution sources:")
    print(f"  Scraper Competition tags:   {len(league_lookup)} game_ids")
    print(f"  Team→league from history:   {len(team_to_league)} team codes")

    weekly_rows = []
    unresolved_leagues = 0

    for _, row in pred_frame.iterrows():
        gid = row["game_id"]
        home_code, away_code = extract_team_codes_from_game_id(gid)
        home_name = team_lookup.get(home_code, home_code) if home_code else home_code
        away_name = team_lookup.get(away_code, away_code) if away_code else away_code

        league_raw = resolve_league_for_game(gid, league_lookup, team_to_league)
        league_display = LEAGUE_DISPLAY.get(league_raw, league_raw)
        if league_raw == "unknown":
            unresolved_leagues += 1

        prob_h = float(row["prob_home"])
        prob_d = float(row["prob_draw"])
        prob_a = float(row["prob_away"])
        outcome_code = int(row["predicted_result"])
        confidence = max(prob_h, prob_d, prob_a)

        weekly_rows.append({
            "game_id": gid,
            "league": league_display,
            "home_team": home_name,
            "away_team": away_name,
            "predicted_home_goals": round(float(row["lambda_home"]), 2),
            "predicted_away_goals": round(float(row["lambda_away"]), 2),
            "predicted_goal_diff": round(float(row["expected_goal_diff"]), 2),
            "prob_home_win": round(prob_h, 3),
            "prob_draw": round(prob_d, 3),
            "prob_away_win": round(prob_a, 3),
            "predicted_outcome": OUTCOME_LABEL[outcome_code],
            "confidence": round(confidence, 3),
        })

    weekly_df = pd.DataFrame(weekly_rows)

    # ------------------------------------------------------------------
    # Save weekly predictions: dated file + latest copy
    # ------------------------------------------------------------------
    dated_path, latest_path = dated_output_paths(WEEKLY_RESULTS_DIR)
    weekly_df.to_csv(dated_path, index=False)
    weekly_df.to_csv(latest_path, index=False)

    print(f"\n  Saved dated weekly predictions: {dated_path.name}")
    print(f"  Refreshed latest copy:          {latest_path.name}")

    if unresolved_leagues > 0:
        print(f"\n⚠ {unresolved_leagues} fixtures could not be assigned a league.")

    # ------------------------------------------------------------------
    # Save model + summary
    # ------------------------------------------------------------------
    torch.save({
        "state_dict": model.state_dict(),
        "config": config.__dict__,
        "home_cols": home_cols,
        "away_cols": away_cols,
        "other_cols": other_cols,
        "home_prior": home_prior,
        "away_prior": away_prior,
        "draw_margin": draw_margin,
        "blend_weight": best_weight,
        "train_priors": train_priors.tolist(),
    }, MODEL_PATH)

    summary = {
        "run_date": datetime.now().isoformat(timespec="seconds"),
        "config": config.__dict__,
        "params": {"total_trainable": n_params},
        "feature_layout": {
            "home_cols": len(home_cols),
            "away_cols": len(away_cols),
            "other_cols": len(other_cols),
        },
        "rows": {
            "train": len(train_df),
            "val": len(val_df),
            "upcoming": len(test_df),
        },
        "league_resolution": {
            "from_scraper": len(league_lookup),
            "from_history": len(team_to_league),
            "unresolved": unresolved_leagues,
        },
        "priors": {"train": train_priors.tolist()},
        "validation": {"draw_margin": draw_margin, "balanced_accuracy": val_bal},
        "blend_weight_selected": best_weight,
        "blend_sweep_top5": sweep_results[:5],
        "best_epoch": best_epoch,
        "output_files": {
            "dated_weekly": str(dated_path),
            "latest_weekly": str(latest_path),
            "raw_predictions": str(PRED_PATH),
            "model": str(MODEL_PATH),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("WEEKLY PREDICTIONS")
    print("=" * 78)
    print(weekly_df.to_string(index=False))

    print(f"\n  Saved {len(weekly_df)} upcoming fixtures.")
    print(f"  Predicted outcome distribution: "
          f"HOME={(weekly_df['predicted_outcome'] == 'HOME_WIN').sum()}  "
          f"DRAW={(weekly_df['predicted_outcome'] == 'DRAW').sum()}  "
          f"AWAY={(weekly_df['predicted_outcome'] == 'AWAY_WIN').sum()}")
    print(f"  League distribution:")
    for lg, count in weekly_df["league"].value_counts().items():
        print(f"    {lg}: {count}")

    print(f"\n  Saved raw predictions to:       {PRED_PATH}")
    print(f"  Saved dated weekly table to:    {dated_path}")
    print(f"  Saved latest weekly table to:   {latest_path}")
    print(f"  Saved model to:                 {MODEL_PATH}")
    print(f"  Saved summary to:               {SUMMARY_PATH}")

    return summary


def main():
    config = LSTMConfig(modules=list(DEFAULT_MODULES))
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_lstm(config, device)


if __name__ == "__main__":
    main()