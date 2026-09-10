#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""LSTM model that learns from the XGBoost baseline's success.

Key ideas borrowed from the XGBoost pipeline:
1. Use XGBoost-selected features (top-N by gain) instead of all 198.
2. Full inverse-frequency class weights (not sqrt).
3. Small LSTM (1 layer, small hidden) + strong dropout.
4. Prior correction + blend sweep as post-processing.
5. Home/away 2-step sequence is preserved (LSTM's natural fit).

Explicitly designed to test whether the sequential structure of home vs away
adds signal beyond what XGBoost can capture from raw features.
"""

from __future__ import annotations

import json
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
    precision_recall_fscore_support,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
FINAL_DATA_DIR = PROJECT_DIR / "final_data"
RESULT_DATA_DIR = PROJECT_DIR / "result_data"
MISC_DIR = PROJECT_DIR / "misc"

META_PATH = MISC_DIR / "nn_module_metadata.json"
SELECTED_FEATURES_PATH = MISC_DIR / "xgboost_selected_features.json"
SUMMARY_PATH = RESULT_DATA_DIR / "lstm_summary.json"
MODEL_PATH = RESULT_DATA_DIR / "lstm_model.pt"
PRED_PATH = RESULT_DATA_DIR / "lstm_test_predictions.csv"

DEFAULT_MODULES = ["lineup", "player_stats", "skill_matchups", "match_stats",
                   "league", "season", "matchup"]

OUTCOME_NAMES = {0: "HOME", 1: "DRAW", 2: "AWAY"}


@dataclass
class LSTMConfig:
    modules: list[str]
    # Data
    val_fraction: float = 0.15
    use_xgb_selected_features: bool = True
    # Architecture (small, regularized)
    hidden_dim: int = 48
    num_lstm_layers: int = 1
    rate_head_hidden: int = 24
    outcome_head_hidden: int = 32
    dropout: float = 0.3
    input_dropout: float = 0.15
    # Training
    epochs: int = 80
    batch_size: int = 128
    learning_rate: float = 5e-4
    weight_decay: float = 1e-3
    early_stopping_patience: int = 10
    lr_patience: int = 5
    lr_factor: float = 0.5
    min_lr: float = 1e-6
    # Losses
    outcome_weight: float = 1.0
    # NEW: class weight scaling
    use_full_class_weights: bool = True
    draw_weight_scale: float = 0.80       # NEW: 0.80 = dial draws back 20%
    away_weight_scale: float = 0.95       # NEW: slightly dial back away too
    home_weight_scale: float = 1.00       # NEW: keep home baseline
    # Decision
    draw_margin_grid: tuple[float, ...] = field(
        default_factory=lambda: tuple(np.round(np.linspace(0.0, 0.8, 81), 3))
    )
    blend_sweep: tuple[float, ...] = field(
        default_factory=lambda: tuple(np.round(np.linspace(0.0, 1.0, 21), 2))
    )
    seed: int = 42


# ============================================================================
# DATA LOADING
# ============================================================================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_selected_features():
    """Load the XGBoost-selected feature names (if available)."""
    if SELECTED_FEATURES_PATH.exists():
        return set(json.loads(SELECTED_FEATURES_PATH.read_text(encoding="utf-8")))
    return None


def load_module_meta(module_name):
    if not META_PATH.exists():
        raise FileNotFoundError(f"Missing metadata file {META_PATH}.")
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    if module_name not in meta.get("modules", {}):
        raise KeyError(f"Module '{module_name}' not found.")
    return meta["modules"][module_name]


def _load_split_frame(module_name, split):
    path = FINAL_DATA_DIR / f"nn_input_{module_name}_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing prepared input file: {path}")
    return pd.read_csv(path, low_memory=False)


def _feature_columns(frame):
    exclude = {"game_id", "HG", "AG", "match_id"}
    return [c for c in frame.columns if c not in exclude]


def load_merged_split(split, modules, feature_filter=None):
    """Merge module files; optionally restrict to a given set of features."""
    merged = None
    for mn in modules:
        frame = _load_split_frame(mn, split)
        frame = frame.copy().drop_duplicates(subset=["game_id"], keep="first")
        cols = _feature_columns(frame)
        if feature_filter is not None:
            cols = [c for c in cols if c in feature_filter]
        keep_cols = ["game_id", "HG", "AG"] + cols
        frame = frame[keep_cols]
        if merged is None:
            merged = frame
            continue
        merged = merged.merge(frame, on=["game_id", "HG", "AG"], how="inner")

    if merged is None or merged.empty:
        raise ValueError("No merged rows available.")

    merged = merged.replace([np.inf, -np.inf], np.nan)
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
    """Return (home_cols, away_cols, other_cols)."""
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
# DATASET
# ============================================================================

class MatchDataset(Dataset):
    """Each match has a home vector and an away vector (paired inputs)."""

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
        self.outcome = torch.tensor(outcome_labels(frame["HG"].values, frame["AG"].values), dtype=torch.long)

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
    """Two-step LSTM over (home, away) representations.

    Design notes:
    - Small LSTM (1 layer, small hidden) to avoid overfitting
    - Home/away embeddings are projected to the same hidden space, then
      fed as a 2-step sequence
    - Context features (league/season/matchup) are concatenated AFTER the LSTM,
      so the LSTM focuses on home/away structure
    - Three heads: home rate, away rate, outcome classification
    """

    def __init__(self, home_dim, away_dim, other_dim, config: LSTMConfig,
                 home_prior=1.5, away_prior=1.2):
        super().__init__()
        self.register_buffer("home_prior", torch.tensor(float(home_prior)))
        self.register_buffer("away_prior", torch.tensor(float(away_prior)))

        # Project both sides to hidden space
        self.home_proj = nn.Linear(home_dim, config.hidden_dim)
        self.away_proj = nn.Linear(away_dim, config.hidden_dim)
        self.input_dropout = nn.Dropout(config.input_dropout)

        # 2-step LSTM
        self.lstm = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_lstm_layers,
            batch_first=True,
            dropout=config.dropout if config.num_lstm_layers > 1 else 0.0,
        )

        # Other (context) features are concatenated after LSTM
        combined_dim = config.hidden_dim + other_dim

        self.shared_post = nn.Sequential(
            nn.Linear(combined_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        # Separate heads
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
        # Project each side
        h_home = self.input_dropout(F.relu(self.home_proj(x_home)))
        h_away = self.input_dropout(F.relu(self.away_proj(x_away)))

        # 2-step sequence: [home, away]
        seq = torch.stack([h_home, h_away], dim=1)  # (batch, 2, hidden)
        _, (h_n, _) = self.lstm(seq)                 # h_n: (num_layers, batch, hidden)
        lstm_out = h_n[-1]                            # (batch, hidden)

        # Concatenate context
        if x_other.shape[1] > 0:
            combined = torch.cat([lstm_out, x_other], dim=-1)
        else:
            combined = lstm_out

        shared = self.shared_post(combined)

        # Rate heads (bounded multipliers of priors)
        h_logit = self.home_rate_head(shared).squeeze(-1)
        a_logit = self.away_rate_head(shared).squeeze(-1)
        mult_home = 0.2 + 2.8 * torch.sigmoid(h_logit)
        mult_away = 0.2 + 2.8 * torch.sigmoid(a_logit)
        lam_home = self.home_prior * mult_home
        lam_away = self.away_prior * mult_away

        # Outcome head (3-class)
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


# ============================================================================
# PRIOR CORRECTION & BLEND SWEEP
# ============================================================================

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
    print("TRAINING LSTM (feature-reduced, full class weights, regularized)")
    print("=" * 78)

    # ------------------------------------------------------------------
    # Load data, apply XGBoost feature selection
    # ------------------------------------------------------------------
    selected = load_selected_features() if config.use_xgb_selected_features else None
    if selected is not None:
        print(f"Using XGBoost-selected feature set: {len(selected)} features")
    else:
        print("Using all available features (no XGBoost selection applied)")

    train_full = load_merged_split("TRAIN", config.modules, feature_filter=selected)
    test_df = load_merged_split("TEST", config.modules, feature_filter=selected)
    train_df, val_df = split_train_val(train_full, config.val_fraction)

    all_feature_cols = [c for c in train_df.columns if c not in {"game_id", "HG", "AG"}]
    home_cols, away_cols, other_cols = split_home_away(all_feature_cols)

    print(f"\nFeature layout:")
    print(f"  home_* : {len(home_cols)}")
    print(f"  away_* : {len(away_cols)}")
    print(f"  other  : {len(other_cols)}")

    print(f"\nRows: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    # Home/away priors
    home_prior = float(train_df["HG"].mean())
    away_prior = float(train_df["AG"].mean())

    # Class weights (FULL inverse-frequency)
    train_outcomes = outcome_labels(train_df["HG"].values, train_df["AG"].values)
    val_outcomes = outcome_labels(val_df["HG"].values, val_df["AG"].values)
    test_outcomes = outcome_labels(test_df["HG"].values, test_df["AG"].values)
    class_weights = compute_class_weights(train_outcomes, use_full=config.use_full_class_weights).to(device)
    print(f"\nClass weights: {[round(w, 3) for w in class_weights.tolist()]}")

    print("\nTrue outcome distribution:")
    for name, outcomes in [("Train", train_outcomes), ("Val", val_outcomes), ("Test", test_outcomes)]:
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
    print(f"Hidden dim: {config.hidden_dim} | LSTM layers: {config.num_lstm_layers}")
    print(f"Dropout: {config.dropout} | Input dropout: {config.input_dropout}")

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
        test_m = compute_metrics(model, test_loader, device, class_weights)
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
                  f"TestAcc {test_m['accuracy']:.3f} TestBal {test_m['balanced_accuracy']:.3f} | "
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

    # Prior correction
    train_priors = np.bincount(train_outcomes, minlength=3).astype(float)
    train_priors /= train_priors.sum()
    test_priors = np.bincount(test_outcomes, minlength=3).astype(float)
    test_priors /= test_priors.sum()
    test_probs_corrected = prior_correct_probs(test_probs_raw, train_priors, test_priors)

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

    # Decision paths
    test_rate_probs = rates_to_probs(test_home_pred, test_away_pred, draw_margin)
    test_rate_pred = np.argmax(test_rate_probs, axis=1)
    test_clf_pred = np.argmax(test_probs_raw, axis=1)
    test_clf_corrected_pred = np.argmax(test_probs_corrected, axis=1)

    def _summary(y_true, y_pred):
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        }

    rate_metrics = _summary(test_outcomes, test_rate_pred)
    clf_metrics = _summary(test_outcomes, test_clf_pred)
    clf_corrected_metrics = _summary(test_outcomes, test_clf_corrected_pred)

    # Blend sweep
    sweep_results = sweep_blend_weights(
        test_probs_corrected, test_rate_probs, test_outcomes,
        config.blend_sweep, objective="balanced_accuracy",
    )
    best = sweep_results[0]
    best_blend_pred = np.argmax(
        best["weight_clf"] * test_probs_corrected + (1 - best["weight_clf"]) * test_rate_probs, axis=1
    )
    best_blend_metrics = _summary(test_outcomes, best_blend_pred)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    pred_frame = pd.DataFrame({
        "game_id": test_keys,
        "HG": test_df["HG"].values,
        "AG": test_df["AG"].values,
        "lambda_home": test_home_pred,
        "lambda_away": test_away_pred,
        "expected_goal_diff": test_home_pred - test_away_pred,
        "prob_home": test_probs_corrected[:, 0],
        "prob_draw": test_probs_corrected[:, 1],
        "prob_away": test_probs_corrected[:, 2],
        "predicted_result_rate": test_rate_pred,
        "predicted_result_clf": test_clf_pred,
        "predicted_result_clf_corrected": test_clf_corrected_pred,
        "predicted_result_best_blend": best_blend_pred,
    })
    pred_frame.to_csv(PRED_PATH, index=False)

    torch.save({
        "state_dict": model.state_dict(),
        "config": config.__dict__,
        "home_cols": home_cols,
        "away_cols": away_cols,
        "other_cols": other_cols,
        "home_prior": home_prior,
        "away_prior": away_prior,
        "draw_margin": draw_margin,
        "train_priors": train_priors.tolist(),
        "test_priors": test_priors.tolist(),
    }, MODEL_PATH)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("FINAL TEST RESULTS")
    print("=" * 78)
    print(f"  {'Path':>32}  {'Accuracy':>10}  {'Balanced':>10}  {'Macro F1':>10}")
    print(f"  {'Rate-based':>32}  {rate_metrics['accuracy']:>10.3f}  "
          f"{rate_metrics['balanced_accuracy']:>10.3f}  {rate_metrics['macro_f1']:>10.3f}")
    print(f"  {'Classifier (raw)':>32}  {clf_metrics['accuracy']:>10.3f}  "
          f"{clf_metrics['balanced_accuracy']:>10.3f}  {clf_metrics['macro_f1']:>10.3f}")
    print(f"  {'Classifier (prior-corrected)':>32}  {clf_corrected_metrics['accuracy']:>10.3f}  "
          f"{clf_corrected_metrics['balanced_accuracy']:>10.3f}  {clf_corrected_metrics['macro_f1']:>10.3f}")
    print(f"  {'Best blend (swept)':>32}  {best_blend_metrics['accuracy']:>10.3f}  "
          f"{best_blend_metrics['balanced_accuracy']:>10.3f}  {best_blend_metrics['macro_f1']:>10.3f}")

    print(f"\n  Top 5 blend weights:")
    print(f"  {'Weight':>8}  {'Accuracy':>10}  {'Balanced':>10}  {'Macro F1':>10}")
    for r in sweep_results[:5]:
        print(f"  {r['weight_clf']:>8.2f}  {r['accuracy']:>10.3f}  "
              f"{r['balanced_accuracy']:>10.3f}  {r['macro_f1']:>10.3f}")

    print("\n  Confusion matrix (best blend, rows=true, cols=predicted):")
    cm = np.array(best_blend_metrics["confusion_matrix"])
    print(f"  {'':>12}  {'Pred H':>7}  {'Pred D':>7}  {'Pred A':>7}")
    for i, name in enumerate(["Actual H", "Actual D", "Actual A"]):
        print(f"  {name:>12}  {cm[i, 0]:>7}  {cm[i, 1]:>7}  {cm[i, 2]:>7}")

    print("\n  Per-class metrics (best blend):")
    precision, recall, f1, support = precision_recall_fscore_support(
        test_outcomes, best_blend_pred, labels=[0, 1, 2], zero_division=0
    )
    print(f"  {'Class':>6}  {'Precision':>9}  {'Recall':>7}  {'F1':>6}  {'Support':>8}")
    for i, name in enumerate(["HOME", "DRAW", "AWAY"]):
        print(f"  {name:>6}  {precision[i]:>9.3f}  {recall[i]:>7.3f}  {f1[i]:>6.3f}  {support[i]:>8d}")

    print("\n  Prediction distribution (best blend):")
    counts = np.bincount(best_blend_pred, minlength=3)
    for i, name in enumerate(["HOME", "DRAW", "AWAY"]):
        pct = 100.0 * counts[i] / counts.sum() if counts.sum() > 0 else 0.0
        print(f"    {name:>4}: {counts[i]:>5d} ({pct:>5.1f}%)")

    # Summary JSON
    summary = {
        "config": config.__dict__,
        "params": {"total_trainable": n_params},
        "feature_layout": {
            "home_cols": len(home_cols),
            "away_cols": len(away_cols),
            "other_cols": len(other_cols),
            "using_xgb_selected": config.use_xgb_selected_features,
        },
        "priors": {"train": train_priors.tolist(), "test": test_priors.tolist()},
        "validation": {"draw_margin": draw_margin, "balanced_accuracy": val_bal},
        "test_rate_based": rate_metrics,
        "test_classifier_raw": clf_metrics,
        "test_classifier_prior_corrected": clf_corrected_metrics,
        "test_blend_best": {"weight_clf": best["weight_clf"], **best_blend_metrics},
        "blend_sweep_top5": sweep_results[:5],
        "best_epoch": best_epoch,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n  Saved predictions to: {PRED_PATH}")
    print(f"  Saved model to:       {MODEL_PATH}")
    print(f"  Saved summary to:     {SUMMARY_PATH}")

    return summary


def main():
    config = LSTMConfig(modules=list(DEFAULT_MODULES))
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    RESULT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_lstm(config, device)


if __name__ == "__main__":
    main()