#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Longitudinal LSTM modules + fusion.

Pipeline
--------
0. Run a2_prepare_lstm_data.py first: it derives final_data/lstm_input_<module>_*.csv,
   where every game carries its current features plus 5 history sequences
   (home/away form, home/away season, head-to-head) built from earlier games.
1. Train one LSTM per module. Each runs over the history sequences, combines them
   with the current-game features and predicts (lambda_home, lambda_away).
2. Combine the module predictions with a fusion model. Styles:

   "mlp"                 - MLP over the concatenated module predictions.
   "lstm"                - LSTM over the ordered sequence of module predictions.
   "matchup_prior_mlp"   - Anchor on one module's prediction, correct it via
                           a small MLP residual over the other modules.
   "matchup_prior_lstm"  - Same idea, but the residual comes from an LSTM
                           over the sequence of the other modules.

The matchup-prior styles exploit the observation that a single module
(typically "matchup") can outperform prediction-level fusion. Instead of
letting the fusion learn the whole prediction from scratch, we anchor on
that module and let the others apply a bounded multiplicative correction
in log-space. At initialisation the residual is zero, so the model starts
from the anchor and learns how much to trust the other modules.

Configuration
-------------
The anchor module must be the first entry of `fusion_modules` when using a
"matchup_prior_*" style. The code checks this and raises otherwise.
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


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
FINAL_DATA_DIR = PROJECT_DIR / "final_data"
RESULT_DATA_DIR = PROJECT_DIR / "result_data"
MISC_DIR = PROJECT_DIR / "misc"

META_PATH = MISC_DIR / "lstm_module_metadata.json"
BASELINE_SUMMARY_PATH = RESULT_DATA_DIR / "lstm_fusion_summary.json"
BASELINE_MODEL_PATH = RESULT_DATA_DIR / "lstm_fusion_model.pt"
BASELINE_PRED_PATH = RESULT_DATA_DIR / "lstm_fusion_predictions.csv"

MODULES = ["lineup", "player_stats", "skill_matchups", "match_stats",
           "matchup", "league", "season"]


@dataclass
class BaselineConfig:
    epochs: int = 100
    batch_size: int = 512
    learning_rate: float = 5e-4
    weight_decay: float = 1e-3
    seed: int = 123
    val_ratio: float = 0.1
    patience: int = 15
    max_goals: int = 12

    # --- Per-module longitudinal LSTM -----------------------------------
    seq_proj: int = 24             # shared per-step encoder width
    seq_hidden: int = 24           # LSTM hidden size, one LSTM per history series
    seq_layers: int = 1
    cur_hidden: int = 24           # encoder width for the current-game features
    module_dropout: float = 0.3

    # --- Fusion: which modules and which style --------------------------
    # For "matchup_prior_*" styles, the first entry is the anchor.
    fusion_modules: tuple = (
        "matchup", "lineup", "player_stats", "skill_matchups", "match_stats", "season"
    )
    fusion_style: str = "weighted_blend"
    # "weighted_blend" | "mlp" | "lstm" | "matchup_prior_mlp" | "matchup_prior_lstm"

    # --- Out-of-fold module predictions for fusion training -------------
    # The fusion is fit on K-fold out-of-fold module predictions so it sees the
    # modules' real generalisation quality instead of their overfit train fits.
    # <= 1 falls back to in-sample predictions.
    fusion_oof_folds: int = 5

    # --- weighted_blend fusion -----------------------------------------
    blend_temperature: float = 1.0     # init weights = softmax(-(module NLL - best NLL) / T); large T = uniform
    # Pull the blend toward the uniform average. Module quality measured on training-side
    # rows can overstate modules whose train/test inputs differ (e.g. embeddings), and a
    # uniform geometric mean is a robust fallback that rarely loses to the best module.
    blend_shrinkage: float = 1.0
    blend_lr: float = 5e-3             # learning rate of the blend weights
    blend_residual_hidden: int = 0     # >0 adds a small non-linear correction (risky under shift)
    blend_residual_scale: float = 0.3  # residual is tanh-bounded to +-scale in log-rate

    # --- MLP fusion hyperparameters ------------------------------------
    fusion_hidden: int = 16
    fusion_dropout: float = 0.3

    # --- LSTM fusion hyperparameters -----------------------------------
    lstm_hidden: int = 32
    lstm_layers: int = 1
    lstm_dropout: float = 0.3

    # --- Anchor correction strength (matchup_prior_* styles) -----------
    # The residual is passed through tanh and multiplied by this scale
    # before exponentiation. residual_scale=0.5 -> correction factor in
    # [exp(-0.5), exp(0.5)] ≈ [0.61, 1.65].
    matchup_prior_scale: float = 0.5


# ============================================================================
# DATASETS
# ============================================================================

class SimpleDataset(Dataset):
    def __init__(self, frame, feature_cols):
        self.x = torch.from_numpy(frame[feature_cols].to_numpy(dtype=np.float32))
        self.hg = torch.tensor(frame["HG"].astype(float).values, dtype=torch.float32)
        self.ag = torch.tensor(frame["AG"].astype(float).values, dtype=torch.float32)
        self.game_id = frame["game_id"].astype(str).values
        self.outcome = torch.tensor(
            np.where(frame["HG"].values > frame["AG"].values, 0,
                     np.where(frame["HG"].values == frame["AG"].values, 1, 2)),
            dtype=torch.long
        )

    def __len__(self):
        return len(self.game_id)

    def __getitem__(self, idx):
        return {
            "game_id": self.game_id[idx],
            "x": self.x[idx],
            "HG": self.hg[idx],
            "AG": self.ag[idx],
            "outcome": self.outcome[idx],
        }


class FusionDataset(Dataset):
    def __init__(self, frame, feature_cols):
        self.x = torch.tensor(frame[feature_cols].astype(float).values, dtype=torch.float32)
        self.hg = torch.tensor(frame["HG"].astype(float).values, dtype=torch.float32)
        self.ag = torch.tensor(frame["AG"].astype(float).values, dtype=torch.float32)
        self.game_id = frame["game_id"].astype(str).values
        self.outcome = torch.tensor(
            np.where(frame["HG"].values > frame["AG"].values, 0,
                     np.where(frame["HG"].values == frame["AG"].values, 1, 2)),
            dtype=torch.long
        )

    def __len__(self):
        return len(self.game_id)

    def __getitem__(self, idx):
        return {
            "game_id": self.game_id[idx],
            "x": self.x[idx],
            "HG": self.hg[idx],
            "AG": self.ag[idx],
            "outcome": self.outcome[idx],
        }


def collate_simple_batch(batch):
    return {
        "game_id": [x["game_id"] for x in batch],
        "x": torch.stack([x["x"] for x in batch]),
        "HG": torch.stack([x["HG"] for x in batch]),
        "AG": torch.stack([x["AG"] for x in batch]),
        "outcome": torch.stack([x["outcome"] for x in batch]),
    }


# ============================================================================
# PER-MODULE MODEL
# ============================================================================

class SequenceRateModel(nn.Module):
    """Longitudinal rate model: log(lambda) = log(prior) + head([current, LSTM(series)...]).

    The flat input is [current features | n_series * seq_len * step_dim]. A shared linear
    encoder embeds every step (padded steps are masked via the last step dim, `valid`);
    one LSTM per series reads its steps oldest -> newest.
    """
    def __init__(self, n_current, n_series, seq_len, step_dim,
                 home_prior=1.35, away_prior=1.10,
                 proj=24, hidden=24, num_layers=1, cur_hidden=24, dropout=0.3):
        super().__init__()
        self.n_current, self.n_series = n_current, n_series
        self.seq_len, self.step_dim = seq_len, step_dim
        self.register_buffer(
            "log_home_prior", torch.tensor(float(np.log(home_prior)))
        )
        self.register_buffer(
            "log_away_prior", torch.tensor(float(np.log(away_prior)))
        )
        self.step_enc = nn.Sequential(nn.Linear(step_dim, proj), nn.ReLU(), nn.Dropout(dropout))
        self.lstms = nn.ModuleList([
            nn.LSTM(proj, hidden, num_layers=num_layers, batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0)
            for _ in range(n_series)
        ])
        self.cur_enc = nn.Sequential(
            nn.Linear(n_current, cur_hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(cur_hidden + n_series * hidden, 2)
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        B = x.shape[0]
        cur = x[:, :self.n_current]
        seq = x[:, self.n_current:].reshape(B, self.n_series, self.seq_len, self.step_dim)
        valid = seq[..., -1:]
        feats = [self.cur_enc(cur)]
        for s, lstm in enumerate(self.lstms):
            steps = self.step_enc(seq[:, s]) * valid[:, s]
            _, (h_n, _) = lstm(steps)
            feats.append(h_n[-1])
        logits = self.head(torch.cat(feats, dim=1))
        return (torch.exp(self.log_home_prior + logits[:, 0]),
                torch.exp(self.log_away_prior + logits[:, 1]))


# ============================================================================
# FUSION MODELS
# ============================================================================

class WeightedBlendFusionModel(nn.Module):
    """Convex (softmax-weighted) geometric blend of the module rates.

    log(lambda) = sum_m w_m * log(lambda_m) [+ small bounded residual], w >= 0, sum w = 1.
    The blend can never be bolder than its inputs, so a module that looks too good on
    in-distribution rows cannot be amplified on the shifted test season. Weights start
    from out-of-fold module quality and can shift fully onto the best module.
    """
    def __init__(self, n_modules, init_weights, home_prior=1.35, away_prior=1.10,
                 hidden=0, dropout=0.3, residual_scale=0.3):
        super().__init__()
        self.n_modules = n_modules
        self.residual_scale = residual_scale
        self.register_buffer(
            "log_home_prior", torch.tensor(float(np.log(home_prior)))
        )
        self.register_buffer(
            "log_away_prior", torch.tensor(float(np.log(away_prior)))
        )
        theta = torch.log(torch.as_tensor(init_weights, dtype=torch.float32).clamp(min=1e-6))
        self.theta_home = nn.Parameter(theta.clone())
        self.theta_away = nn.Parameter(theta.clone())
        self.residual = None
        if hidden > 0:
            self.residual = nn.Sequential(
                nn.Linear(2 * n_modules, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, 2),
            )
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)

    def shrinkage(self):
        """Squared distance of the blend weights from the uniform blend."""
        uniform = 1.0 / self.n_modules
        return ((torch.softmax(self.theta_home, dim=0) - uniform) ** 2).sum() + \
               ((torch.softmax(self.theta_away, dim=0) - uniform) ** 2).sum()

    def forward(self, x):
        seq = x.view(x.shape[0], self.n_modules, 3)
        log_h = torch.log(seq[:, :, 0].clamp(min=1e-6)) - self.log_home_prior
        log_a = torch.log(seq[:, :, 1].clamp(min=1e-6)) - self.log_away_prior
        out_h = (log_h * torch.softmax(self.theta_home, dim=0)).sum(dim=1)
        out_a = (log_a * torch.softmax(self.theta_away, dim=0)).sum(dim=1)
        if self.residual is not None:
            res = torch.tanh(self.residual(torch.cat([log_h, log_a], dim=1))) * self.residual_scale
            out_h = out_h + res[:, 0]
            out_a = out_a + res[:, 1]
        return (torch.exp(self.log_home_prior + out_h),
                torch.exp(self.log_away_prior + out_a))


class MLPFusionModel(nn.Module):
    """Concatenate module (lambda_h, lambda_a, diff) and predict final rates."""
    def __init__(self, n_modules, home_prior=1.35, away_prior=1.10,
                 hidden=32, dropout=0.1):
        super().__init__()
        self.register_buffer(
            "log_home_prior", torch.tensor(float(np.log(home_prior)))
        )
        self.register_buffer(
            "log_away_prior", torch.tensor(float(np.log(away_prior)))
        )
        in_dim = n_modules * 3
        if hidden > 0:
            self.trunk = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            )
            head_in = hidden
        else:
            self.trunk = nn.Identity()
            head_in = in_dim
        self.head = nn.Linear(head_in, 2)
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        h = self.trunk(x.view(x.shape[0], -1))
        logits = self.head(h)
        return (torch.exp(self.log_home_prior + logits[:, 0]),
                torch.exp(self.log_away_prior + logits[:, 1]))


class LSTMFusionModel(nn.Module):
    """LSTM over the ordered sequence of module predictions.

    Each module contributes one timestep of shape (3,) = (lambda_h, lambda_a,
    diff). The final hidden state of the LSTM is projected to two log-rate
    offsets, added to the training priors.
    """
    def __init__(self, n_modules, home_prior=1.35, away_prior=1.10,
                 hidden=32, num_layers=1, dropout=0.3):
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
        self.head = nn.Linear(hidden, 2)
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        B = x.shape[0]
        seq = x.view(B, self.n_modules, 3)
        seq = self.input_dropout(seq)
        _, (h_n, _) = self.lstm(seq)
        h = h_n[-1]
        logits = self.head(h)
        return (torch.exp(self.log_home_prior + logits[:, 0]),
                torch.exp(self.log_away_prior + logits[:, 1]))


class MatchupPriorMLPModel(nn.Module):
    """Anchor on the first module; MLP residual over the others."""
    def __init__(self, n_other, residual_scale=0.5, hidden=16, dropout=0.3):
        super().__init__()
        self.residual_scale = residual_scale
        in_dim = n_other * 3
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )
        nn.init.zeros_(self.trunk[-1].weight)
        nn.init.zeros_(self.trunk[-1].bias)

    def forward(self, x):
        B = x.shape[0]
        seq = x.view(B, -1, 3)
        anchor = seq[:, 0, :]
        other = seq[:, 1:, :].reshape(B, -1)
        residual = torch.tanh(self.trunk(other)) * self.residual_scale
        anchor_lh = anchor[:, 0].clamp(min=1e-6)
        anchor_la = anchor[:, 1].clamp(min=1e-6)
        return (anchor_lh * torch.exp(residual[:, 0]),
                anchor_la * torch.exp(residual[:, 1]))


class MatchupPriorLSTMModel(nn.Module):
    """Anchor on the first module; LSTM residual over the others.

    The correction is multiplicative in rate-space:
        final_lambda_h = anchor_lambda_h * exp(residual_h)
        final_lambda_a = anchor_lambda_a * exp(residual_a)
    where residual ∈ [-scale, +scale] via tanh. At init residual = 0, so
    final_lambda = anchor_lambda and the model starts from the anchor.
    """
    def __init__(self, n_other, residual_scale=0.5,
                 hidden=32, num_layers=1, dropout=0.3):
        super().__init__()
        self.residual_scale = residual_scale
        self.lstm = nn.LSTM(
            input_size=3,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.input_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        B = x.shape[0]
        seq = x.view(B, -1, 3)
        anchor = seq[:, 0, :]
        other = seq[:, 1:, :]
        other = self.input_dropout(other)
        _, (h_n, _) = self.lstm(other)
        h = h_n[-1]
        residual = torch.tanh(self.head(h)) * self.residual_scale
        anchor_lh = anchor[:, 0].clamp(min=1e-6)
        anchor_la = anchor[:, 1].clamp(min=1e-6)
        return (anchor_lh * torch.exp(residual[:, 0]),
                anchor_la * torch.exp(residual[:, 1]))


# ============================================================================
# LOSS
# ============================================================================

def poisson_nll_loss(lam_h, lam_a, hg, ag):
    return F.poisson_nll_loss(lam_h, hg, log_input=False, full=True) + \
           F.poisson_nll_loss(lam_a, ag, log_input=False, full=True)


# ============================================================================
# METRICS
# ============================================================================

def poisson_outcome_probs(lam_h, lam_a, max_goals=12):
    device = lam_h.device
    goals = torch.arange(0, max_goals + 1, device=device).float()
    log_fact = torch.lgamma(goals + 1)

    log_p_h = -lam_h.unsqueeze(1) + goals.unsqueeze(0) * torch.log(lam_h.unsqueeze(1) + 1e-8) - log_fact.unsqueeze(0)
    log_p_a = -lam_a.unsqueeze(1) + goals.unsqueeze(0) * torch.log(lam_a.unsqueeze(1) + 1e-8) - log_fact.unsqueeze(0)

    p_h = torch.exp(log_p_h)
    p_a = torch.exp(log_p_a)
    mat = p_h.unsqueeze(2) * p_a.unsqueeze(1)

    G = max_goals + 1
    idx = torch.arange(G, device=device)
    home_mask = idx.unsqueeze(1) > idx.unsqueeze(0)
    draw_mask = idx.unsqueeze(1) == idx.unsqueeze(0)
    away_mask = idx.unsqueeze(1) < idx.unsqueeze(0)

    p_home = (mat * home_mask.unsqueeze(0)).sum(dim=(1, 2))
    p_draw = (mat * draw_mask.unsqueeze(0)).sum(dim=(1, 2))
    p_away = (mat * away_mask.unsqueeze(0)).sum(dim=(1, 2))

    probs = torch.stack([p_home, p_draw, p_away], dim=1)
    return probs / probs.sum(dim=1, keepdim=True)


def compute_metrics(model, loader, device, max_goals=12):
    model.eval()
    losses = []
    all_lam_h, all_lam_a = [], []
    pred_outcomes, true_outcomes = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            hg = batch["HG"].to(device)
            ag = batch["AG"].to(device)
            outcome = batch["outcome"].to(device)

            lam_h, lam_a = model(x)
            loss = poisson_nll_loss(lam_h, lam_a, hg, ag)
            losses.append(float(loss.item()))

            probs = poisson_outcome_probs(lam_h, lam_a, max_goals)
            pred = probs.argmax(dim=1)

            pred_outcomes.extend(pred.cpu().numpy().tolist())
            true_outcomes.extend(outcome.cpu().numpy().tolist())
            all_lam_h.append(lam_h.cpu())
            all_lam_a.append(lam_a.cpu())

    all_lam_h = torch.cat(all_lam_h)
    all_lam_a = torch.cat(all_lam_a)
    accuracy = float((np.array(pred_outcomes) == np.array(true_outcomes)).mean())

    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy,
        "mean_home_rate": float(all_lam_h.mean().item()),
        "mean_away_rate": float(all_lam_a.mean().item()),
        "std_home_rate": float(all_lam_h.std().item()),
        "std_away_rate": float(all_lam_a.std().item()),
    }


# ============================================================================
# UTILITIES
# ============================================================================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_module_meta(module_name):
    if not META_PATH.exists():
        raise FileNotFoundError(f"Missing metadata file {META_PATH}.")
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    if module_name not in meta.get("modules", {}):
        raise KeyError(f"Module '{module_name}' not found.")
    return meta["modules"][module_name]


def load_module_frames(module_name):
    module_meta = load_module_meta(module_name)
    key_col = module_meta["key_col"]
    feature_cols = module_meta["feature_cols"]

    train_path = FINAL_DATA_DIR / f"lstm_input_{module_name}_TRAIN.csv"
    test_path = FINAL_DATA_DIR / f"lstm_input_{module_name}_TEST.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing prepared inputs for {module_name}.")

    cols = pd.read_csv(train_path, nrows=0).columns
    dtypes = {c: np.float32 for c in cols if c != key_col}
    train_df = pd.read_csv(train_path, dtype=dtypes)
    test_df = pd.read_csv(test_path, dtype=dtypes)

    for df in (train_df, test_df):
        df[feature_cols] = df[feature_cols].fillna(0.0)

    return train_df, test_df, key_col, feature_cols


def predict_frame(model, df, feature_cols, config, device):
    ds = SimpleDataset(df, feature_cols)
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False,
                        collate_fn=collate_simple_batch)
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            lam_h, lam_a = model(x)
            for i, gid in enumerate(batch["game_id"]):
                rows.append({
                    "game_id": gid,
                    "HG": float(batch["HG"][i]),
                    "AG": float(batch["AG"][i]),
                    "lambda_home": float(lam_h[i]),
                    "lambda_away": float(lam_a[i]),
                })
    return pd.DataFrame(rows)


# ============================================================================
# TRAINING — SINGLE MODULE
# ============================================================================

def fit_module_lstm(module_name, fit_df, val_df, feature_cols, config, device, verbose=True):
    """Train one longitudinal LSTM with early stopping on val_df.

    Inputs are already standardised upstream (a_prepare_data / a2_prepare_lstm_data).
    Returns (model, best_epoch).
    """
    meta = load_module_meta(module_name)
    home_prior = float(fit_df["HG"].mean())
    away_prior = float(fit_df["AG"].mean())

    train_ds = SimpleDataset(fit_df, feature_cols)
    val_ds = SimpleDataset(val_df, feature_cols)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                              collate_fn=collate_simple_batch)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                            collate_fn=collate_simple_batch)

    model = SequenceRateModel(
        meta["n_current"], meta["n_series"], meta["seq_len"], meta["step_dim"],
        home_prior, away_prior,
        proj=config.seq_proj, hidden=config.seq_hidden, num_layers=config.seq_layers,
        cur_hidden=config.cur_hidden, dropout=config.module_dropout,
    ).to(device)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n  [{module_name}] Training (Current feats: {meta['n_current']}, "
              f"Series: {meta['n_series']}x{meta['seq_len']}x{meta['step_dim']}, "
              f"Rows: {len(train_ds)}, Params: {n_params:,})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            hg = batch["HG"].to(device)
            ag = batch["AG"].to(device)
            optimizer.zero_grad()
            lam_h, lam_a = model(x)
            loss = poisson_nll_loss(lam_h, lam_a, hg, ag)
            loss.backward()
            optimizer.step()

        val_m = compute_metrics(model, val_loader, device, config.max_goals)

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            if verbose:
                print(f"    Early stopping at epoch {epoch}")
            break

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"    Ep {epoch:03d} | Val Loss {val_m['loss']:.4f} | "
                  f"Acc {val_m['accuracy']:.3f} | "
                  f"std(λh) {val_m['std_home_rate']:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_epoch


def oof_module_predictions(module_name, train_df_full, feature_cols, config, device):
    """Out-of-fold predictions for every training row.

    Folds are contiguous date blocks (game_id starts with YYYYMMDD), the same for every
    module. Random folds leak across neighbouring matches and overstate modules that
    memorise teams or form, which then get over-weighted on the future test season.
    """
    n_folds = config.fusion_oof_folds
    dates = pd.to_numeric(train_df_full["game_id"].astype(str).str[:8], errors="coerce")
    edges = dates.quantile(np.linspace(0, 1, n_folds + 1)[1:-1]).to_numpy()
    fold_of_row = np.searchsorted(edges, dates.fillna(dates.max()).to_numpy(), side="right")

    parts = []
    for k in range(n_folds):
        held = train_df_full[fold_of_row == k]
        pool = train_df_full[fold_of_row != k]
        order = np.random.default_rng(config.seed + k).permutation(len(pool))
        n_val = int(len(pool) * config.val_ratio)
        val_part = pool.iloc[order[:n_val]]
        fit_part = pool.iloc[order[n_val:]]

        model, _ = fit_module_lstm(
            module_name, fit_part, val_part, feature_cols, config, device, verbose=False
        )
        parts.append(predict_frame(model, held, feature_cols, config, device))
    return pd.concat(parts, ignore_index=True)


def train_single_module(module_name, config, device):
    train_df_full, test_df, key_col, feature_cols = load_module_frames(module_name)

    rng = np.random.default_rng(config.seed)
    indices = rng.permutation(len(train_df_full))
    n_val = int(len(train_df_full) * config.val_ratio)
    val_df = train_df_full.iloc[indices[:n_val]].reset_index(drop=True)
    train_df = train_df_full.iloc[indices[n_val:]].reset_index(drop=True)

    model, best_epoch = fit_module_lstm(
        module_name, train_df, val_df, feature_cols, config, device
    )

    test_ds = SimpleDataset(test_df, feature_cols)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False,
                             collate_fn=collate_simple_batch)

    test_m = compute_metrics(model, test_loader, device, config.max_goals)
    print(f"  [{module_name}] Test | Loss {test_m['loss']:.4f} | "
          f"Acc {test_m['accuracy']:.3f} | std(λh) {test_m['std_home_rate']:.4f}")

    model_path = RESULT_DATA_DIR / f"lstm_module_{module_name}_model.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "module_name": module_name,
        "best_epoch": best_epoch,
    }, model_path)

    train_pred = predict_frame(model, train_df_full, feature_cols, config, device)
    test_pred = predict_frame(model, test_df, feature_cols, config, device)

    train_pred.to_csv(
        RESULT_DATA_DIR / f"lstm_module_{module_name}_train_predictions.csv",
        index=False,
    )
    test_pred.to_csv(
        RESULT_DATA_DIR / f"lstm_module_{module_name}_predictions.csv",
        index=False,
    )

    oof_path = RESULT_DATA_DIR / f"lstm_module_{module_name}_oof_predictions.csv"
    if config.fusion_oof_folds > 1:
        oof = oof_module_predictions(module_name, train_df_full, feature_cols, config, device)
        oof.to_csv(oof_path, index=False)
        print(f"  [{module_name}] Saved {len(oof)} out-of-fold predictions "
              f"({config.fusion_oof_folds} folds)")
    else:
        oof_path.unlink(missing_ok=True)  # never let a stale OOF file reach the fusion

    return model, test_ds, best_epoch, test_m


# ============================================================================
# FUSION
# ============================================================================

def build_fusion_frame(split, fusion_modules):
    """Merge per-module predictions. Column order matches fusion_modules."""
    dfs = []
    for mn in fusion_modules:
        if split == "train":
            path = RESULT_DATA_DIR / f"lstm_module_{mn}_oof_predictions.csv"
            if not path.exists():
                path = RESULT_DATA_DIR / f"lstm_module_{mn}_train_predictions.csv"
        else:
            path = RESULT_DATA_DIR / f"lstm_module_{mn}_predictions.csv"

        if not path.exists():
            print(f"    Missing {path}")
            return None, []

        df = pd.read_csv(path, low_memory=False)
        df = df.rename(columns={
            "lambda_home": f"{mn}_lambda_home",
            "lambda_away": f"{mn}_lambda_away",
        })
        df[f"{mn}_diff"] = df[f"{mn}_lambda_home"] - df[f"{mn}_lambda_away"]
        dfs.append(df[["game_id", "HG", "AG",
                       f"{mn}_lambda_home", f"{mn}_lambda_away", f"{mn}_diff"]])

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df.drop(columns=["HG", "AG"]),
                              on="game_id", how="inner")

    feature_cols = []
    for mn in fusion_modules:
        feature_cols += [f"{mn}_lambda_home", f"{mn}_lambda_away", f"{mn}_diff"]

    return merged, feature_cols


def build_fusion_model(config, n_modules, home_prior, away_prior, init_weights=None):
    """Instantiate the fusion model according to config.fusion_style."""
    style = config.fusion_style
    if style == "weighted_blend":
        if init_weights is None:
            init_weights = np.full(n_modules, 1.0 / n_modules)
        return WeightedBlendFusionModel(
            n_modules, init_weights, home_prior, away_prior,
            hidden=config.blend_residual_hidden, dropout=config.fusion_dropout,
            residual_scale=config.blend_residual_scale,
        )
    if style == "mlp":
        return MLPFusionModel(
            n_modules, home_prior, away_prior,
            hidden=config.fusion_hidden, dropout=config.fusion_dropout,
        )
    if style == "lstm":
        return LSTMFusionModel(
            n_modules, home_prior, away_prior,
            hidden=config.lstm_hidden,
            num_layers=config.lstm_layers,
            dropout=config.lstm_dropout,
        )
    if style == "matchup_prior_mlp":
        if n_modules < 2:
            raise ValueError("matchup_prior_mlp requires at least 2 modules.")
        return MatchupPriorMLPModel(
            n_other=n_modules - 1,
            residual_scale=config.matchup_prior_scale,
            hidden=config.fusion_hidden,
            dropout=config.fusion_dropout,
        )
    if style == "matchup_prior_lstm":
        if n_modules < 2:
            raise ValueError("matchup_prior_lstm requires at least 2 modules.")
        return MatchupPriorLSTMModel(
            n_other=n_modules - 1,
            residual_scale=config.matchup_prior_scale,
            hidden=config.lstm_hidden,
            num_layers=config.lstm_layers,
            dropout=config.lstm_dropout,
        )
    raise ValueError(f"Unknown fusion_style: {style!r}")


def train_fusion(config, device):
    fusion_modules = list(config.fusion_modules)
    train_df, feature_cols = build_fusion_frame("train", fusion_modules)
    test_df, _ = build_fusion_frame("test", fusion_modules)

    if train_df is None or test_df is None:
        print("  Fusion skipped: missing module predictions.")
        return None

    # Most recent rows validate, as the test season is the future.
    train_df = train_df.sort_values("game_id").reset_index(drop=True)
    n_val = int(len(train_df) * config.val_ratio)
    fusion_train = train_df.iloc[:len(train_df) - n_val].reset_index(drop=True)
    fusion_val = train_df.iloc[len(train_df) - n_val:].reset_index(drop=True)

    home_prior = float(fusion_train["HG"].mean())
    away_prior = float(fusion_train["AG"].mean())

    # Per-module Poisson NLL on the most recent (validation) rows; drives the blend initialisation.
    hg_t = torch.tensor(fusion_val["HG"].to_numpy(), dtype=torch.float32)
    ag_t = torch.tensor(fusion_val["AG"].to_numpy(), dtype=torch.float32)
    module_nll = {}
    for mn in fusion_modules:
        lh = torch.tensor(fusion_val[f"{mn}_lambda_home"].to_numpy(), dtype=torch.float32)
        la = torch.tensor(fusion_val[f"{mn}_lambda_away"].to_numpy(), dtype=torch.float32)
        module_nll[mn] = float(poisson_nll_loss(lh, la, hg_t, ag_t))
    nll = np.array([module_nll[mn] for mn in fusion_modules])
    init_weights = np.exp(-(nll - nll.min()) / config.blend_temperature)
    init_weights = init_weights / init_weights.sum()

    train_ds = FusionDataset(fusion_train, feature_cols)
    val_ds = FusionDataset(fusion_val, feature_cols)
    test_ds = FusionDataset(test_df, feature_cols)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                              collate_fn=collate_simple_batch)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                            collate_fn=collate_simple_batch)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False,
                             collate_fn=collate_simple_batch)

    model = build_fusion_model(config, len(fusion_modules),
                               home_prior, away_prior, init_weights).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n  [fusion] Style: {config.fusion_style}")
    print(f"  [fusion] Modules ({len(fusion_modules)}): {fusion_modules}")
    print(f"  [fusion] Module NLL on fusion-val rows: "
          + ", ".join(f"{mn}={module_nll[mn]:.4f}" for mn in fusion_modules))
    if config.fusion_style.startswith("matchup_prior"):
        print(f"  [fusion] Anchor module: {fusion_modules[0]}")
        print(f"  [fusion] Residual scale: ±{config.matchup_prior_scale} "
              f"(correction factor ∈ [exp(-s), exp(s)])")
    print(f"  [fusion] Input dim: {len(feature_cols)}, "
          f"Params: {n_params:,}")

    # Blend weights must not be weight-decayed and learn faster.
    blend_names = {"theta_home", "theta_away"}
    blend_params = [p for n, p in model.named_parameters() if n in blend_names]
    other_params = [p for n, p in model.named_parameters() if n not in blend_names]
    param_groups = []
    if other_params:
        param_groups.append({"params": other_params})
    if blend_params:
        param_groups.append({"params": blend_params, "lr": config.blend_lr, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(param_groups, lr=config.learning_rate,
                                  weight_decay=config.weight_decay)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            hg = batch["HG"].to(device)
            ag = batch["AG"].to(device)
            optimizer.zero_grad()
            lam_h, lam_a = model(x)
            loss = poisson_nll_loss(lam_h, lam_a, hg, ag)
            if hasattr(model, "shrinkage"):
                loss = loss + config.blend_shrinkage * model.shrinkage()
            loss.backward()
            optimizer.step()

        val_m = compute_metrics(model, val_loader, device, config.max_goals)

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            print(f"    Early stopping at epoch {epoch}")
            break

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Ep {epoch:03d} | Val Loss {val_m['loss']:.4f} | "
                  f"Acc {val_m['accuracy']:.3f} | "
                  f"std(λh) {val_m['std_home_rate']:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    if config.fusion_style == "weighted_blend":
        w_home = torch.softmax(model.theta_home, dim=0).detach().cpu().numpy()
        print("  [fusion] Blend weights (home): "
              + ", ".join(f"{mn}={w:.2f}" for mn, w in zip(fusion_modules, w_home)))

    test_m = compute_metrics(model, test_loader, device, config.max_goals)
    print(f"  [fusion] Test | Loss {test_m['loss']:.4f} | "
          f"Acc {test_m['accuracy']:.3f} | std(λh) {test_m['std_home_rate']:.4f} | "
          f"λh {test_m['mean_home_rate']:.2f} λa {test_m['mean_away_rate']:.2f}")

    model.eval()
    rows = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            lam_h, lam_a = model(x)
            for i, gid in enumerate(batch["game_id"]):
                rows.append({
                    "game_id": gid,
                    "HG": float(batch["HG"][i]),
                    "AG": float(batch["AG"][i]),
                    "lambda_home": float(lam_h[i]),
                    "lambda_away": float(lam_a[i]),
                    "expected_goal_diff": float(lam_h[i] - lam_a[i]),
                })

    pd.DataFrame(rows).to_csv(BASELINE_PRED_PATH, index=False)
    torch.save({"state_dict": model.state_dict(),
                "fusion_style": config.fusion_style,
                "fusion_modules": fusion_modules}, BASELINE_MODEL_PATH)

    return test_m


# ============================================================================
# MAIN
# ============================================================================

def main():
    config = BaselineConfig()

    # Sanity: matchup-prior styles require the anchor to be first.
    if config.fusion_style.startswith("matchup_prior"):
        if not config.fusion_modules:
            raise ValueError("fusion_modules is empty.")
        # The first entry is the anchor by construction.

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Config: epochs={config.epochs} lr={config.learning_rate} "
          f"wd={config.weight_decay} patience={config.patience} "
          f"mod_dropout={config.module_dropout} seq_hidden={config.seq_hidden}")
    print(f"Fusion style: {config.fusion_style}")
    print(f"Fusion modules: {list(config.fusion_modules)}")

    RESULT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("LONGITUDINAL LSTM MODULES")
    print("=" * 70)

    results = {}

    for module_name in MODULES:
        try:
            model, test_ds, best_epoch, test_m = train_single_module(
                module_name, config, device
            )
            results[module_name] = {
                "best_epoch": best_epoch,
                "test_metrics": test_m,
            }
        except Exception as e:
            print(f"  ❌ Error in {module_name}: {e}")
            results[module_name] = {"error": str(e)}

    print("\n" + "=" * 70)
    print(f"BASELINE: FUSION ({config.fusion_style})")
    print("=" * 70)

    try:
        fusion_metrics = train_fusion(config, device)
        if fusion_metrics is not None:
            results["fusion"] = fusion_metrics
    except Exception as e:
        print(f"  ❌ Fusion error: {e}")
        results["fusion"] = {"error": str(e)}

    summary = {
        "config": {
            **config.__dict__,
            "fusion_modules": list(config.fusion_modules),
        },
        "results": results,
    }
    BASELINE_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Cross-reference: print a compact comparison table
    print("\n" + "=" * 70)
    print("SUMMARY (test accuracy)")
    print("=" * 70)
    rows = []
    for name, r in results.items():
        if "test_metrics" in r:
            rows.append((name, r["test_metrics"]["accuracy"],
                         r["test_metrics"]["loss"]))
        elif "accuracy" in r:
            rows.append((name, r["accuracy"], r["loss"]))
    rows.sort(key=lambda x: -x[1])
    for name, acc, loss in rows:
        print(f"  {name:>15}  acc={acc:.4f}  loss={loss:.4f}")

    print(f"\nSaved baseline summary to {BASELINE_SUMMARY_PATH}")


if __name__ == "__main__":
    main()