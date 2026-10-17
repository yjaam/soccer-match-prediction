#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""BASELINE: Minimal MLP model for match prediction WITH FUSION.

Modules: features → MLP → rates (PCsInit first layer for modules without upstream PCA)
Fusion:  module predictions → final rates, selected by BaselineConfig.fusion_style:
    "attention"  self-attention over the module tokens; per match it scores how much
                 each module's rate counts, and the final rate is the softmax-weighted
                 geometric blend of the module rates (cannot be bolder than its inputs)
    "gbdt"       gradient-boosted trees (Poisson loss) stacked on the module rates
    "mlp"        the original small MLP over the concatenated module rates

Both the per-module model and the fusion model are small MLPs. The head
of each MLP is initialized with a small non-zero std so the model starts
near the goal prior while gradients still flow to earlier layers.

Refinements (v2)
----------------
- Cap on auto-sized hidden layers, to prevent wide modules from overfitting.
- Stronger weight decay and dropout, especially in the fusion MLP.
- Earlier stopping.
- Fusion can be restricted to a subset of modules. `league` is excluded by
  default because its rate predictions carry almost no variance and only
  add noise to the fusion input.
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
from sklearn.ensemble import HistGradientBoostingRegressor

from pcs_init import module_pca_applied, pca_projection


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
FINAL_DATA_DIR = PROJECT_DIR / "final_data"
RESULT_DATA_DIR = PROJECT_DIR / "result_data"
MISC_DIR = PROJECT_DIR / "misc"

META_PATH = MISC_DIR / "nn_module_metadata.json"
BASELINE_SUMMARY_PATH = RESULT_DATA_DIR / "baseline_mlp_summary.json"
BASELINE_MODEL_PATH = RESULT_DATA_DIR / "baseline_mlp_model.pt"
BASELINE_PRED_PATH = RESULT_DATA_DIR / "baseline_mlp_fusion_predictions.csv"

MODULES = ["lineup", "player_stats", "skill_matchups", "match_stats", "matchup"]


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

    # MLP architecture
    module_hidden1: int = 0        # 0 = auto
    module_hidden2: int = 0        # 0 = auto
    module_hidden_cap: int = 64    # cap on the auto-sized first hidden layer
    module_dropout: float = 0.3

    # PCsInit: for modules whose upstream script did NOT apply PCA, initialise a
    # linear first layer with the principal components of the training inputs.
    use_pcsinit: bool = True
    pcsinit_variance: float = 0.95         # variance retained by the PC layer
    pcsinit_frozen_epochs: int = 10        # PC layer frozen for these epochs, then fine-tuned
    pcsinit_activation: bool = False       # True = PCsInit-Act (ReLU after the PC layer)
    pcsinit_subset_fraction: float = 1.0   # < 1.0 = PCsInit-Sub (PCA on a row subset)
    fusion_hidden: int = 16
    fusion_dropout: float = 0.3

    # Fusion head: "attention" | "gbdt" | "mlp" | "auto" (best validation loss of the three)
    fusion_style: str = "attention"

    # The fusion is fit on K-fold out-of-fold module predictions (contiguous date blocks)
    # so it sees the modules' real generalisation quality. <= 1 = in-sample predictions.
    fusion_oof_folds: int = 5
    fusion_seeds: int = 3              # neural heads: average this many seeds (log-rate space)
    attn_shrinkage: float = 1.0        # pull attention weights toward the uniform blend

    # Attention fusion
    attn_d_model: int = 16
    attn_heads: int = 2
    attn_layers: int = 1
    attn_ff: int = 32

    # GBDT fusion (sklearn HistGradientBoostingRegressor, Poisson loss)
    gbdt_max_iter: int = 300
    gbdt_learning_rate: float = 0.05
    gbdt_max_depth: int = 3
    gbdt_min_samples_leaf: int = 100
    gbdt_l2: float = 1.0

    # Which modules participate in the fusion step
    fusion_modules: tuple = (
        "lineup", "player_stats", "skill_matchups", "match_stats", "matchup"
    )


# ============================================================================
# DATASETS
# ============================================================================

class SimpleDataset(Dataset):
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


class FusionDataset(Dataset):
    """Dataset for fusion: contains predictions from all modules."""
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
# MODELS
# ============================================================================

class MLPRateModel(nn.Module):
    """MLP rate model: log(lambda) = log(prior) + head(trunk(x)).

    The head is initialized with a small non-zero std so the model starts
    close to the goal prior while gradients still flow to earlier layers.
    """
    def __init__(self, input_dim, home_prior=1.35, away_prior=1.10,
                 hidden1=64, hidden2=32, dropout=0.2,
                 pca_weights=None, pca_activation=False):
        super().__init__()
        self.register_buffer(
            "log_home_prior", torch.tensor(float(np.log(home_prior)))
        )
        self.register_buffer(
            "log_away_prior", torch.tensor(float(np.log(away_prior)))
        )
        # PCsInit: linear layer with zero bias, initialised with principal directions.
        self.pca_layer = None
        trunk_in = input_dim
        if pca_weights is not None:
            self.pca_layer = nn.Linear(input_dim, pca_weights.shape[0])
            with torch.no_grad():
                self.pca_layer.weight.copy_(torch.as_tensor(pca_weights, dtype=torch.float32))
                self.pca_layer.bias.zero_()
            self.pca_act = nn.ReLU() if pca_activation else nn.Identity()
            trunk_in = pca_weights.shape[0]
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden2, 2)
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.head.bias)

    def set_pca_frozen(self, frozen):
        if self.pca_layer is not None:
            for p in self.pca_layer.parameters():
                p.requires_grad_(not frozen)

    def forward(self, x):
        if self.pca_layer is not None:
            x = self.pca_act(self.pca_layer(x))
        h = self.trunk(x)
        logits = self.head(h)
        lam_h = torch.exp(self.log_home_prior + logits[:, 0])
        lam_a = torch.exp(self.log_away_prior + logits[:, 1])
        return lam_h, lam_a


class MLPFusionModel(nn.Module):
    """MLP fusion: concat(module rates + diffs) -> MLP -> final rates."""
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
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden, 2)
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        h = self.trunk(x.view(x.shape[0], -1))
        logits = self.head(h)
        lam_h = torch.exp(self.log_home_prior + logits[:, 0])
        lam_a = torch.exp(self.log_away_prior + logits[:, 1])
        return lam_h, lam_a


class AttentionFusionModel(nn.Module):
    """Self-attention over the module tokens.

    Each module is one token (log-rate home/away relative to the prior, rate diff) plus a
    learned module embedding. After the encoder, every token gets a score; per match the
    softmax of these scores weights the module log-rates. Scores start near zero, so the
    model starts from the uniform blend and can only move weight between modules.
    """
    def __init__(self, n_modules, home_prior=1.35, away_prior=1.10,
                 d_model=16, n_heads=2, n_layers=1, ff=32, dropout=0.3):
        super().__init__()
        self.n_modules = n_modules
        self.register_buffer(
            "log_home_prior", torch.tensor(float(np.log(home_prior)))
        )
        self.register_buffer(
            "log_away_prior", torch.tensor(float(np.log(away_prior)))
        )
        self.embed = nn.Linear(3, d_model)
        self.module_emb = nn.Parameter(torch.randn(n_modules, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.score = nn.Linear(d_model, 2)
        nn.init.normal_(self.score.weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.score.bias)

    def forward(self, x):
        seq = x.view(x.shape[0], self.n_modules, 3)
        log_h = torch.log(seq[:, :, 0].clamp(min=1e-6)) - self.log_home_prior
        log_a = torch.log(seq[:, :, 1].clamp(min=1e-6)) - self.log_away_prior
        tokens = torch.stack([log_h, log_a, seq[:, :, 2]], dim=-1)
        h = self.encoder(self.embed(tokens) + self.module_emb)
        w = torch.softmax(self.score(h), dim=1)          # (B, M, 2) weights over modules
        self.last_weights = w
        out_h = (log_h * w[:, :, 0]).sum(dim=1)
        out_a = (log_a * w[:, :, 1]).sum(dim=1)
        return (torch.exp(self.log_home_prior + out_h),
                torch.exp(self.log_away_prior + out_a))

    def shrinkage(self):
        """Mean squared distance of the per-match weights from the uniform blend."""
        return ((self.last_weights - 1.0 / self.n_modules) ** 2).sum(dim=(1, 2)).mean()


class EnsembleFusionModel(nn.Module):
    """Average of several fusion models in log-rate space."""
    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x):
        outs = [m(x) for m in self.members]
        lam_h = torch.exp(torch.stack([torch.log(o[0]) for o in outs]).mean(dim=0))
        lam_a = torch.exp(torch.stack([torch.log(o[1]) for o in outs]).mean(dim=0))
        return lam_h, lam_a


class GBDTFusionModel(nn.Module):
    """Wraps two fitted tree ensembles so the shared metric code works unchanged."""
    def __init__(self, model_home, model_away):
        super().__init__()
        self.models = (model_home, model_away)

    def forward(self, x):
        arr = x.detach().cpu().numpy()
        lam_h = np.clip(self.models[0].predict(arr), 0.05, 8.0)
        lam_a = np.clip(self.models[1].predict(arr), 0.05, 8.0)
        return (torch.tensor(lam_h, dtype=torch.float32, device=x.device),
                torch.tensor(lam_a, dtype=torch.float32, device=x.device))


def fit_gbdt_fusion(fusion_train, feature_cols, config):
    def make():
        return HistGradientBoostingRegressor(
            loss="poisson", max_iter=config.gbdt_max_iter,
            learning_rate=config.gbdt_learning_rate, max_depth=config.gbdt_max_depth,
            min_samples_leaf=config.gbdt_min_samples_leaf,
            l2_regularization=config.gbdt_l2, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=20, random_state=config.seed,
        )
    x = fusion_train[feature_cols].to_numpy(dtype=float)
    return GBDTFusionModel(make().fit(x, fusion_train["HG"].to_numpy(dtype=float)),
                           make().fit(x, fusion_train["AG"].to_numpy(dtype=float)))


def build_fusion_model(config, style, n_modules, home_prior, away_prior):
    if style == "attention":
        return AttentionFusionModel(
            n_modules, home_prior, away_prior,
            d_model=config.attn_d_model, n_heads=config.attn_heads,
            n_layers=config.attn_layers, ff=config.attn_ff,
            dropout=config.fusion_dropout,
        )
    if style == "mlp":
        return MLPFusionModel(
            n_modules, home_prior, away_prior,
            hidden=config.fusion_hidden, dropout=config.fusion_dropout,
        )
    raise ValueError(f"Unknown fusion_style: {style!r}")


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
    """Return P(home win), P(draw), P(away win) under independent Poisson."""
    device = lam_h.device
    goals = torch.arange(0, max_goals + 1, device=device).float()
    log_fact = torch.lgamma(goals + 1)

    log_p_h = -lam_h.unsqueeze(1) + goals.unsqueeze(0) * torch.log(lam_h.unsqueeze(1) + 1e-8) - log_fact.unsqueeze(0)
    log_p_a = -lam_a.unsqueeze(1) + goals.unsqueeze(0) * torch.log(lam_a.unsqueeze(1) + 1e-8) - log_fact.unsqueeze(0)

    p_h = torch.exp(log_p_h)  # (B, G)
    p_a = torch.exp(log_p_a)  # (B, G)

    mat = p_h.unsqueeze(2) * p_a.unsqueeze(1)  # (B, G, G)

    G = max_goals + 1
    idx = torch.arange(G, device=device)
    home_mask = idx.unsqueeze(1) > idx.unsqueeze(0)
    draw_mask = idx.unsqueeze(1) == idx.unsqueeze(0)
    away_mask = idx.unsqueeze(1) < idx.unsqueeze(0)

    p_home = (mat * home_mask.unsqueeze(0)).sum(dim=(1, 2))
    p_draw = (mat * draw_mask.unsqueeze(0)).sum(dim=(1, 2))
    p_away = (mat * away_mask.unsqueeze(0)).sum(dim=(1, 2))

    probs = torch.stack([p_home, p_draw, p_away], dim=1)
    probs = probs / probs.sum(dim=1, keepdim=True)
    return probs


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

    train_path = FINAL_DATA_DIR / f"nn_input_{module_name}_TRAIN.csv"
    test_path = FINAL_DATA_DIR / f"nn_input_{module_name}_TEST.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing prepared inputs for {module_name}.")

    train_df = pd.read_csv(train_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)

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

def fit_module_mlp(module_name, fit_df, val_df, feature_cols, config, device, verbose=True):
    """Standardise on fit_df, train one MLP with early stopping on val_df.

    Returns (model, feature_mean, feature_std, best_epoch). Inputs are not modified.
    """
    fit_df = fit_df.copy()
    val_df = val_df.copy()

    mean = fit_df[feature_cols].mean()
    std = fit_df[feature_cols].std().replace(0, 1.0)
    for df in (fit_df, val_df):
        df[feature_cols] = (df[feature_cols] - mean) / std

    home_prior = float(fit_df["HG"].mean())
    away_prior = float(fit_df["AG"].mean())

    train_ds = SimpleDataset(fit_df, feature_cols)
    val_ds = SimpleDataset(val_df, feature_cols)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                              collate_fn=collate_simple_batch)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                            collate_fn=collate_simple_batch)

    # Auto-size hidden layers, capped
    h1 = config.module_hidden1 or min(
        config.module_hidden_cap, max(16, 2 * len(feature_cols))
    )
    h2 = config.module_hidden2 or max(8, h1 // 2)

    pca_weights = None
    if config.use_pcsinit and module_pca_applied(META_PATH, module_name) is False:
        pca_weights = pca_projection(
            fit_df[feature_cols].to_numpy(), config.pcsinit_variance,
            config.pcsinit_subset_fraction, config.seed,
        )

    model = MLPRateModel(
        len(feature_cols), home_prior, away_prior,
        hidden1=h1, hidden2=h2, dropout=config.module_dropout,
        pca_weights=pca_weights, pca_activation=config.pcsinit_activation,
    ).to(device)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        pcs_note = "" if pca_weights is None else f", PCsInit: {pca_weights.shape[0]} PCs"
        print(f"\n  [{module_name}] Training (Features: {len(feature_cols)}, "
              f"Rows: {len(train_ds)}, MLP: {len(feature_cols)}->{h1}->{h2}->2, "
              f"Params: {n_params:,}{pcs_note})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    pcs_frozen = pca_weights is not None and config.pcsinit_frozen_epochs > 0
    model.set_pca_frozen(pcs_frozen)

    for epoch in range(1, config.epochs + 1):
        if pcs_frozen:
            if epoch > config.pcsinit_frozen_epochs:
                model.set_pca_frozen(False)
                pcs_frozen = False
            else:
                patience_counter = 0  # no early stopping while the PC layer is frozen
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

    return model, mean, std, best_epoch


def _scaled(df, feature_cols, mean, std):
    out = df.copy()
    out[feature_cols] = (out[feature_cols] - mean) / std
    return out


def oof_module_predictions(module_name, train_df_full, feature_cols, config, device):
    """Out-of-fold predictions for every training row.

    Folds are contiguous date blocks (game_id starts with YYYYMMDD), the same for every
    module. Random folds leak across neighbouring matches and overstate modules that
    memorise teams or form.
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

        model, mean, std, _ = fit_module_mlp(
            module_name, fit_part, val_part, feature_cols, config, device, verbose=False
        )
        parts.append(predict_frame(model, _scaled(held, feature_cols, mean, std),
                                   feature_cols, config, device))
    return pd.concat(parts, ignore_index=True)


def train_single_module(module_name, config, device):
    train_df_full, test_df, key_col, feature_cols = load_module_frames(module_name)

    # Split train into train/validation
    rng = np.random.default_rng(config.seed)
    indices = rng.permutation(len(train_df_full))
    n_val = int(len(train_df_full) * config.val_ratio)
    val_df = train_df_full.iloc[indices[:n_val]].reset_index(drop=True)
    train_df = train_df_full.iloc[indices[n_val:]].reset_index(drop=True)

    model, mean, std, best_epoch = fit_module_mlp(
        module_name, train_df, val_df, feature_cols, config, device
    )

    test_scaled = _scaled(test_df, feature_cols, mean, std)
    train_full_scaled = _scaled(train_df_full, feature_cols, mean, std)

    test_ds = SimpleDataset(test_scaled, feature_cols)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False,
                             collate_fn=collate_simple_batch)

    test_m = compute_metrics(model, test_loader, device, config.max_goals)
    print(f"  [{module_name}] Test | Loss {test_m['loss']:.4f} | "
          f"Acc {test_m['accuracy']:.3f} | std(λh) {test_m['std_home_rate']:.4f}")

    # Save model
    model_path = RESULT_DATA_DIR / f"baseline_mlp_{module_name}_model.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "module_name": module_name,
        "best_epoch": best_epoch,
        "feature_mean": mean.to_dict(),
        "feature_std": std.to_dict(),
    }, model_path)

    # Save train and test predictions for fusion
    train_pred = predict_frame(model, train_full_scaled, feature_cols, config, device)
    test_pred = predict_frame(model, test_scaled, feature_cols, config, device)

    train_pred.to_csv(
        RESULT_DATA_DIR / f"baseline_mlp_{module_name}_train_predictions.csv",
        index=False,
    )
    test_pred.to_csv(
        RESULT_DATA_DIR / f"baseline_mlp_{module_name}_predictions.csv",
        index=False,
    )

    oof_path = RESULT_DATA_DIR / f"baseline_mlp_{module_name}_oof_predictions.csv"
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
    """Load module predictions and merge them into a single frame.

    Only the modules listed in `fusion_modules` are included.
    """
    dfs = []
    for mn in fusion_modules:
        if split == "train":
            path = RESULT_DATA_DIR / f"baseline_mlp_{mn}_oof_predictions.csv"
            if not path.exists():
                path = RESULT_DATA_DIR / f"baseline_mlp_{mn}_train_predictions.csv"
        else:
            path = RESULT_DATA_DIR / f"baseline_mlp_{mn}_predictions.csv"

        if not path.exists():
            print(f"    Missing {path}")
            continue

        df = pd.read_csv(path, low_memory=False)
        df = df.rename(columns={
            "lambda_home": f"{mn}_lambda_home",
            "lambda_away": f"{mn}_lambda_away",
        })
        df[f"{mn}_diff"] = df[f"{mn}_lambda_home"] - df[f"{mn}_lambda_away"]
        dfs.append(df[["game_id", "HG", "AG",
                       f"{mn}_lambda_home", f"{mn}_lambda_away", f"{mn}_diff"]])

    if len(dfs) < len(fusion_modules):
        return None, []

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df.drop(columns=["HG", "AG"]), on="game_id", how="inner")

    feature_cols = []
    for mn in fusion_modules:
        feature_cols += [f"{mn}_lambda_home", f"{mn}_lambda_away", f"{mn}_diff"]

    return merged, feature_cols


def fit_torch_fusion(style, config, device, n_modules, home_prior, away_prior,
                     train_loader, val_loader, seed, verbose):
    set_seed(seed)
    model = build_fusion_model(config, style, n_modules, home_prior, away_prior).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)

    best_val_loss = float("inf")
    best_state = None
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
                loss = loss + config.attn_shrinkage * model.shrinkage()
            loss.backward()
            optimizer.step()

        val_m = compute_metrics(model, val_loader, device, config.max_goals)

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
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
    return model


def fit_fusion_head(style, config, device, fusion_train, feature_cols, n_modules,
                    home_prior, away_prior, train_loader, val_loader):
    """Fit one fusion head; neural heads are averaged over config.fusion_seeds seeds."""
    if style == "gbdt":
        model = fit_gbdt_fusion(fusion_train, feature_cols, config)
    else:
        members = [
            fit_torch_fusion(style, config, device, n_modules, home_prior, away_prior,
                             train_loader, val_loader, config.seed + i, verbose=(i == 0))
            for i in range(max(1, config.fusion_seeds))
        ]
        model = members[0] if len(members) == 1 else EnsembleFusionModel(members).to(device)
    val_m = compute_metrics(model, val_loader, device, config.max_goals)
    print(f"  [fusion] {style}: Val Loss {val_m['loss']:.4f} | Acc {val_m['accuracy']:.3f}")
    return model, val_m["loss"]


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

    train_ds = FusionDataset(fusion_train, feature_cols)
    val_ds = FusionDataset(fusion_val, feature_cols)
    test_ds = FusionDataset(test_df, feature_cols)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                              collate_fn=collate_simple_batch)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                            collate_fn=collate_simple_batch)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False,
                             collate_fn=collate_simple_batch)

    print(f"\n  [fusion] Style: {config.fusion_style} (Rows: {len(train_ds)}, "
          f"Modules: {len(fusion_modules)}, Input dim: {len(feature_cols)})")
    print(f"    Using modules: {fusion_modules}")

    styles = ["attention", "mlp", "gbdt"] if config.fusion_style == "auto" else [config.fusion_style]
    fitted = [
        fit_fusion_head(style, config, device, fusion_train, feature_cols, len(fusion_modules),
                        home_prior, away_prior, train_loader, val_loader)
        for style in styles
    ]
    best_idx = int(np.argmin([loss for _, loss in fitted]))
    model = fitted[best_idx][0]
    chosen_style = styles[best_idx]
    if len(styles) > 1:
        print(f"  [fusion] auto selected: {chosen_style}")

    test_m = compute_metrics(model, test_loader, device, config.max_goals)
    print(f"  [fusion] Test | Loss {test_m['loss']:.4f} | "
          f"Acc {test_m['accuracy']:.3f} | std(λh) {test_m['std_home_rate']:.4f} | "
          f"λh {test_m['mean_home_rate']:.2f} λa {test_m['mean_away_rate']:.2f}")

    # Save fusion predictions
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
    torch.save({"state_dict": model.state_dict(), "fusion_style": chosen_style,
                "gbdt_models": getattr(model, "models", None)}, BASELINE_MODEL_PATH)

    return test_m


# ============================================================================
# MAIN
# ============================================================================

def main():
    config = BaselineConfig()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: epochs={config.epochs} lr={config.learning_rate} "
          f"wd={config.weight_decay} patience={config.patience} "
          f"mod_dropout={config.module_dropout} "
          f"fus_hidden={config.fusion_hidden} fus_dropout={config.fusion_dropout} "
          f"hidden_cap={config.module_hidden_cap}")
    print(f"Fusion modules: {list(config.fusion_modules)}")
    RESULT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BASELINE: INDEPENDENT MLP MODULES")
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
    print(f"\nSaved baseline summary to {BASELINE_SUMMARY_PATH}")


if __name__ == "__main__":
    main()