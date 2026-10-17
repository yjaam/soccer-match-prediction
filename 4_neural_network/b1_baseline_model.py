#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""BASELINE: Minimal linear model for match prediction WITH FUSION.

Modules: features → linear → rates
Fusion: concat(module predictions) → linear → final rates
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from pcs_init import module_pca_applied, pca_projection


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
FINAL_DATA_DIR = PROJECT_DIR / "final_data"
RESULT_DATA_DIR = PROJECT_DIR / "result_data"
MISC_DIR = PROJECT_DIR / "misc"

META_PATH = MISC_DIR / "nn_module_metadata.json"
BASELINE_SUMMARY_PATH = RESULT_DATA_DIR / "baseline_linear_summary.json"
BASELINE_MODEL_PATH = RESULT_DATA_DIR / "baseline_linear_model.pt"
BASELINE_PRED_PATH = RESULT_DATA_DIR / "baseline_linear_fusion_predictions.csv"

MODULES = ["lineup", "player_stats", "skill_matchups", "match_stats", "matchup", "season"]


@dataclass
class BaselineConfig:
    epochs: int = 100
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    val_ratio: float = 0.1
    patience: int = 20
    max_goals: int = 12

    # Modules whose upstream script did NOT apply PCA are reduced to principal
    # components (fit on the training split) before the linear model.
    use_pca: bool = True
    pca_variance: float = 0.95


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
# SIMPLE MODELS
# ============================================================================

class LinearRateModel(nn.Module):
    """Log-linear Poisson model: log(lambda) = log(prior) + linear(x)."""
    def __init__(self, input_dim, home_prior=1.35, away_prior=1.10):
        super().__init__()
        self.home_bias = nn.Parameter(torch.tensor(float(np.log(home_prior)), dtype=torch.float32))
        self.away_bias = nn.Parameter(torch.tensor(float(np.log(away_prior)), dtype=torch.float32))
        self.linear = nn.Linear(input_dim, 2)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        logits = self.linear(x)
        lam_h = torch.exp(self.home_bias + logits[:, 0])
        lam_a = torch.exp(self.away_bias + logits[:, 1])
        return lam_h, lam_a


class LinearFusionModel(nn.Module):
    """Log-linear fusion: concat(module predictions) → linear → rates."""
    def __init__(self, n_modules, home_prior=1.35, away_prior=1.10):
        super().__init__()
        self.home_bias = nn.Parameter(torch.tensor(float(np.log(home_prior)), dtype=torch.float32))
        self.away_bias = nn.Parameter(torch.tensor(float(np.log(away_prior)), dtype=torch.float32))
        self.linear = nn.Linear(n_modules * 3, 2)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x_flat = x.view(x.shape[0], -1)
        logits = self.linear(x_flat)
        lam_h = torch.exp(self.home_bias + logits[:, 0])
        lam_a = torch.exp(self.away_bias + logits[:, 1])
        return lam_h, lam_a


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
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_simple_batch)
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
# TRAINING
# ============================================================================

def train_single_module(module_name, config, device):
    train_df_full, test_df, key_col, feature_cols = load_module_frames(module_name)

    # Split train into train/validation
    rng = np.random.default_rng(config.seed)
    indices = rng.permutation(len(train_df_full))
    n_val = int(len(train_df_full) * config.val_ratio)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_df = train_df_full.iloc[train_idx].reset_index(drop=True)
    val_df = train_df_full.iloc[val_idx].reset_index(drop=True)

    # Standardize using only the training split
    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std().replace(0, 1.0)

    for df in (train_df, val_df, test_df):
        df[feature_cols] = (df[feature_cols] - mean) / std

    train_full_scaled = train_df_full.copy()
    train_full_scaled[feature_cols] = (train_full_scaled[feature_cols] - mean) / std

    components = None
    if config.use_pca and module_pca_applied(META_PATH, module_name) is False:
        components = pca_projection(train_df[feature_cols].to_numpy(), config.pca_variance)
    if components is not None:
        pc_cols = [f"pc_{i + 1}" for i in range(components.shape[0])]

        def _to_pcs(frame):
            out = frame[["game_id", "HG", "AG"]].copy()
            out[pc_cols] = frame[feature_cols].to_numpy(dtype=float) @ components.T
            return out

        train_df, val_df, test_df, train_full_scaled = (
            _to_pcs(f) for f in (train_df, val_df, test_df, train_full_scaled)
        )
        print(f"  [{module_name}] PCA: {len(feature_cols)} features -> {len(pc_cols)} components")
        feature_cols = pc_cols

    home_prior = float(train_df["HG"].mean())
    away_prior = float(train_df["AG"].mean())

    train_ds = SimpleDataset(train_df, feature_cols)
    val_ds = SimpleDataset(val_df, feature_cols)
    test_ds = SimpleDataset(test_df, feature_cols)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_simple_batch)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_simple_batch)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_simple_batch)

    model = LinearRateModel(len(feature_cols), home_prior, away_prior).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    print(f"\n  [{module_name}] Training (Features: {len(feature_cols)}, Rows: {len(train_ds)})")

    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch in train_loader:
            x, hg, ag = batch["x"].to(device), batch["HG"].to(device), batch["AG"].to(device)
            optimizer.zero_grad()
            lam_h, lam_a = model(x)
            loss = poisson_nll_loss(lam_h, lam_a, hg, ag)
            loss.backward()
            optimizer.step()

        val_m = compute_metrics(model, val_loader, device, config.max_goals)

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            print(f"    Early stopping at epoch {epoch}")
            break

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Ep {epoch:03d} | Val Loss {val_m['loss']:.4f} | Acc {val_m['accuracy']:.3f} | "
                  f"std(λh) {val_m['std_home_rate']:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_m = compute_metrics(model, test_loader, device, config.max_goals)
    print(f"  [{module_name}] Test | Loss {test_m['loss']:.4f} | Acc {test_m['accuracy']:.3f} | "
          f"std(λh) {test_m['std_home_rate']:.4f}")

    # Save model
    model_path = RESULT_DATA_DIR / f"baseline_linear_{module_name}_model.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "module_name": module_name,
        "best_epoch": best_epoch,
        "feature_mean": mean.to_dict(),
        "feature_std": std.to_dict(),
    }, model_path)

    # Save train and test predictions for fusion
    train_pred = predict_frame(model, train_full_scaled, feature_cols, config, device)
    test_pred = predict_frame(model, test_df, feature_cols, config, device)

    train_pred.to_csv(RESULT_DATA_DIR / f"baseline_linear_{module_name}_train_predictions.csv", index=False)
    test_pred.to_csv(RESULT_DATA_DIR / f"baseline_linear_{module_name}_predictions.csv", index=False)

    return model, test_ds, best_epoch, test_m


# ============================================================================
# FUSION
# ============================================================================

def build_fusion_frame(split):
    """Load module predictions and merge them into a single frame."""
    dfs = []
    for mn in MODULES:
        if split == "train":
            path = RESULT_DATA_DIR / f"baseline_linear_{mn}_train_predictions.csv"
        else:
            path = RESULT_DATA_DIR / f"baseline_linear_{mn}_predictions.csv"

        if not path.exists():
            print(f"    Missing {path}")
            continue

        df = pd.read_csv(path, low_memory=False)
        df = df.rename(columns={
            "lambda_home": f"{mn}_lambda_home",
            "lambda_away": f"{mn}_lambda_away",
        })
        df[f"{mn}_diff"] = df[f"{mn}_lambda_home"] - df[f"{mn}_lambda_away"]
        dfs.append(df[["game_id", "HG", "AG", f"{mn}_lambda_home", f"{mn}_lambda_away", f"{mn}_diff"]])

    if len(dfs) < len(MODULES):
        return None, []

    merged = dfs[0]
    for df in dfs[1:]:
        # Merge on game_id only to avoid float precision issues with HG/AG
        merged = merged.merge(df.drop(columns=["HG", "AG"]), on="game_id", how="inner")

    feature_cols = []
    for mn in MODULES:
        feature_cols += [f"{mn}_lambda_home", f"{mn}_lambda_away", f"{mn}_diff"]

    return merged, feature_cols


def train_fusion(config, device):
    train_df, feature_cols = build_fusion_frame("train")
    test_df, _ = build_fusion_frame("test")

    if train_df is None or test_df is None:
        print("  Fusion skipped: missing module predictions.")
        return None

    # Split fusion train into train/validation
    rng = np.random.default_rng(config.seed)
    indices = rng.permutation(len(train_df))
    n_val = int(len(train_df) * config.val_ratio)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    fusion_train = train_df.iloc[train_idx].reset_index(drop=True)
    fusion_val = train_df.iloc[val_idx].reset_index(drop=True)

    home_prior = float(fusion_train["HG"].mean())
    away_prior = float(fusion_train["AG"].mean())

    train_ds = FusionDataset(fusion_train, feature_cols)
    val_ds = FusionDataset(fusion_val, feature_cols)
    test_ds = FusionDataset(test_df, feature_cols)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_simple_batch)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_simple_batch)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_simple_batch)

    model = LinearFusionModel(len(MODULES), home_prior, away_prior).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    print(f"\n  [fusion] Training (Rows: {len(train_ds)})")

    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch in train_loader:
            x, hg, ag = batch["x"].to(device), batch["HG"].to(device), batch["AG"].to(device)
            optimizer.zero_grad()
            lam_h, lam_a = model(x)
            loss = poisson_nll_loss(lam_h, lam_a, hg, ag)
            loss.backward()
            optimizer.step()

        val_m = compute_metrics(model, val_loader, device, config.max_goals)

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            print(f"    Early stopping at epoch {epoch}")
            break

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Ep {epoch:03d} | Val Loss {val_m['loss']:.4f} | Acc {val_m['accuracy']:.3f} | "
                  f"std(λh) {val_m['std_home_rate']:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_m = compute_metrics(model, test_loader, device, config.max_goals)
    print(f"  [fusion] Test | Loss {test_m['loss']:.4f} | Acc {test_m['accuracy']:.3f} | "
          f"std(λh) {test_m['std_home_rate']:.4f} | "
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
    torch.save({"state_dict": model.state_dict()}, BASELINE_MODEL_PATH)

    return test_m


# ============================================================================
# MAIN
# ============================================================================

def main():
    config = BaselineConfig()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    RESULT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BASELINE: INDEPENDENT LINEAR MODULES")
    print("=" * 70)

    results = {}

    for module_name in MODULES:
        try:
            model, test_ds, best_epoch, test_m = train_single_module(module_name, config, device)
            results[module_name] = {
                "best_epoch": best_epoch,
                "test_metrics": test_m,
            }
        except Exception as e:
            print(f"  ❌ Error in {module_name}: {e}")
            results[module_name] = {"error": str(e)}

    print("\n" + "=" * 70)
    print("BASELINE: LINEAR FUSION")
    print("=" * 70)

    try:
        fusion_metrics = train_fusion(config, device)
        if fusion_metrics is not None:
            results["fusion"] = fusion_metrics
    except Exception as e:
        print(f"  ❌ Fusion error: {e}")
        results["fusion"] = {"error": str(e)}

    summary = {
        "config": config.__dict__,
        "results": results,
    }
    BASELINE_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved baseline summary to {BASELINE_SUMMARY_PATH}")


if __name__ == "__main__":
    main()