#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train all module models + fusion model in one script (PATCHED: per-module LR).

Expected inputs from script a:
- final_data/nn_input_lineup_TRAIN.csv / TEST.csv
- final_data/nn_input_player_stats_TRAIN.csv / TEST.csv
- final_data/nn_input_skill_matchups_TRAIN.csv / TEST.csv
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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

META_PATH = MISC_DIR / "nn_module_metadata.json"
INDEPENDENT_SUMMARY_PATH = RESULT_DATA_DIR / "independent_module_training_summary.json"
FUSION_SUMMARY_PATH = RESULT_DATA_DIR / "skellam_fusion_summary.json"
FUSION_MODEL_PATH = RESULT_DATA_DIR / "skellam_fusion_model.pt"
FUSION_TEST_PRED_PATH = RESULT_DATA_DIR / "skellam_fusion_test_predictions.csv"

MODULES = ["lineup", "player_stats", "skill_matchups", "match_stats"]


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3            # Base LR (will be scaled per-module)
    weight_decay: float = 1e-5
    hidden_dim: int = 64
    dropout: float = 0.15
    max_abs_goal_diff: int = 12
    seed: int = 42
    early_stopping_patience: int = 12      # Slightly more patient
    lr_patience: int = 5
    lr_factor: float = 0.5
    min_lr: float = 1e-5


# Per-module learning rate multipliers (addressing lineup collapse)
MODULE_LR_MULTIPLIERS = {
    "lineup": 0.1,          # Very gentle - lineup features are noisy
    "player_stats": 0.3,    # Moderate - few features, needs caution
    "skill_matchups": 0.5,  # Moderate - stable architecture
    "match_stats": 0.4,     # Moderate - dense PCA features
    "fusion": 1.0,          # Full LR - only 9 features
}


# ============================================================================
# DATASETS
# ============================================================================

class NumericModuleDataset(Dataset):
    def __init__(self, frame, feature_cols, key_col):
        self.keys = frame[key_col].astype(str).values
        self.x = torch.tensor(frame[feature_cols].astype(float).values, dtype=torch.float32)
        self.hg = torch.tensor(frame["HG"].astype(float).values, dtype=torch.float32)
        self.ag = torch.tensor(frame["AG"].astype(float).values, dtype=torch.float32)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        return {"key": self.keys[idx], "x": self.x[idx], "HG": self.hg[idx], "AG": self.ag[idx]}


class FusionDataset(Dataset):
    def __init__(self, frame, feature_cols):
        self.game_id = frame["game_id"].astype(str).values
        self.x = torch.tensor(frame[feature_cols].astype(float).values, dtype=torch.float32)
        self.hg = torch.tensor(frame["HG"].astype(float).values, dtype=torch.float32)
        self.ag = torch.tensor(frame["AG"].astype(float).values, dtype=torch.float32)

    def __len__(self):
        return len(self.game_id)

    def __getitem__(self, idx):
        return {"game_id": self.game_id[idx], "x": self.x[idx], "HG": self.hg[idx], "AG": self.ag[idx]}


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class BaseSkellamNet(nn.Module):
    def __init__(self, home_prior=1.35, away_prior=1.10):
        super().__init__()
        self.register_buffer("home_prior", torch.tensor(float(home_prior)))
        self.register_buffer("away_prior", torch.tensor(float(away_prior)))
        self.MIN_MULT = 0.2
        self.MAX_MULT = 3.0

    def _rates_from_logits(self, logits):
        mult_home = self.MIN_MULT + (self.MAX_MULT - self.MIN_MULT) * torch.sigmoid(logits[:, 0])
        mult_away = self.MIN_MULT + (self.MAX_MULT - self.MIN_MULT) * torch.sigmoid(logits[:, 1])
        return self.home_prior * mult_home, self.away_prior * mult_away


class LineupSkellamNet(BaseSkellamNet):
    def __init__(self, input_dim, hidden_dim, dropout, home_prior=1.35, away_prior=1.10):
        super().__init__(home_prior, away_prior)
        h1 = max(hidden_dim, 64)
        h2 = max(hidden_dim // 2, 32)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(h2, 2)

    def forward(self, x):
        return self._rates_from_logits(self.head(self.feature_extractor(x)))


class PlayerStatsSkellamNet(BaseSkellamNet):
    def __init__(self, input_dim, hidden_dim, dropout, home_prior=1.35, away_prior=1.10):
        super().__init__(home_prior, away_prior)
        h = max(hidden_dim // 2, 24)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, h), nn.Tanh(), nn.Dropout(dropout * 0.5),
            nn.Linear(h, h), nn.ReLU(),
        )
        self.head = nn.Linear(h, 2)

    def forward(self, x):
        return self._rates_from_logits(self.head(self.feature_extractor(x)))


class SkillMatchupsSkellamNet(BaseSkellamNet):
    def __init__(self, input_dim, hidden_dim, dropout, home_prior=1.35, away_prior=1.10):
        super().__init__(home_prior, away_prior)
        h1 = max(hidden_dim, 64)
        h2 = max(hidden_dim // 2, 32)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.ReLU(),
        )
        self.head = nn.Linear(h2, 2)

    def forward(self, x):
        return self._rates_from_logits(self.head(self.feature_extractor(x)))


class MatchStatsSkellamNet(BaseSkellamNet):
    def __init__(self, input_dim, hidden_dim, dropout, home_prior=1.35, away_prior=1.10):
        super().__init__(home_prior, away_prior)
        h1 = max(hidden_dim, 96)
        h2 = max(hidden_dim // 2, 48)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(dropout * 0.5),
        )
        self.head = nn.Linear(h2, 2)

    def forward(self, x):
        return self._rates_from_logits(self.head(self.feature_extractor(x)))


class FusionSkellamNet(BaseSkellamNet):
    def __init__(self, input_dim, hidden_dim, dropout, home_prior=1.35, away_prior=1.10):
        super().__init__(home_prior, away_prior)
        mid = max(16, hidden_dim // 2)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, mid), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(mid, 2)

    def forward(self, x):
        return self._rates_from_logits(self.head(self.feature_extractor(x)))


# ============================================================================
# LOSS & METRICS
# ============================================================================

def skellam_nll(lambda_home, lambda_away, goal_diff, max_abs_goal_diff=12):
    d = torch.clamp(goal_diff, -max_abs_goal_diff, max_abs_goal_diff)
    d_long = torch.abs(d.to(torch.int64))
    z = 2.0 * torch.sqrt(lambda_home * lambda_away)
    i0 = torch.special.modified_bessel_i0(z)
    i1 = torch.special.modified_bessel_i1(z)
    if max_abs_goal_diff == 0:
        bessel_stack = i0.unsqueeze(1)
    else:
        i_vals = [i0, i1]
        z_safe = torch.clamp(z, min=1e-6)
        for n in range(1, max_abs_goal_diff):
            next_val = i_vals[n - 1] - (2.0 * n / z_safe) * i_vals[n]
            next_val = torch.clamp(next_val, min=1e-12)
            i_vals.append(next_val)
        bessel_stack = torch.stack(i_vals, dim=1)
    i_abs_d = torch.gather(bessel_stack, 1, d_long.unsqueeze(1)).squeeze(1)
    i_abs_d = torch.clamp(i_abs_d, min=1e-12)
    log_prob = (-(lambda_home + lambda_away) + 0.5 * d * (torch.log(lambda_home) - torch.log(lambda_away)) + torch.log(i_abs_d))
    return -log_prob.mean()


def compute_metrics(model, loader, device):
    model.eval()
    nlls, mses, maes = [], [], []
    home_rates, away_rates = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            hg, ag = batch["HG"].to(device), batch["AG"].to(device)
            d = hg - ag
            lam_h, lam_a = model(x)
            nlls.append(float(skellam_nll(lam_h, lam_a, d, 12).item()))
            mses.append(float(F.mse_loss(lam_h - lam_a, d).item()))
            maes.append(float(F.l1_loss(lam_h - lam_a, d).item()))
            home_rates.append(float(lam_h.mean().item()))
            away_rates.append(float(lam_a.mean().item()))
    return {
        "nll": float(np.mean(nlls)), "mse": float(np.mean(mses)), "mae": float(np.mean(maes)),
        "mean_home_rate": float(np.mean(home_rates)), "mean_away_rate": float(np.mean(away_rates)),
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
        raise FileNotFoundError(f"Missing metadata file {META_PATH}. Run a_prepare_data.py first.")
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    if module_name not in meta.get("modules", {}):
        raise KeyError(f"Module '{module_name}' not found in metadata.")
    return meta["modules"][module_name]


def load_module_frames(module_name):
    module_meta = load_module_meta(module_name)
    key_col = module_meta["key_col"]
    feature_cols = module_meta["feature_cols"]
    train_path = FINAL_DATA_DIR / f"nn_input_{module_name}_TRAIN.csv"
    test_path = FINAL_DATA_DIR / f"nn_input_{module_name}_TEST.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing prepared inputs for {module_name}.")
    return pd.read_csv(train_path, low_memory=False), pd.read_csv(test_path, low_memory=False), key_col, feature_cols


def build_module_model(module_name, input_dim, config, home_prior, away_prior):
    if module_name == "lineup":
        return LineupSkellamNet(input_dim, config.hidden_dim, config.dropout, home_prior, away_prior), "LineupSkellamNet"
    if module_name == "player_stats":
        return PlayerStatsSkellamNet(input_dim, config.hidden_dim, config.dropout, home_prior, away_prior), "PlayerStatsSkellamNet"
    if module_name == "skill_matchups":
        return SkillMatchupsSkellamNet(input_dim, config.hidden_dim, config.dropout, home_prior, away_prior), "SkillMatchupsSkellamNet"
    if module_name == "match_stats":
        return MatchStatsSkellamNet(input_dim, config.hidden_dim, config.dropout, home_prior, away_prior), "MatchStatsSkellamNet"
    raise ValueError(f"Unsupported module: {module_name}")


def save_module_predictions(model, device, config, key_col, feature_cols, train_df, test_df, module_name):
    train_pred_path = RESULT_DATA_DIR / f"skellam_{module_name}_train_predictions.csv"
    test_pred_path = RESULT_DATA_DIR / f"skellam_{module_name}_test_predictions.csv"

    def _build(df):
        loader = DataLoader(NumericModuleDataset(df, feature_cols, key_col), batch_size=config.batch_size, shuffle=False)
        rows = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(device)
                lam_h, lam_a = model(x)
                for i, k in enumerate(batch["key"]):
                    rows.append({key_col: k, "HG": float(batch["HG"][i]), "AG": float(batch["AG"][i]),
                                 "lambda_home": float(lam_h[i]), "lambda_away": float(lam_a[i]),
                                 "expected_goal_diff": float(lam_h[i] - lam_a[i])})
        return pd.DataFrame(rows)

    _build(train_df).to_csv(train_pred_path, index=False)
    _build(test_df).to_csv(test_pred_path, index=False)
    return train_pred_path, test_pred_path


# ============================================================================
# TRAINING LOOP
# ============================================================================

def _train_loop(model, optimizer, scheduler, train_loader, test_loader, device, config, label):
    best_test_nll = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    best_state = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch in train_loader:
            x, hg, ag = batch["x"].to(device), batch["HG"].to(device), batch["AG"].to(device)
            d = hg - ag
            optimizer.zero_grad()
            lam_h, lam_a = model(x)
            loss = skellam_nll(lam_h, lam_a, d, config.max_abs_goal_diff)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        train_m = compute_metrics(model, train_loader, device)
        test_m = compute_metrics(model, test_loader, device)
        scheduler.step(test_m["nll"])

        if test_m["nll"] < best_test_nll - 1e-4:
            best_test_nll = test_m["nll"]
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        lr_now = optimizer.param_groups[0]['lr']
        if epoch % 5 == 0 or epoch == 1:
            print(f"[{label}] Ep {epoch:03d}/{config.epochs} | Train NLL {train_m['nll']:.4f} | Test NLL {test_m['nll']:.4f} | "
                  f"λh {test_m['mean_home_rate']:.2f} λa {test_m['mean_away_rate']:.2f} | LR {lr_now:.1e}")

        if epochs_no_improve >= config.early_stopping_patience:
            print(f"  ⏹ Early stop at epoch {epoch} (best ep {best_epoch}, NLL {best_test_nll:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  ✓ Restored best model from epoch {best_epoch}")
    return best_test_nll, best_epoch


# ============================================================================
# MODULE TRAINING
# ============================================================================

def train_module(module_name, config, device):
    train_df, test_df, key_col, feature_cols = load_module_frames(module_name)
    home_prior = float(train_df["HG"].mean())
    away_prior = float(train_df["AG"].mean())

    train_ds = NumericModuleDataset(train_df, feature_cols, key_col)
    test_ds = NumericModuleDataset(test_df, feature_cols, key_col)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    model, arch = build_module_model(module_name, len(feature_cols), config, home_prior, away_prior)
    model = model.to(device)

    # Per-module learning rate
    lr_mult = MODULE_LR_MULTIPLIERS.get(module_name, 1.0)
    effective_lr = config.learning_rate * lr_mult
    print(f"  Effective LR: {effective_lr:.1e} (base {config.learning_rate:.1e} × {lr_mult})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.lr_factor,
                                                           patience=config.lr_patience, min_lr=config.min_lr)

    print("=" * 70)
    print(f"TRAINING MODULE: {module_name} | {arch}")
    print("=" * 70)
    print(f"Rows: {len(train_ds)}/{len(test_ds)} | Features: {len(feature_cols)} | Priors: λh={home_prior:.2f} λa={away_prior:.2f}")

    best_nll, best_epoch = _train_loop(model, optimizer, scheduler, train_loader, test_loader, device, config, label=module_name)

    model_path = RESULT_DATA_DIR / f"skellam_{module_name}_model.pt"
    torch.save({"state_dict": model.state_dict(), "module_name": module_name, "architecture": arch,
                "feature_cols": feature_cols, "key_col": key_col, "config": config.__dict__,
                "home_prior": home_prior, "away_prior": away_prior, "effective_lr": effective_lr,
                "best_test_nll": best_nll, "best_epoch": best_epoch}, model_path)

    train_pred_path, test_pred_path = save_module_predictions(model, device, config, key_col, feature_cols, train_df, test_df, module_name)

    result = {"module": module_name, "architecture": arch, "feature_count": len(feature_cols),
              "rows_train": len(train_ds), "rows_test": len(test_ds), "best_epoch": best_epoch,
              "best_test_nll": best_nll, "home_prior": home_prior, "away_prior": away_prior,
              "effective_lr": effective_lr, "model_path": str(model_path),
              "train_predictions_path": str(train_pred_path), "test_predictions_path": str(test_pred_path)}
    print(f"Saved model + predictions for {module_name}\n")
    return result


# ============================================================================
# FUSION TRAINING
# ============================================================================

def load_pred_frame(module_name, split):
    path = RESULT_DATA_DIR / f"skellam_{module_name}_{split}_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path, low_memory=False)
    return df[["game_id", "HG", "AG", "lambda_home", "lambda_away", "expected_goal_diff"]].rename(
        columns={"lambda_home": f"{module_name}_lambda_home", "lambda_away": f"{module_name}_lambda_away",
                 "expected_goal_diff": f"{module_name}_expected_goal_diff"})


def build_fusion_frame(split):
    merged = None
    for mn in MODULES:
        df = load_pred_frame(mn, split)
        if merged is None:
            merged = df
            continue
        merged = merged.merge(df, on="game_id", how="inner", suffixes=("", f"_{mn}"))
        for label in ["HG", "AG"]:
            dup = f"{label}_{mn}"
            if dup in merged.columns:
                merged[label] = merged[label].combine_first(merged[dup])
                merged = merged.drop(columns=[dup])
    feature_cols = []
    for mn in MODULES:
        feature_cols += [f"{mn}_lambda_home", f"{mn}_lambda_away", f"{mn}_expected_goal_diff"]
    merged = merged.dropna(subset=["HG", "AG"]).sort_values("game_id").reset_index(drop=True)
    return merged, feature_cols


def train_fusion(config, device):
    train_df, feature_cols = build_fusion_frame("train")
    test_df, _ = build_fusion_frame("test")
    home_prior = float(train_df["HG"].mean())
    away_prior = float(train_df["AG"].mean())

    train_ds = FusionDataset(train_df, feature_cols)
    test_ds = FusionDataset(test_df, feature_cols)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    model = FusionSkellamNet(len(feature_cols), max(32, config.hidden_dim // 2), config.dropout, home_prior, away_prior).to(device)
    
    lr_mult = MODULE_LR_MULTIPLIERS.get("fusion", 1.0)
    effective_lr = config.learning_rate * lr_mult

    optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.lr_factor,
                                                           patience=config.lr_patience, min_lr=config.min_lr)

    print("=" * 70)
    print("TRAINING FUSION HEAD")
    print("=" * 70)
    print(f"Rows: {len(train_ds)}/{len(test_ds)} | Features: {len(feature_cols)} | Priors: λh={home_prior:.2f} λa={away_prior:.2f} | LR: {effective_lr:.1e}")

    best_nll, best_epoch = _train_loop(model, optimizer, scheduler, train_loader, test_loader, device, config, label="fusion")

    torch.save({"state_dict": model.state_dict(), "architecture": "FusionSkellamNet", "feature_cols": feature_cols,
                "config": config.__dict__, "home_prior": home_prior, "away_prior": away_prior,
                "effective_lr": effective_lr, "best_test_nll": best_nll, "best_epoch": best_epoch}, FUSION_MODEL_PATH)

    model.eval()
    rows = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            lam_h, lam_a = model(x)
            for i, gid in enumerate(batch["game_id"]):
                rows.append({"game_id": gid, "HG": float(batch["HG"][i]), "AG": float(batch["AG"][i]),
                             "lambda_home": float(lam_h[i]), "lambda_away": float(lam_a[i]),
                             "expected_goal_diff": float(lam_h[i] - lam_a[i])})
    pd.DataFrame(rows).to_csv(FUSION_TEST_PRED_PATH, index=False)

    summary = {"architecture": "FusionSkellamNet", "best_epoch": best_epoch, "best_test_nll": best_nll,
               "rows_train": len(train_ds), "rows_test": len(test_ds), "feature_cols": feature_cols,
               "home_prior": home_prior, "away_prior": away_prior, "effective_lr": effective_lr,
               "model_path": str(FUSION_MODEL_PATH), "test_predictions_path": str(FUSION_TEST_PRED_PATH)}
    FUSION_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved fusion model + predictions\n")


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--max-abs-goal-diff", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--early-stopping-patience", type=int, default=12)
    p.add_argument("--lr-patience", type=int, default=5)
    args = p.parse_args()
    return TrainConfig(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
                       weight_decay=args.weight_decay, hidden_dim=args.hidden_dim, dropout=args.dropout,
                       max_abs_goal_diff=args.max_abs_goal_diff, seed=args.seed,
                       early_stopping_patience=args.early_stopping_patience, lr_patience=args.lr_patience)


def main():
    config = parse_args()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Learning rate multipliers: {MODULE_LR_MULTIPLIERS}\n")
    RESULT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    results = [train_module(mn, config, device) for mn in MODULES]
    INDEPENDENT_SUMMARY_PATH.write_text(json.dumps({"config": config.__dict__, "results": results}, indent=2), encoding="utf-8")
    print(f"Saved module summary → {INDEPENDENT_SUMMARY_PATH}")

    train_fusion(config, device)


if __name__ == "__main__":
    main()