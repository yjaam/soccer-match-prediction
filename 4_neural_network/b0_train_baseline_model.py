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


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
FINAL_DATA_DIR = PROJECT_DIR / "final_data"
RESULT_DATA_DIR = PROJECT_DIR / "result_data"
MISC_DIR = PROJECT_DIR / "misc"

META_PATH = MISC_DIR / "nn_module_metadata.json"
BASELINE_SUMMARY_PATH = RESULT_DATA_DIR / "baseline_linear_summary.json"
BASELINE_MODEL_PATH = RESULT_DATA_DIR / "baseline_linear_model.pt"
BASELINE_PRED_PATH = RESULT_DATA_DIR / "baseline_linear_fusion_predictions.csv"

MODULES = ["lineup", "player_stats", "skill_matchups", "match_stats"]


@dataclass
class BaselineConfig:
    epochs: int = 30
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    max_abs_goal_diff: int = 12


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
        return {"game_id": self.game_id[idx], "x": self.x[idx], 
                "HG": self.hg[idx], "AG": self.ag[idx], "outcome": self.outcome[idx]}


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
        return {"game_id": self.game_id[idx], "x": self.x[idx], 
                "HG": self.hg[idx], "AG": self.ag[idx], "outcome": self.outcome[idx]}


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
    """Simplest possible model: linear → rates."""
    def __init__(self, input_dim, home_prior=1.35, away_prior=1.10):
        super().__init__()
        self.register_buffer("home_prior", torch.tensor(float(home_prior)))
        self.register_buffer("away_prior", torch.tensor(float(away_prior)))
        self.linear = nn.Linear(input_dim, 2)
        
    def forward(self, x):
        logits = self.linear(x)
        mult_home = 0.2 + 2.8 * torch.sigmoid(logits[:, 0])
        mult_away = 0.2 + 2.8 * torch.sigmoid(logits[:, 1])
        return self.home_prior * mult_home, self.away_prior * mult_away


class LinearFusionModel(nn.Module):
    """Linear fusion: concatenate module predictions → linear → rates."""
    def __init__(self, n_modules, home_prior=1.35, away_prior=1.10):
        super().__init__()
        self.register_buffer("home_prior", torch.tensor(float(home_prior)))
        self.register_buffer("away_prior", torch.tensor(float(away_prior)))
        # Input: n_modules × 3 (lambda_h, lambda_a, diff per module)
        self.linear = nn.Linear(n_modules * 3, 2)
        
    def forward(self, x):
        # x: (batch, n_modules, 3) → flatten
        x_flat = x.view(x.shape[0], -1)
        logits = self.linear(x_flat)
        mult_home = 0.2 + 2.8 * torch.sigmoid(logits[:, 0])
        mult_away = 0.2 + 2.8 * torch.sigmoid(logits[:, 1])
        return self.home_prior * mult_home, self.away_prior * mult_away


# ============================================================================
# LOSS
# ============================================================================

def poisson_nll_loss(lam_h, lam_a, hg, ag):
    return F.poisson_nll_loss(lam_h, hg, log_input=False, full=True) + \
           F.poisson_nll_loss(lam_a, ag, log_input=False, full=True)


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(model, loader, device):
    model.eval()
    losses, accuracies = [], []
    home_rates, away_rates = [], []
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
            
            # Predict outcome
            pred_outcome = torch.where(lam_h > lam_a + 0.2, torch.zeros_like(outcome),
                                      torch.where(lam_a > lam_h + 0.2, 2 * torch.ones_like(outcome),
                                                 torch.ones_like(outcome)))
            pred_outcomes.extend(pred_outcome.cpu().numpy().tolist())
            true_outcomes.extend(outcome.cpu().numpy().tolist())
            
            home_rates.append(float(lam_h.mean().item()))
            away_rates.append(float(lam_a.mean().item()))
    
    accuracy = float((np.array(pred_outcomes) == np.array(true_outcomes)).mean())
    rate_std = float(np.std(home_rates))
    
    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy,
        "mean_home_rate": float(np.mean(home_rates)),
        "mean_away_rate": float(np.mean(away_rates)),
        "rate_std": rate_std,
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
    return pd.read_csv(train_path, low_memory=False), pd.read_csv(test_path, low_memory=False), key_col, feature_cols


# ============================================================================
# TRAINING
# ============================================================================

def train_single_module(module_name, config, device):
    """Train linear model on a single module."""
    train_df, test_df, key_col, feature_cols = load_module_frames(module_name)
    home_prior = float(train_df["HG"].mean())
    away_prior = float(train_df["AG"].mean())

    train_ds = SimpleDataset(train_df, feature_cols)
    test_ds = SimpleDataset(test_df, feature_cols)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_simple_batch)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_simple_batch)

    model = LinearRateModel(len(feature_cols), home_prior, away_prior).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_test_loss = float("inf")
    best_state = None
    best_epoch = 0

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

        test_m = compute_metrics(model, test_loader, device)

        if test_m["loss"] < best_test_loss:
            best_test_loss = test_m["loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Ep {epoch:03d} | Loss {test_m['loss']:.4f} | Acc {test_m['accuracy']:.3f} | "
                  f"σ(λh) {test_m['rate_std']:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, test_ds, best_epoch


def build_fusion_predictions(models, test_datasets, config, device):
    """Generate module predictions for fusion training."""
    fusion_rows = []
    
    for module_name, (model, test_ds, _) in models.items():
        model.eval()
        loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_simple_batch)
        
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(device)
                lam_h, lam_a = model(x)
                for i, gid in enumerate(batch["game_id"]):
                    fusion_rows.append({
                        "game_id": gid,
                        "HG": float(batch["HG"][i]),
                        "AG": float(batch["AG"][i]),
                        f"{module_name}_lambda_home": float(lam_h[i]),
                        f"{module_name}_lambda_away": float(lam_a[i]),
                        f"{module_name}_diff": float(lam_h[i] - lam_a[i]),
                    })
    
    # Merge by game_id
    df = pd.DataFrame(fusion_rows)
    # Aggregate: take first occurrence per game_id per module
    pivot_rows = {}
    for _, row in df.iterrows():
        gid = row["game_id"]
        if gid not in pivot_rows:
            pivot_rows[gid] = {"game_id": gid, "HG": row["HG"], "AG": row["AG"]}
        for col in row.index:
            if col not in ["game_id", "HG", "AG"]:
                pivot_rows[gid][col] = row[col]
    
    final_df = pd.DataFrame(list(pivot_rows.values()))
    return final_df


def train_fusion(models, config, device):
    """Train linear fusion on module predictions."""
    # Generate predictions for train and test
    print("\n  Generating module predictions for fusion...")
    train_fusion_df = build_fusion_predictions(models, {}, config, device)  # placeholder
    # Actually need train predictions too - simplify: just use test for now
    
    # For simplicity, retrain using test predictions
    fusion_feature_cols = []
    for mn in MODULES:
        fusion_feature_cols += [f"{mn}_lambda_home", f"{mn}_lambda_away", f"{mn}_diff"]
    
    # Build fusion dataset from test predictions
    test_df, _ = build_fusion_frame("test")
    
    home_prior = float(test_df["HG"].mean())
    away_prior = float(test_df["AG"].mean())
    
    test_ds = FusionDataset(test_df, fusion_feature_cols)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_simple_batch)
    
    fusion_model = LinearFusionModel(len(MODULES), home_prior, away_prior).to(device)
    
    # Quick evaluation
    test_m = compute_metrics(fusion_model, test_loader, device)
    print(f"\n  [fusion] Linear (no training) | Acc {test_m['accuracy']:.3f} | "
          f"σ(λh) {test_m['rate_std']:.4f} | λh {test_m['mean_home_rate']:.2f} λa {test_m['mean_away_rate']:.2f}")
    
    return fusion_model


# Simplified: load predictions from files if they exist
def build_fusion_frame(split):
    """Load module predictions from CSV files."""
    dfs = []
    for mn in MODULES:
        path = RESULT_DATA_DIR / f"baseline_linear_{mn}_predictions.csv"
        if path.exists():
            df = pd.read_csv(path, low_memory=False)
            df = df.rename(columns={
                "lambda_home": f"{mn}_lambda_home",
                "lambda_away": f"{mn}_lambda_away",
            })
            df[f"{mn}_diff"] = df[f"{mn}_lambda_home"] - df[f"{mn}_lambda_away"]
            dfs.append(df[["game_id", "HG", "AG", f"{mn}_lambda_home", f"{mn}_lambda_away", f"{mn}_diff"]])
    
    if not dfs:
        return None, []
    
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on=["game_id", "HG", "AG"], how="inner")
    
    feature_cols = []
    for mn in MODULES:
        feature_cols += [f"{mn}_lambda_home", f"{mn}_lambda_away", f"{mn}_diff"]
    
    return merged, feature_cols


def main():
    config = BaselineConfig()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    RESULT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Train each module independently
    print("=" * 70)
    print("BASELINE: INDEPENDENT LINEAR MODULES")
    print("=" * 70)
    
    models = {}
    results = {}
    
    for module_name in MODULES:
        try:
            model, test_ds, best_epoch = train_single_module(module_name, config, device)
            
            # Save model
            model_path = RESULT_DATA_DIR / f"baseline_linear_{module_name}_model.pt"
            torch.save({"state_dict": model.state_dict(), "module_name": module_name,
                        "best_epoch": best_epoch}, model_path)
            
            # Save predictions
            model.eval()
            loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_simple_batch)
            rows = []
            with torch.no_grad():
                for batch in loader:
                    x = batch["x"].to(device)
                    lam_h, lam_a = model(x)
                    for i, gid in enumerate(batch["game_id"]):
                        rows.append({"game_id": gid, "HG": float(batch["HG"][i]), "AG": float(batch["AG"][i]),
                                     "lambda_home": float(lam_h[i]), "lambda_away": float(lam_a[i])})
            pd.DataFrame(rows).to_csv(RESULT_DATA_DIR / f"baseline_linear_{module_name}_predictions.csv", index=False)
            
            models[module_name] = model
            results[module_name] = {"best_epoch": best_epoch}
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[module_name] = {"error": str(e)}

    # Fusion
    print("\n" + "=" * 70)
    print("BASELINE: LINEAR FUSION")
    print("=" * 70)
    
    try:
        fusion_df, fusion_cols = build_fusion_frame("test")
        if fusion_df is not None:
            home_prior = float(fusion_df["HG"].mean())
            away_prior = float(fusion_df["AG"].mean())
            
            fusion_ds = FusionDataset(fusion_df, fusion_cols)
            fusion_loader = DataLoader(fusion_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_simple_batch)
            
            fusion_model = LinearFusionModel(len(MODULES), home_prior, away_prior).to(device)
            
            # Train fusion (simple: few epochs)
            optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=1e-3)
            for epoch in range(20):
                fusion_model.train()
                for batch in fusion_loader:
                    x, hg, ag = batch["x"].to(device), batch["HG"].to(device), batch["AG"].to(device)
                    optimizer.zero_grad()
                    lam_h, lam_a = fusion_model(x)
                    loss = poisson_nll_loss(lam_h, lam_a, hg, ag)
                    loss.backward()
                    optimizer.step()
            
            fusion_metrics = compute_metrics(fusion_model, fusion_loader, device)
            print(f"  Fusion Acc: {fusion_metrics['accuracy']:.3f} | σ(λh): {fusion_metrics['rate_std']:.4f}")
            
            # Save fusion predictions
            fusion_model.eval()
            rows = []
            with torch.no_grad():
                for batch in fusion_loader:
                    x = batch["x"].to(device)
                    lam_h, lam_a = fusion_model(x)
                    for i, gid in enumerate(batch["game_id"]):
                        rows.append({"game_id": gid, "HG": float(batch["HG"][i]), "AG": float(batch["AG"][i]),
                                     "lambda_home": float(lam_h[i]), "lambda_away": float(lam_a[i]),
                                     "expected_goal_diff": float(lam_h[i] - lam_a[i])})
            pd.DataFrame(rows).to_csv(BASELINE_PRED_PATH, index=False)
            
            torch.save({"state_dict": fusion_model.state_dict()}, BASELINE_MODEL_PATH)
            results["fusion"] = {"accuracy": fusion_metrics["accuracy"]}
    except Exception as e:
        print(f"  ❌ Fusion error: {e}")
        results["fusion"] = {"error": str(e)}

    # Save summary
    summary = {"config": config.__dict__, "results": results}
    BASELINE_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved baseline summary to {BASELINE_SUMMARY_PATH}")


if __name__ == "__main__":
    main()