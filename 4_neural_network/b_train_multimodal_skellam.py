#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""COMPROMISE (UPDATED with class weights): Shallow multimodal model.

Key changes:
1. Class weights added to prevent collapse to majority (home win) class
2. Both modules AND fusion use balanced cross-entropy
3. Output includes predicted_result column
4. Everything else stays as the original compromise architecture
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
COMPROMISE_SUMMARY_PATH = RESULT_DATA_DIR / "compromise_model_summary.json"
COMPROMISE_MODEL_PATH = RESULT_DATA_DIR / "compromise_fusion_model.pt"
COMPROMISE_PRED_PATH = RESULT_DATA_DIR / "compromise_fusion_predictions.csv"

MODULES = ["lineup", "player_stats", "skill_matchups", "match_stats",
           "league", "season", "matchup"]


@dataclass
class CompromiseConfig:
    epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 32
    dropout: float = 0.1
    seed: int = 42
    outcome_weight: float = 0.5
    val_fraction: float = 0.15
    use_class_weights: bool = True      # NEW


# ============================================================================
# DATASETS
# ============================================================================

class ModuleDataset(Dataset):
    def __init__(self, frame, feature_cols, key_col):
        self.keys = frame[key_col].astype(str).values
        self.x = torch.tensor(frame[feature_cols].astype(float).values, dtype=torch.float32)
        self.hg = torch.tensor(frame["HG"].astype(float).values, dtype=torch.float32)
        self.ag = torch.tensor(frame["AG"].astype(float).values, dtype=torch.float32)
        self.outcome = torch.tensor(
            np.where(frame["HG"].values > frame["AG"].values, 0,
                     np.where(frame["HG"].values == frame["AG"].values, 1, 2)),
            dtype=torch.long
        )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        return {"key": self.keys[idx], "x": self.x[idx], "HG": self.hg[idx],
                "AG": self.ag[idx], "outcome": self.outcome[idx]}


class FusionDataset(Dataset):
    def __init__(self, frame, feature_cols):
        self.game_id = frame["game_id"].astype(str).values
        self.x = torch.tensor(frame[feature_cols].astype(float).values, dtype=torch.float32)
        self.hg = torch.tensor(frame["HG"].astype(float).values, dtype=torch.float32)
        self.ag = torch.tensor(frame["AG"].astype(float).values, dtype=torch.float32)
        self.outcome = torch.tensor(
            np.where(frame["HG"].values > frame["AG"].values, 0,
                     np.where(frame["HG"].values == frame["AG"].values, 1, 2)),
            dtype=torch.long
        )

    def __len__(self):
        return len(self.game_id)

    def __getitem__(self, idx):
        return {"game_id": self.game_id[idx], "x": self.x[idx], "HG": self.hg[idx],
                "AG": self.ag[idx], "outcome": self.outcome[idx]}


def collate_batch(batch):
    return {
        "key": [x.get("key", x.get("game_id")) for x in batch],
        "x": torch.stack([x["x"] for x in batch]),
        "HG": torch.stack([x["HG"] for x in batch]),
        "AG": torch.stack([x["AG"] for x in batch]),
        "outcome": torch.stack([x["outcome"] for x in batch]),
    }


# ============================================================================
# MODELS
# ============================================================================

class ShallowRateModel(nn.Module):
    """One hidden layer + dual output (rates + outcome logits)."""

    def __init__(self, input_dim, hidden_dim=32, dropout=0.1, home_prior=1.5, away_prior=1.2):
        super().__init__()
        self.register_buffer("home_prior", torch.tensor(float(home_prior)))
        self.register_buffer("away_prior", torch.tensor(float(away_prior)))

        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.rate_head = nn.Linear(hidden_dim, 2)
        self.outcome_head = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        h = self.hidden(x)

        logits = self.rate_head(h)
        mult_home = 0.2 + 2.8 * torch.sigmoid(logits[:, 0])
        mult_away = 0.2 + 2.8 * torch.sigmoid(logits[:, 1])
        lam_home = self.home_prior * mult_home
        lam_away = self.away_prior * mult_away

        outcome_logits = self.outcome_head(h)

        return lam_home, lam_away, outcome_logits


class ShallowFusionModel(nn.Module):
    """Simple fusion: one hidden layer on concatenated module predictions."""

    def __init__(self, input_dim, hidden_dim=32, dropout=0.1, home_prior=1.5, away_prior=1.2):
        super().__init__()
        self.register_buffer("home_prior", torch.tensor(float(home_prior)))
        self.register_buffer("away_prior", torch.tensor(float(away_prior)))

        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.rate_head = nn.Linear(hidden_dim, 2)
        self.outcome_head = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        h = self.hidden(x)

        logits = self.rate_head(h)
        mult_home = 0.2 + 2.8 * torch.sigmoid(logits[:, 0])
        mult_away = 0.2 + 2.8 * torch.sigmoid(logits[:, 1])
        lam_home = self.home_prior * mult_home
        lam_away = self.away_prior * mult_away

        outcome_logits = self.outcome_head(h)

        return lam_home, lam_away, outcome_logits


# ============================================================================
# LOSS (with class weights)
# ============================================================================

def combined_loss(lam_h, lam_a, outcome_logits, hg, ag, outcome,
                  outcome_weight=0.5, class_weights=None):
    poisson_loss = F.poisson_nll_loss(lam_h, hg, log_input=False, full=True) + \
                   F.poisson_nll_loss(lam_a, ag, log_input=False, full=True)
    outcome_loss = F.cross_entropy(outcome_logits, outcome, weight=class_weights)
    return poisson_loss + outcome_weight * outcome_loss


# ============================================================================
# METRICS (with class weights)
# ============================================================================

def compute_metrics(model, loader, device, class_weights=None):
    model.eval()
    losses = []
    home_rates, away_rates = [], []
    pred_outcomes, true_outcomes = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            hg = batch["HG"].to(device)
            ag = batch["AG"].to(device)
            outcome = batch["outcome"].to(device)

            lam_h, lam_a, outcome_logits = model(x)
            loss = combined_loss(lam_h, lam_a, outcome_logits, hg, ag, outcome,
                                  class_weights=class_weights)
            losses.append(float(loss.item()))

            pred_outcome = torch.argmax(outcome_logits, dim=1)
            pred_outcomes.extend(pred_outcome.cpu().numpy().tolist())
            true_outcomes.extend(outcome.cpu().numpy().tolist())

            home_rates.append(float(lam_h.mean().item()))
            away_rates.append(float(lam_a.mean().item()))

    accuracy = float((np.array(pred_outcomes) == np.array(true_outcomes)).mean())
    rate_std = float(np.std(home_rates))

    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(true_outcomes, pred_outcomes):
        cm[t, p] += 1

    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy,
        "mean_home_rate": float(np.mean(home_rates)),
        "mean_away_rate": float(np.mean(away_rates)),
        "rate_std": rate_std,
        "confusion_matrix": cm.tolist(),
    }


# ============================================================================
# UTILITIES
# ============================================================================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_train_val(frame, val_fraction=0.15):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1.")

    frame = frame.copy().sort_values("game_id").reset_index(drop=True)
    split_idx = int(len(frame) * (1 - val_fraction))
    split_idx = min(max(split_idx, 1), len(frame) - 1)
    train_df = frame.iloc[:split_idx].copy()
    val_df = frame.iloc[split_idx:].copy()
    return train_df, val_df


def compute_class_weights(outcome: np.ndarray, num_classes: int = 3) -> torch.Tensor:
    """Inverse-frequency class weights so the model doesn't collapse to majority class."""
    counts = np.bincount(outcome, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def outcome_array(hg, ag):
    return np.where(hg > ag, 0, np.where(hg == ag, 1, 2))


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

def train_module(module_name, config, device):
    train_df, test_df, key_col, feature_cols = load_module_frames(module_name)
    home_prior = float(train_df["HG"].mean())
    away_prior = float(train_df["AG"].mean())

    train_df, val_df = split_train_val(train_df, config.val_fraction)

    train_ds = ModuleDataset(train_df, feature_cols, key_col)
    val_ds = ModuleDataset(val_df, feature_cols, key_col)
    test_ds = ModuleDataset(test_df, feature_cols, key_col)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch)

    # Compute class weights from training split
    class_weights = None
    if config.use_class_weights:
        train_outcomes = outcome_array(train_df["HG"].values, train_df["AG"].values)
        class_weights = compute_class_weights(train_outcomes).to(device)

    model = ShallowRateModel(len(feature_cols), config.hidden_dim, config.dropout, home_prior, away_prior).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_val_acc = 0.0
    best_test_acc = 0.0
    best_state = None
    best_epoch = 0

    print(f"\n  [{module_name}] Training (Features: {len(feature_cols)})")
    if class_weights is not None:
        print(f"    Class weights: {[round(w, 2) for w in class_weights.tolist()]}")

    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            hg = batch["HG"].to(device)
            ag = batch["AG"].to(device)
            outcome = batch["outcome"].to(device)

            optimizer.zero_grad()
            lam_h, lam_a, outcome_logits = model(x)
            loss = combined_loss(lam_h, lam_a, outcome_logits, hg, ag, outcome,
                                  config.outcome_weight, class_weights)
            loss.backward()
            optimizer.step()

        val_m = compute_metrics(model, val_loader, device, class_weights)
        test_m = compute_metrics(model, test_loader, device, class_weights)

        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            best_test_acc = test_m["accuracy"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Ep {epoch:03d} | ValAcc {val_m['accuracy']:.3f} | TestAcc {test_m['accuracy']:.3f} | "
                  f"Loss {val_m['loss']:.4f} | σ(λh) {test_m['rate_std']:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"    ✓ Best val accuracy {best_val_acc:.3f} at epoch {best_epoch} (test acc {best_test_acc:.3f})")

    # Save predictions
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            lam_h, lam_a, outcome_logits = model(x)
            pred_outcome = torch.argmax(outcome_logits, dim=1)
            for i, k in enumerate(batch["key"]):
                rows.append({key_col: k, "HG": float(batch["HG"][i]), "AG": float(batch["AG"][i]),
                             "lambda_home": float(lam_h[i]), "lambda_away": float(lam_a[i]),
                             "predicted_result": int(pred_outcome[i])})
    pd.DataFrame(rows).to_csv(RESULT_DATA_DIR / f"compromise_{module_name}_predictions.csv", index=False)

    torch.save({"state_dict": model.state_dict(), "module_name": module_name,
                "best_epoch": best_epoch, "best_accuracy": best_test_acc},
               RESULT_DATA_DIR / f"compromise_{module_name}_model.pt")

    return model, {"best_accuracy": best_test_acc, "best_epoch": best_epoch}


def train_fusion(config, device):
    """Simple fusion on module predictions (7 modules)."""
    print("\n" + "=" * 70)
    print("TRAINING FUSION (Shallow, 7 modules, class-weighted)")
    print("=" * 70)

    dfs = []
    for mn in MODULES:
        path = RESULT_DATA_DIR / f"compromise_{mn}_predictions.csv"
        if path.exists():
            df = pd.read_csv(path, low_memory=False)
            df = df.rename(columns={
                "lambda_home": f"{mn}_lambda_home",
                "lambda_away": f"{mn}_lambda_away",
            })
            df[f"{mn}_diff"] = df[f"{mn}_lambda_home"] - df[f"{mn}_lambda_away"]
            dfs.append(df[["game_id", "HG", "AG", f"{mn}_lambda_home", f"{mn}_lambda_away", f"{mn}_diff"]])

    if not dfs:
        print("  ❌ No module predictions found!")
        return

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on=["game_id", "HG", "AG"], how="inner")

    feature_cols = []
    for mn in MODULES:
        feature_cols += [f"{mn}_lambda_home", f"{mn}_lambda_away", f"{mn}_diff"]

    home_prior = float(merged["HG"].mean())
    away_prior = float(merged["AG"].mean())

    train_merged, val_merged = split_train_val(merged, config.val_fraction)
    fusion_train_ds = FusionDataset(train_merged, feature_cols)
    fusion_val_ds = FusionDataset(val_merged, feature_cols)
    fusion_test_ds = FusionDataset(merged, feature_cols)
    fusion_train_loader = DataLoader(fusion_train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_batch)
    fusion_val_loader = DataLoader(fusion_val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch)
    fusion_test_loader = DataLoader(fusion_test_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch)

    # Class weights for fusion
    class_weights = None
    if config.use_class_weights:
        train_outcomes = outcome_array(train_merged["HG"].values, train_merged["AG"].values)
        class_weights = compute_class_weights(train_outcomes).to(device)
        print(f"  Fusion class weights: {[round(w, 2) for w in class_weights.tolist()]}")

    model = ShallowFusionModel(len(feature_cols), config.hidden_dim, config.dropout, home_prior, away_prior).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(30):
        model.train()
        for batch in fusion_train_loader:
            x = batch["x"].to(device)
            hg = batch["HG"].to(device)
            ag = batch["AG"].to(device)
            outcome = batch["outcome"].to(device)

            optimizer.zero_grad()
            lam_h, lam_a, outcome_logits = model(x)
            loss = combined_loss(lam_h, lam_a, outcome_logits, hg, ag, outcome,
                                  config.outcome_weight, class_weights)
            loss.backward()
            optimizer.step()

        val_m = compute_metrics(model, fusion_val_loader, device, class_weights)
        test_m = compute_metrics(model, fusion_test_loader, device, class_weights)
        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Ep {epoch:03d} | ValAcc {val_m['accuracy']:.3f} | TestAcc {test_m['accuracy']:.3f} | "
                  f"σ(λh) {test_m['rate_std']:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save fusion predictions
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in fusion_test_loader:
            x = batch["x"].to(device)
            lam_h, lam_a, outcome_logits = model(x)
            pred_outcome = torch.argmax(outcome_logits, dim=1)
            for i, gid in enumerate(batch["key"]):
                rows.append({"game_id": gid, "HG": float(batch["HG"][i]), "AG": float(batch["AG"][i]),
                             "lambda_home": float(lam_h[i]), "lambda_away": float(lam_a[i]),
                             "expected_goal_diff": float(lam_h[i] - lam_a[i]),
                             "predicted_result": int(pred_outcome[i])})
    pd.DataFrame(rows).to_csv(COMPROMISE_PRED_PATH, index=False)

    torch.save({"state_dict": model.state_dict(), "best_accuracy": best_val_acc}, COMPROMISE_MODEL_PATH)

    final_metrics = compute_metrics(model, fusion_test_loader, device, class_weights)
    print(f"\n  Final Fusion Accuracy: {final_metrics['accuracy']:.3f}")
    print(f"  Confusion Matrix (rows=actual, cols=predicted):")
    cm = np.array(final_metrics["confusion_matrix"])
    print(cm)
    print(f"\n  Saved to: {COMPROMISE_PRED_PATH}")

    return {"best_accuracy": best_val_acc, "test_accuracy": final_metrics["accuracy"]}


def main():
    config = CompromiseConfig()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Compromise model: 1 hidden layer ({config.hidden_dim} units)")
    print(f"Modules: {MODULES}")
    print(f"Class weights enabled: {config.use_class_weights}")
    RESULT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for mn in MODULES:
        try:
            _, module_result = train_module(mn, config, device)
            results[mn] = module_result
        except Exception as e:
            print(f"  ❌ Error training {mn}: {e}")
            results[mn] = {"error": str(e)}

    fusion_result = train_fusion(config, device)
    results["fusion"] = fusion_result

    summary = {"config": config.__dict__, "results": results}
    COMPROMISE_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved compromise summary to {COMPROMISE_SUMMARY_PATH}")


if __name__ == "__main__":
    main()