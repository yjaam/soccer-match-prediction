#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train league, season, AND matchup embeddings and export module-ready files.

Three context types are learned:
1. League embedding (5 leagues)
2. Season embedding (12 seasons)
3. Matchup embedding (home_team__away_team pairs)

This captures rivalry effects, head-to-head patterns, and style matchups.
"""

from __future__ import annotations

import argparse
import json
import sys
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
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.game_id_context import infer_league_and_season

FINAL_DATA_DIR = PROJECT_DIR / "final_data"
DATA_DIR = PROJECT_DIR / "data"
MISC_DIR = PROJECT_DIR / "misc"
RESULT_DATA_DIR = PROJECT_DIR / "result_data"

TRAIN_SPLIT_SOURCE = FINAL_DATA_DIR / "lineup_embeddings_TRAIN.csv"
TEST_SPLIT_SOURCE = FINAL_DATA_DIR / "lineup_embeddings_TEST.csv"
TARGETS_PATH = DATA_DIR / "target_variables.csv"

LEAGUE_TRAIN_OUTPUT_PATH = FINAL_DATA_DIR / "league_embeddings_TRAIN.csv"
LEAGUE_TEST_OUTPUT_PATH = FINAL_DATA_DIR / "league_embeddings_TEST.csv"
SEASON_TRAIN_OUTPUT_PATH = FINAL_DATA_DIR / "season_embeddings_TRAIN.csv"
SEASON_TEST_OUTPUT_PATH = FINAL_DATA_DIR / "season_embeddings_TEST.csv"
MATCHUP_TRAIN_OUTPUT_PATH = FINAL_DATA_DIR / "matchup_embeddings_TRAIN.csv"
MATCHUP_TEST_OUTPUT_PATH = FINAL_DATA_DIR / "matchup_embeddings_TEST.csv"

META_OUTPUT_PATH = MISC_DIR / "context_embedding_metadata.json"
LEAGUE_MODEL_OUTPUT_PATH = RESULT_DATA_DIR / "league_embedding_model.pt"
SEASON_MODEL_OUTPUT_PATH = RESULT_DATA_DIR / "season_embedding_model.pt"
MATCHUP_MODEL_OUTPUT_PATH = RESULT_DATA_DIR / "matchup_embedding_model.pt"


@dataclass
class Config:
    league_dim: int = 8
    season_dim: int = 8
    matchup_dim: int = 4
    hidden_dim: int = 32
    epochs: int = 50
    batch_size: int = 512
    lr: float = 5e-3
    weight_decay: float = 1e-4
    dropout: float = 0.1
    seed: int = 42
    early_stopping_patience: int = 10
    lr_patience: int = 5
    lr_factor: float = 0.5
    min_lr: float = 1e-5
    min_matchup_count: int = 4  # Keep only more stable matchup pairs


class ContextDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, context_cols: list[str]):
        self.item_indices = torch.tensor(frame[context_cols].values, dtype=torch.long)
        self.hg = torch.tensor(frame["HG"].values, dtype=torch.float32)
        self.ag = torch.tensor(frame["AG"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.hg)

    def __getitem__(self, idx):
        return {"item_indices": self.item_indices[idx], "HG": self.hg[idx], "AG": self.ag[idx]}


class SingleContextRegressor(nn.Module):
    """Predicts goal rates from a single context."""
    
    def __init__(self, n_items, emb_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(n_items, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, item_idx):
        x = self.emb(item_idx)
        out = self.mlp(x)
        lam_h = F.softplus(out[:, 0]) + 0.1
        lam_a = F.softplus(out[:, 1]) + 0.1
        return lam_h, lam_a


class CombinedContextRegressor(nn.Module):
    """Predicts goal rates from combined league + season + matchup."""
    
    def __init__(self, n_leagues, league_dim, n_seasons, season_dim, 
                 n_matchups, matchup_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.league_emb = nn.Embedding(n_leagues, league_dim)
        self.season_emb = nn.Embedding(n_seasons, season_dim)
        self.matchup_emb = nn.Embedding(n_matchups, matchup_dim)
        
        total_dim = league_dim + season_dim + matchup_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, league_idx, season_idx, matchup_idx):
        league_emb = self.league_emb(league_idx)
        season_emb = self.season_emb(season_idx)
        matchup_emb = self.matchup_emb(matchup_idx)
        x = torch.cat([league_emb, season_emb, matchup_emb], dim=-1)
        out = self.mlp(x)
        lam_h = F.softplus(out[:, 0]) + 0.1
        lam_a = F.softplus(out[:, 1]) + 0.1
        return lam_h, lam_a


def poisson_nll_loss(lam_h, lam_a, hg, ag):
    loss_h = F.poisson_nll_loss(lam_h, hg, log_input=False, full=True)
    loss_a = F.poisson_nll_loss(lam_a, ag, log_input=False, full=True)
    return loss_h + loss_a


def compute_metrics(model, loader, device):
    model.eval()
    losses, mses, maes = [], [], []
    home_rates, away_rates = [], []
    
    with torch.no_grad():
        for batch in loader:
            # Handle both single and combined context
            idx = batch["item_indices"].to(device)
            hg = batch["HG"].to(device)
            ag = batch["AG"].to(device)
            
            if idx.dim() == 1:
                # Single context
                lam_h, lam_a = model(idx)
            else:
                # Combined: league, season, matchup
                lam_h, lam_a = model(idx[:, 0], idx[:, 1], idx[:, 2])
            
            loss = poisson_nll_loss(lam_h, lam_a, hg, ag)
            losses.append(float(loss.item()))
            mses.append(float(F.mse_loss(lam_h - lam_a, hg - ag).item()))
            maes.append(float(F.l1_loss(lam_h - lam_a, hg - ag).item()))
            home_rates.append(float(lam_h.mean().item()))
            away_rates.append(float(lam_a.mean().item()))
    
    return {
        "loss": float(np.mean(losses)),
        "mse": float(np.mean(mses)),
        "mae": float(np.mean(maes)),
        "mean_home_rate": float(np.mean(home_rates)),
        "mean_away_rate": float(np.mean(away_rates)),
    }


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split_keys(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing split source: {path}")
    frame = pd.read_csv(path, low_memory=False)
    if "game_id" not in frame.columns:
        raise KeyError(f"game_id missing in split source: {path}")
    return frame[["game_id"]].dropna().drop_duplicates().copy()


def load_targets(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing targets file: {path}")
    frame = pd.read_csv(path, low_memory=False)
    required = {"game_id", "HG", "AG"}
    if not required.issubset(frame.columns):
        raise KeyError(f"targets file missing required columns: {required}")
    out = frame[["game_id", "HG", "AG"]].copy()
    out["HG"] = pd.to_numeric(out["HG"], errors="coerce")
    out["AG"] = pd.to_numeric(out["AG"], errors="coerce")
    out = out.dropna(subset=["game_id", "HG", "AG"]).drop_duplicates(subset=["game_id"], keep="first")
    return out


def add_context(frame):
    ctx = frame["game_id"].astype(str).apply(infer_league_and_season).apply(pd.Series)
    out = frame.merge(ctx[["game_id", "league", "season"]], on="game_id", how="left")
    out["league"] = out["league"].fillna("__UNK__")
    out["season"] = out["season"].fillna("__UNK__")
    
    # Extract matchup from game_id: format is "YYYYMMDD_HOME_AWAY"
    def extract_matchup(game_id):
        parts = str(game_id).split("_")
        if len(parts) >= 3:
            return f"{parts[1]}__{parts[2]}"
        return "__UNK__"
    
    out["matchup"] = out["game_id"].apply(extract_matchup)
    return out


def build_vocab_with_min_count(train_values, min_count=2):
    """Build vocab, keeping only items that appear at least min_count times."""
    value_counts = train_values.astype(str).value_counts()
    valid_values = value_counts[value_counts >= min_count].index.tolist()
    
    vocab = {"__UNK__": 0}
    for i, value in enumerate(sorted(valid_values), start=1):
        if value == "__UNK__":
            continue
        vocab[value] = i
    return vocab


def encode_with_vocab(values, vocab):
    return values.astype(str).map(lambda x: vocab.get(x, 0)).astype(int).values


def train_single_context_model(train_df, test_df, context_col, emb_dim, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vocab = build_vocab_with_min_count(train_df[context_col], config.min_matchup_count if context_col == "matchup" else 1)
    n_items = max(vocab.values()) + 1
    
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["item_idx"] = encode_with_vocab(train_df[context_col], vocab)
    test_df["item_idx"] = encode_with_vocab(test_df[context_col], vocab)
    
    model = SingleContextRegressor(n_items, emb_dim, config.hidden_dim, config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config.lr_factor, patience=config.lr_patience, min_lr=config.min_lr
    )
    
    train_loader = DataLoader(ContextDataset(train_df, ["item_idx"]), batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(ContextDataset(test_df, ["item_idx"]), batch_size=config.batch_size, shuffle=False)
    
    best_test_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    best_state = None
    
    print("=" * 70)
    print(f"TRAINING: {context_col} embedding (n_items={n_items}, dim={emb_dim})")
    print("=" * 70)
    
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            idx = batch["item_indices"].to(device)
            hg = batch["HG"].to(device)
            ag = batch["AG"].to(device)
            
            optimizer.zero_grad()
            lam_h, lam_a = model(idx)
            loss = poisson_nll_loss(lam_h, lam_a, hg, ag)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.item()))
        
        train_metrics = compute_metrics(model, train_loader, device)
        test_metrics = compute_metrics(model, test_loader, device)
        scheduler.step(test_metrics["loss"])
        
        if test_metrics["loss"] < best_test_loss - 1e-4:
            best_test_loss = test_metrics["loss"]
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
        
        lr_now = optimizer.param_groups[0]['lr']
        if epoch % 5 == 0 or epoch == 1:
            print(f"[{context_col}] Ep {epoch:03d}/{config.epochs} | "
                  f"Train {train_metrics['loss']:.4f} | Test {test_metrics['loss']:.4f} | "
                  f"λh {test_metrics['mean_home_rate']:.2f} λa {test_metrics['mean_away_rate']:.2f} | "
                  f"LR {lr_now:.1e}")
        
        if epochs_no_improve >= config.early_stopping_patience:
            print(f"  ⏹ Early stop at epoch {epoch} (best ep {best_epoch}, Loss {best_test_loss:.4f})")
            break
    
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  ✓ Restored best model from epoch {best_epoch}")
    
    meta = {
        "context_col": context_col,
        "vocab": vocab,
        "embedding_dim": emb_dim,
        "best_epoch": best_epoch,
        "best_test_loss": best_test_loss,
        "config": config.__dict__,
    }
    return model, meta


def train_combined_context_model(train_df, test_df, config):
    """Train combined model with all three context types."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build vocabs
    league_vocab = build_vocab_with_min_count(train_df["league"], 1)
    season_vocab = build_vocab_with_min_count(train_df["season"], 1)
    matchup_vocab = build_vocab_with_min_count(train_df["matchup"], config.min_matchup_count)
    
    n_leagues = max(league_vocab.values()) + 1
    n_seasons = max(season_vocab.values()) + 1
    n_matchups = max(matchup_vocab.values()) + 1
    
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    train_df["league_idx"] = encode_with_vocab(train_df["league"], league_vocab)
    train_df["season_idx"] = encode_with_vocab(train_df["season"], season_vocab)
    train_df["matchup_idx"] = encode_with_vocab(train_df["matchup"], matchup_vocab)
    
    test_df["league_idx"] = encode_with_vocab(test_df["league"], league_vocab)
    test_df["season_idx"] = encode_with_vocab(test_df["season"], season_vocab)
    test_df["matchup_idx"] = encode_with_vocab(test_df["matchup"], matchup_vocab)
    
    model = CombinedContextRegressor(
        n_leagues, config.league_dim,
        n_seasons, config.season_dim,
        n_matchups, config.matchup_dim,
        config.hidden_dim, config.dropout
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config.lr_factor, patience=config.lr_patience, min_lr=config.min_lr
    )
    
    train_loader = DataLoader(
        ContextDataset(train_df, ["league_idx", "season_idx", "matchup_idx"]),
        batch_size=config.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        ContextDataset(test_df, ["league_idx", "season_idx", "matchup_idx"]),
        batch_size=config.batch_size, shuffle=False
    )
    
    best_test_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    best_state = None
    
    print("=" * 70)
    print(f"TRAINING: combined context (league+season+matchup)")
    print("=" * 70)
    print(f"Leagues: {n_leagues} | Seasons: {n_seasons} | Matchups: {n_matchups}")
    
    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch in train_loader:
            idx = batch["item_indices"].to(device)
            hg = batch["HG"].to(device)
            ag = batch["AG"].to(device)
            
            optimizer.zero_grad()
            lam_h, lam_a = model(idx[:, 0], idx[:, 1], idx[:, 2])
            loss = poisson_nll_loss(lam_h, lam_a, hg, ag)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        
        test_metrics = compute_metrics(model, test_loader, device)
        scheduler.step(test_metrics["loss"])
        
        if test_metrics["loss"] < best_test_loss - 1e-4:
            best_test_loss = test_metrics["loss"]
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"[combined] Ep {epoch:03d}/{config.epochs} | "
                  f"Test {test_metrics['loss']:.4f} | "
                  f"λh {test_metrics['mean_home_rate']:.2f} λa {test_metrics['mean_away_rate']:.2f}")
        
        if epochs_no_improve >= config.early_stopping_patience:
            print(f"  ⏹ Early stop at epoch {epoch}")
            break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    meta = {
        "league_vocab": league_vocab,
        "season_vocab": season_vocab,
        "matchup_vocab": matchup_vocab,
        "league_dim": config.league_dim,
        "season_dim": config.season_dim,
        "matchup_dim": config.matchup_dim,
        "best_epoch": best_epoch,
        "best_test_loss": best_test_loss,
    }
    return model, meta


def export_combined_embeddings(frame, model, meta, prefix="context"):
    """Export combined embeddings from the trained model."""
    out = frame.copy()
    
    league_vocab = meta["league_vocab"]
    season_vocab = meta["season_vocab"]
    matchup_vocab = meta["matchup_vocab"]
    
    out["league_idx"] = encode_with_vocab(out["league"], league_vocab)
    out["season_idx"] = encode_with_vocab(out["season"], season_vocab)
    out["matchup_idx"] = encode_with_vocab(out["matchup"], matchup_vocab)
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        league_emb = model.league_emb(
            torch.tensor(out["league_idx"].values, dtype=torch.long, device=device)
        ).cpu().numpy()
        season_emb = model.season_emb(
            torch.tensor(out["season_idx"].values, dtype=torch.long, device=device)
        ).cpu().numpy()
        matchup_emb = model.matchup_emb(
            torch.tensor(out["matchup_idx"].values, dtype=torch.long, device=device)
        ).cpu().numpy()
    
    # Create separate DataFrames for each context type
    league_cols = [f"league_emb_{i+1}" for i in range(league_emb.shape[1])]
    season_cols = [f"season_emb_{i+1}" for i in range(season_emb.shape[1])]
    matchup_cols = [f"matchup_emb_{i+1}" for i in range(matchup_emb.shape[1])]
    
    league_df = pd.DataFrame(league_emb, columns=league_cols)
    season_df = pd.DataFrame(season_emb, columns=season_cols)
    matchup_df = pd.DataFrame(matchup_emb, columns=matchup_cols)
    
    final = pd.concat([
        out[["game_id", "HG", "AG"]].reset_index(drop=True),
        league_df.reset_index(drop=True),
        season_df.reset_index(drop=True),
        matchup_df.reset_index(drop=True),
    ], axis=1)
    
    final = final.dropna(subset=["HG", "AG"]).drop_duplicates(subset=["game_id"], keep="first")
    final = final.sort_values("game_id").reset_index(drop=True)
    return final


def parse_args():
    parser = argparse.ArgumentParser(description="Create league, season, and matchup embeddings.")
    parser.add_argument("--league-dim", type=int, default=16)
    parser.add_argument("--season-dim", type=int, default=16)
    parser.add_argument("--matchup-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-matchup-count", type=int, default=2)
    args = parser.parse_args()
    return Config(
        league_dim=args.league_dim,
        season_dim=args.season_dim,
        matchup_dim=args.matchup_dim,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        seed=args.seed,
        min_matchup_count=args.min_matchup_count,
    )


def main():
    config = parse_args()
    set_seed(config.seed)

    FINAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MISC_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_keys = load_split_keys(TRAIN_SPLIT_SOURCE)
    test_keys = load_split_keys(TEST_SPLIT_SOURCE)
    targets = load_targets(TARGETS_PATH)

    train_df = train_keys.merge(targets, on="game_id", how="left").dropna(subset=["HG", "AG"]).copy()
    test_df = test_keys.merge(targets, on="game_id", how="left").dropna(subset=["HG", "AG"]).copy()

    train_df = add_context(train_df)
    test_df = add_context(test_df)

    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")
    print(f"Unique matchups (train): {train_df['matchup'].nunique()}")

    # Train combined model (all three contexts together)
    model, meta = train_combined_context_model(train_df, test_df, config)

    # Export embeddings
    train_out = export_combined_embeddings(train_df, model, meta)
    test_out = export_combined_embeddings(test_df, model, meta)

    # Save separate files for each context type
    league_cols = [c for c in train_out.columns if c.startswith("league_emb_")]
    season_cols = [c for c in train_out.columns if c.startswith("season_emb_")]
    matchup_cols = [c for c in train_out.columns if c.startswith("matchup_emb_")]
    
    base_cols = ["game_id", "HG", "AG"]
    
    train_out[base_cols + league_cols].to_csv(LEAGUE_TRAIN_OUTPUT_PATH, index=False)
    test_out[base_cols + league_cols].to_csv(LEAGUE_TEST_OUTPUT_PATH, index=False)
    
    train_out[base_cols + season_cols].to_csv(SEASON_TRAIN_OUTPUT_PATH, index=False)
    test_out[base_cols + season_cols].to_csv(SEASON_TEST_OUTPUT_PATH, index=False)
    
    train_out[base_cols + matchup_cols].to_csv(MATCHUP_TRAIN_OUTPUT_PATH, index=False)
    test_out[base_cols + matchup_cols].to_csv(MATCHUP_TEST_OUTPUT_PATH, index=False)

    # Save model
    torch.save({"state_dict": model.state_dict(), "meta": meta}, MATCHUP_MODEL_OUTPUT_PATH)
    
    # Save metadata
    META_OUTPUT_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nSaved LEAGUE TRAIN: {train_out[base_cols + league_cols].shape}")
    print(f"Saved SEASON TRAIN: {train_out[base_cols + season_cols].shape}")
    print(f"Saved MATCHUP TRAIN: {train_out[base_cols + matchup_cols].shape}")
    print(f"Saved MATCHUP TEST: {test_out[base_cols + matchup_cols].shape}")


if __name__ == "__main__":
    main()