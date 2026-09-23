#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train league, season, and team attack/defence embeddings.

Replaces the previous matchup-embedding approach (which exploded
combinatorially in the number of team pairs) with a Dixon-Coles-style
parameterization:

    log(lambda_home) = mu_home + head_home( home_off, league_emb, season_emb )
    log(lambda_away) = mu_away + head_away( away_off, league_emb, season_emb )

where
    home_off = attack(home_team) - defence(away_team)
    away_off = attack(away_team) - defence(home_team)

This reduces the number of learned matchup-specific parameters from
O(n_teams^2) to O(n_teams), which is the difference between an
unlearnable model and one that can extract real signal from ~20k matches.

Outputs (same file names as before):
    league_embeddings_{TRAIN,TEST}.csv    (league_emb_1..K)
    season_embeddings_{TRAIN,TEST}.csv    (season_emb_1..K)
    matchup_embeddings_{TRAIN,TEST}.csv   (matchup_emb_1..4K, team embeddings)
Each file also carries game_id, HG, AG.
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
GOALS_SOURCE_PATH = DATA_DIR / "big5_matches_fixed.csv"

LEAGUE_TRAIN_OUTPUT_PATH = FINAL_DATA_DIR / "league_embeddings_TRAIN.csv"
LEAGUE_TEST_OUTPUT_PATH = FINAL_DATA_DIR / "league_embeddings_TEST.csv"
SEASON_TRAIN_OUTPUT_PATH = FINAL_DATA_DIR / "season_embeddings_TRAIN.csv"
SEASON_TEST_OUTPUT_PATH = FINAL_DATA_DIR / "season_embeddings_TEST.csv"
MATCHUP_TRAIN_OUTPUT_PATH = FINAL_DATA_DIR / "matchup_embeddings_TRAIN.csv"
MATCHUP_TEST_OUTPUT_PATH = FINAL_DATA_DIR / "matchup_embeddings_TEST.csv"

META_OUTPUT_PATH = MISC_DIR / "context_embedding_metadata.json"
MODEL_OUTPUT_PATH = RESULT_DATA_DIR / "context_embedding_model.pt"


@dataclass
class Config:
    team_dim: int = 4
    league_dim: int = 16
    season_dim: int = 16
    head_hidden: int = 32
    epochs: int = 200
    batch_size: int = 512
    lr: float = 3e-3
    embedding_wd: float = 1e-2
    head_wd: float = 1e-4
    dropout: float = 0.1
    patience: int = 20
    val_frac: float = 0.1
    grad_clip: float = 5.0
    seed: int = 42


# ============================================================================
# DATA LOADING
# ============================================================================

def load_split_keys(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing split source: {path}")
    frame = pd.read_csv(path, low_memory=False)
    if "game_id" not in frame.columns:
        raise KeyError(f"game_id missing in split source: {path}")
    return frame[["game_id"]].dropna().drop_duplicates().copy()


def load_targets(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing goals source file: {path}")
    frame = pd.read_csv(path, low_memory=False)
    home_candidates = ["HG", "FTHG", "Home_goals", "home_goals"]
    away_candidates = ["AG", "FTAG", "Away_goals", "away_goals"]
    hg_col = next((c for c in home_candidates if c in frame.columns), None)
    ag_col = next((c for c in away_candidates if c in frame.columns), None)
    if "game_id" not in frame.columns or hg_col is None or ag_col is None:
        raise KeyError(
            f"Could not locate game_id + goal columns in {path}. "
            f"Available: {list(frame.columns)}"
        )
    out = frame[["game_id", hg_col, ag_col]].rename(
        columns={hg_col: "HG", ag_col: "AG"}
    ).copy()
    out["HG"] = pd.to_numeric(out["HG"], errors="coerce")
    out["AG"] = pd.to_numeric(out["AG"], errors="coerce")
    out = out.dropna(subset=["game_id", "HG", "AG"]).drop_duplicates(
        subset=["game_id"], keep="first"
    )
    return out


def parse_teams(game_id: str):
    """Extract (home_team, away_team) from '<date>_<home>_<away>'."""
    parts = str(game_id).split("_")
    if len(parts) >= 3:
        return parts[1], parts[2]
    return "__UNK__", "__UNK__"


def add_context(frame: pd.DataFrame) -> pd.DataFrame:
    # League and season via the existing helper (game_id -> league, season)
    ctx = (
        frame["game_id"].astype(str)
        .apply(infer_league_and_season)
        .apply(pd.Series)
        .reset_index(drop=True)
    )
    out = frame.copy().reset_index(drop=True)
    out["league"] = ctx["league"].fillna("__UNK__").values if "league" in ctx.columns else "__UNK__"
    out["season"] = ctx["season"].fillna("__UNK__").values if "season" in ctx.columns else "__UNK__"

    teams = out["game_id"].apply(parse_teams).apply(pd.Series)
    out["home_team"] = teams[0].fillna("__UNK__").values
    out["away_team"] = teams[1].fillna("__UNK__").values
    return out


def build_vocab(values, min_count: int = 1) -> dict:
    vc = values.astype(str).value_counts()
    valid = vc[vc >= min_count].index.tolist()
    vocab = {"__UNK__": 0}
    for i, v in enumerate(sorted(valid), start=1):
        if v == "__UNK__":
            continue
        vocab[v] = i
    return vocab


def encode(values, vocab: dict) -> np.ndarray:
    return values.astype(str).map(lambda x: vocab.get(x, 0)).astype(int).values


# ============================================================================
# MODEL: Dixon-Coles attack/defence + league and season offsets
# ============================================================================

class ContextDixonColes(nn.Module):
    """Team attack/defence embeddings + league and season offsets.

    The rate for each side is:
        log(lambda_home) = log(home_prior) + head_home(home_off, league, season)
        log(lambda_away) = log(away_prior) + head_away(away_off, league, season)
    where
        home_off = attack(home) - defence(away)
        away_off = attack(away) - defence(home)

    Head output layers are zero-initialized, so the model starts exactly at
    the priors. This avoids the well-known pathology where a randomly
    initialized head produces rates far from the data distribution, forcing
    the optimizer to first undo an offset before learning anything useful.
    """

    def __init__(self, n_teams, n_leagues, n_seasons, config: Config,
                 home_prior=1.4, away_prior=1.15):
        super().__init__()
        self.attack = nn.Embedding(n_teams, config.team_dim)
        self.defence = nn.Embedding(n_teams, config.team_dim)
        self.league = nn.Embedding(n_leagues, config.league_dim)
        self.season = nn.Embedding(n_seasons, config.season_dim)

        nn.init.normal_(self.attack.weight, std=0.05)
        nn.init.normal_(self.defence.weight, std=0.05)
        nn.init.normal_(self.league.weight, std=0.05)
        nn.init.normal_(self.season.weight, std=0.05)

        self.home_bias = nn.Parameter(torch.tensor(float(np.log(home_prior))))
        self.away_bias = nn.Parameter(torch.tensor(float(np.log(away_prior))))

        head_in = config.team_dim + config.league_dim + config.season_dim

        self.head_h = nn.Sequential(
            nn.Linear(head_in, config.head_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden, 1),
        )
        self.head_a = nn.Sequential(
            nn.Linear(head_in, config.head_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden, 1),
        )
        for h in (self.head_h, self.head_a):
            nn.init.zeros_(h[-1].weight)
            nn.init.zeros_(h[-1].bias)

    def forward(self, home_idx, away_idx, league_idx, season_idx):
        a_h = self.attack(home_idx)     # (B, team_dim)
        d_h = self.defence(home_idx)
        a_a = self.attack(away_idx)
        d_a = self.defence(away_idx)

        home_off = a_h - d_a            # home attacking strength vs. away defence
        away_off = a_a - d_h

        l_emb = self.league(league_idx)
        s_emb = self.season(season_idx)

        home_in = torch.cat([home_off, l_emb, s_emb], dim=-1)
        away_in = torch.cat([away_off, l_emb, s_emb], dim=-1)

        log_lam_h = self.home_bias + self.head_h(home_in).squeeze(-1)
        log_lam_a = self.away_bias + self.head_a(away_in).squeeze(-1)
        return torch.exp(log_lam_h), torch.exp(log_lam_a)


# ============================================================================
# DATASET
# ============================================================================

class MatchDataset(Dataset):
    def __init__(self, df: pd.DataFrame, key_cols):
        self.idx = {c: torch.tensor(df[c].values, dtype=torch.long) for c in key_cols}
        self.hg = torch.tensor(df["HG"].values, dtype=torch.float32)
        self.ag = torch.tensor(df["AG"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.hg)

    def __getitem__(self, i):
        out = {k: v[i] for k, v in self.idx.items()}
        out["HG"] = self.hg[i]
        out["AG"] = self.ag[i]
        return out


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ============================================================================
# TRAINING
# ============================================================================

def poisson_nll(lh, la, hg, ag):
    return (F.poisson_nll_loss(lh, hg, log_input=False, full=True)
            + F.poisson_nll_loss(la, ag, log_input=False, full=True))


def eval_loss(model, loader, device) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            lh, la = model(b["home_idx"], b["away_idx"],
                            b["league_idx"], b["season_idx"])
            total += float(poisson_nll(lh, la, b["HG"], b["AG"]).mean().item())
            n += 1
    return total / max(1, n)


def train_model(model, train_loader, val_loader, config: Config, device):
    emb_params = (list(model.attack.parameters())
                  + list(model.defence.parameters())
                  + list(model.league.parameters())
                  + list(model.season.parameters()))
    head_params = (list(model.head_h.parameters())
                   + list(model.head_a.parameters())
                   + [model.home_bias, model.away_bias])

    optimizer = torch.optim.AdamW([
        {"params": emb_params, "weight_decay": config.embedding_wd},
        {"params": head_params, "weight_decay": config.head_wd},
    ], lr=config.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5
    )

    best_val = float("inf")
    best_state = None
    no_improve = 0

    print("\n" + "=" * 70)
    print("TRAINING CONTEXT MODEL (Dixon-Coles attack/defence + league + season)")
    print("=" * 70)
    print(f"  n_teams={len(model.attack.weight)}  n_leagues={len(model.league.weight)}  "
          f"n_seasons={len(model.season.weight)}")
    print(f"  team_dim={model.attack.embedding_dim}  "
          f"league_dim={model.league.embedding_dim}  "
          f"season_dim={model.season.embedding_dim}")

    for epoch in range(1, config.epochs + 1):
        model.train()
        tr_losses = []
        for b in train_loader:
            b = {k: v.to(device) for k, v in b.items()}
            optimizer.zero_grad()
            lh, la = model(b["home_idx"], b["away_idx"],
                            b["league_idx"], b["season_idx"])
            loss = poisson_nll(lh, la, b["HG"], b["AG"]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            tr_losses.append(float(loss.item()))

        val_loss = eval_loss(model, val_loader, device)
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Ep {epoch:03d}/{config.epochs} | "
                  f"train {np.mean(tr_losses):.4f} | val {val_loss:.4f}")

        if no_improve >= config.patience:
            print(f"  Early stop at epoch {epoch} (best val {best_val:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


# ============================================================================
# EXPORT
# ============================================================================

def export_embeddings(model, df, team_vocab, league_vocab, season_vocab, device):
    """Return (league_df, season_df, matchup_df)."""
    home_idx = torch.tensor(encode(df["home_team"], team_vocab), dtype=torch.long, device=device)
    away_idx = torch.tensor(encode(df["away_team"], team_vocab), dtype=torch.long, device=device)
    league_idx = torch.tensor(encode(df["league"], league_vocab), dtype=torch.long, device=device)
    season_idx = torch.tensor(encode(df["season"], season_vocab), dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        a_h = model.attack(home_idx).cpu().numpy()
        d_h = model.defence(home_idx).cpu().numpy()
        a_a = model.attack(away_idx).cpu().numpy()
        d_a = model.defence(away_idx).cpu().numpy()
        l_emb = model.league(league_idx).cpu().numpy()
        s_emb = model.season(season_idx).cpu().numpy()

    # Matchup features = [home_attack, home_defence, away_attack, away_defence]
    matchup = np.concatenate([a_h, d_h, a_a, d_a], axis=1)

    base = df[["game_id", "HG", "AG"]].reset_index(drop=True)

    def make(feat, prefix):
        cols = [f"{prefix}_{i+1}" for i in range(feat.shape[1])]
        return pd.concat([base, pd.DataFrame(feat, columns=cols)], axis=1)

    return (make(l_emb, "league_emb"),
            make(s_emb, "season_emb"),
            make(matchup, "matchup_emb"))


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team-dim", type=int, default=4)
    parser.add_argument("--league-dim", type=int, default=16)
    parser.add_argument("--season-dim", type=int, default=16)
    parser.add_argument("--head-hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--embedding-wd", type=float, default=1e-2)
    parser.add_argument("--head-wd", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = Config(
        team_dim=args.team_dim, league_dim=args.league_dim, season_dim=args.season_dim,
        head_hidden=args.head_hidden, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, embedding_wd=args.embedding_wd, head_wd=args.head_wd,
        dropout=args.dropout, patience=args.patience, seed=args.seed,
    )

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    FINAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MISC_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load splits and goals ---
    train_keys = load_split_keys(TRAIN_SPLIT_SOURCE)
    test_keys = load_split_keys(TEST_SPLIT_SOURCE)
    targets = load_targets(GOALS_SOURCE_PATH)
    print(f"Loaded {len(targets)} match results from {GOALS_SOURCE_PATH.name}")

    train_full = train_keys.merge(targets, on="game_id", how="left").dropna(subset=["HG", "AG"]).copy()
    test_df = test_keys.merge(targets, on="game_id", how="left").dropna(subset=["HG", "AG"]).copy()

    if train_full.empty or test_df.empty:
        raise ValueError(f"Empty split. Train={len(train_full)}, Test={len(test_df)}")

    train_full = add_context(train_full)
    test_df = add_context(test_df)
    print(f"Train rows: {len(train_full)} | Test rows: {len(test_df)}")

    # --- Build vocabularies from training data only ---
    all_train_teams = pd.concat([train_full["home_team"], train_full["away_team"]])
    team_vocab = build_vocab(all_train_teams, min_count=2)
    league_vocab = build_vocab(train_full["league"], min_count=1)
    season_vocab = build_vocab(train_full["season"], min_count=1)

    n_teams = max(team_vocab.values()) + 1
    n_leagues = max(league_vocab.values()) + 1
    n_seasons = max(season_vocab.values()) + 1
    print(f"Vocabs: teams={n_teams} leagues={n_leagues} seasons={n_seasons}")

    # --- Encode on both splits (unseen values -> __UNK__ = index 0) ---
    for df in (train_full, test_df):
        df["home_idx"] = encode(df["home_team"], team_vocab)
        df["away_idx"] = encode(df["away_team"], team_vocab)
        df["league_idx"] = encode(df["league"], league_vocab)
        df["season_idx"] = encode(df["season"], season_vocab)

    # --- Train/val split of training data ---
    rng = np.random.default_rng(config.seed)
    perm = rng.permutation(len(train_full))
    n_val = int(len(train_full) * config.val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_df = train_full.iloc[train_idx].reset_index(drop=True)
    val_df = train_full.iloc[val_idx].reset_index(drop=True)
    print(f"Split: train={len(train_df)}  val={len(val_df)}")

    key_cols = ["home_idx", "away_idx", "league_idx", "season_idx"]
    train_loader = DataLoader(MatchDataset(train_df, key_cols),
                              batch_size=config.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(MatchDataset(val_df, key_cols),
                            batch_size=config.batch_size, shuffle=False, collate_fn=collate)

    # --- Model ---
    home_prior = float(train_df["HG"].mean())
    away_prior = float(train_df["AG"].mean())
    model = ContextDixonColes(n_teams, n_leagues, n_seasons, config,
                              home_prior=home_prior, away_prior=away_prior).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}  (priors: home={home_prior:.3f}, away={away_prior:.3f})")

    model, best_val = train_model(model, train_loader, val_loader, config, device)

    # --- Test evaluation ---
    test_loader = DataLoader(MatchDataset(test_df, key_cols),
                             batch_size=config.batch_size, shuffle=False, collate_fn=collate)
    test_nll = eval_loss(model, test_loader, device)

    naive_nll = float(poisson_nll(
        torch.full((len(test_df),), float(test_df["HG"].mean())),
        torch.full((len(test_df),), float(test_df["AG"].mean())),
        torch.tensor(test_df["HG"].values, dtype=torch.float32),
        torch.tensor(test_df["AG"].values, dtype=torch.float32),
    ).mean().item())

    print(f"\nBest val NLL: {best_val:.4f}")
    print(f"Test  NLL:    {test_nll:.4f}  (naive: {naive_nll:.4f}, "
          f"delta: {test_nll - naive_nll:+.4f})")

    # --- Spread diagnostics ---
    model.eval()
    with torch.no_grad():
        th = torch.tensor(test_df["home_idx"].values, dtype=torch.long, device=device)
        ta = torch.tensor(test_df["away_idx"].values, dtype=torch.long, device=device)
        tl = torch.tensor(test_df["league_idx"].values, dtype=torch.long, device=device)
        tsea = torch.tensor(test_df["season_idx"].values, dtype=torch.long, device=device)
        lh, la = model(th, ta, tl, tsea)
        lh, la = lh.cpu().numpy(), la.cpu().numpy()
    print(f"Test predicted lambda_home: mean={lh.mean():.3f} std={lh.std():.3f}")
    print(f"Test predicted lambda_away: mean={la.mean():.3f} std={la.std():.3f}")

    # --- Export embeddings ---
    league_train, season_train, matchup_train = export_embeddings(
        model, train_full, team_vocab, league_vocab, season_vocab, device
    )
    league_test, season_test, matchup_test = export_embeddings(
        model, test_df, team_vocab, league_vocab, season_vocab, device
    )

    league_train.to_csv(LEAGUE_TRAIN_OUTPUT_PATH, index=False)
    league_test.to_csv(LEAGUE_TEST_OUTPUT_PATH, index=False)
    season_train.to_csv(SEASON_TRAIN_OUTPUT_PATH, index=False)
    season_test.to_csv(SEASON_TEST_OUTPUT_PATH, index=False)
    matchup_train.to_csv(MATCHUP_TRAIN_OUTPUT_PATH, index=False)
    matchup_test.to_csv(MATCHUP_TEST_OUTPUT_PATH, index=False)

    meta = {
        "team_vocab": team_vocab,
        "league_vocab": league_vocab,
        "season_vocab": season_vocab,
        "team_dim": config.team_dim,
        "league_dim": config.league_dim,
        "season_dim": config.season_dim,
        "matchup_dim": 4 * config.team_dim,
        "best_val_nll": best_val,
        "test_nll": test_nll,
        "naive_test_nll": naive_nll,
        "test_std_lambda_home": float(lh.std()),
        "test_std_lambda_away": float(la.std()),
        "config": config.__dict__,
    }
    torch.save({"state_dict": model.state_dict(), "meta": meta}, MODEL_OUTPUT_PATH)
    META_OUTPUT_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nSaved league:  train {league_train.shape}, test {league_test.shape}")
    print(f"Saved season:  train {season_train.shape}, test {season_test.shape}")
    print(f"Saved matchup: train {matchup_train.shape}, test {matchup_test.shape}")
    print(f"Saved model:   {MODEL_OUTPUT_PATH}")
    print(f"Saved meta:    {META_OUTPUT_PATH}")


if __name__ == "__main__":
    main()