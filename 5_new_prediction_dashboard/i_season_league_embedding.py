#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compute league / season / matchup embeddings for UPCOMING fixtures (dashboard).

Reuses the trained combined context model and its vocabularies. Never retrains,
never rebuilds vocabularies.

Reads:
- 5_new_prediction_dashboard/upcoming_lineups/upcoming_lineups_latest.csv
- result_data/matchup_embedding_model.pt   (contains both state_dict + meta)
- src/game_id_context.py (for infer_league_and_season)

Writes:
- prediction_data/upcoming_league_embeddings.csv
- prediction_data/upcoming_season_embeddings.csv
- prediction_data/upcoming_matchup_embeddings.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# PATHS
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.game_id_context import infer_league_and_season  # noqa: E402

UPCOMING_LINEUPS_PATH = SCRIPT_DIR / "upcoming_lineups" / "upcoming_lineups_latest.csv"
MODEL_PATH = PROJECT_DIR / "result_data" / "matchup_embedding_model.pt"
PREDICTION_DATA_DIR = SCRIPT_DIR / "prediction_data"

LEAGUE_OUTPUT_PATH = PREDICTION_DATA_DIR / "upcoming_league_embeddings.csv"
SEASON_OUTPUT_PATH = PREDICTION_DATA_DIR / "upcoming_season_embeddings.csv"
MATCHUP_OUTPUT_PATH = PREDICTION_DATA_DIR / "upcoming_matchup_embeddings.csv"

PREDICTION_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# MODEL DEFINITION (identical to training)
# ============================================================================

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


# ============================================================================
# HELPERS
# ============================================================================

def add_context(frame):
    """Compute league / season / matchup from game_id, same as training."""
    ctx = frame["game_id"].astype(str).apply(infer_league_and_season).apply(pd.Series)
    out = frame.merge(ctx[["game_id", "league", "season"]], on="game_id", how="left")
    out["league"] = out["league"].fillna("__UNK__")
    out["season"] = out["season"].fillna("__UNK__")

    def extract_matchup(game_id):
        parts = str(game_id).split("_")
        if len(parts) >= 3:
            return f"{parts[1]}__{parts[2]}"
        return "__UNK__"

    out["matchup"] = out["game_id"].apply(extract_matchup)
    return out


def encode_with_vocab(values, vocab):
    """Map raw strings to vocab indices. Unknown → 0 (__UNK__)."""
    return values.astype(str).map(lambda x: vocab.get(x, 0)).astype(int).values


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 78)
    print("DASHBOARD: League / Season / Matchup embeddings for upcoming fixtures")
    print("=" * 78)

    # ------------------------------------------------------------------
    # Load trained model + metadata
    # ------------------------------------------------------------------
    if not MODEL_PATH.exists():
        print(f"Error: trained context model not found at {MODEL_PATH}")
        return

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    meta = checkpoint.get("meta", None)
    if meta is None:
        print(f"Error: {MODEL_PATH.name} is missing the 'meta' payload "
              f"(vocabularies + dims are required).")
        return

    league_vocab = meta["league_vocab"]
    season_vocab = meta["season_vocab"]
    matchup_vocab = meta["matchup_vocab"]

    league_dim = meta["league_dim"]
    season_dim = meta["season_dim"]
    matchup_dim = meta["matchup_dim"]

    n_leagues = max(league_vocab.values()) + 1
    n_seasons = max(season_vocab.values()) + 1
    n_matchups = max(matchup_vocab.values()) + 1

    print(f"Loaded trained context model:")
    print(f"  Leagues:  {n_leagues} (dim {league_dim})")
    print(f"  Seasons:  {n_seasons} (dim {season_dim})")
    print(f"  Matchups: {n_matchups} (dim {matchup_dim})")

    # Determine hidden_dim from state_dict (safer than hardcoding)
    hidden_dim = state_dict["mlp.0.weight"].shape[0]

    model = CombinedContextRegressor(
        n_leagues, league_dim,
        n_seasons, season_dim,
        n_matchups, matchup_dim,
        hidden_dim,
    )
    model.load_state_dict(state_dict)
    model.eval()

    # ------------------------------------------------------------------
    # Load upcoming fixtures
    # ------------------------------------------------------------------
    if not UPCOMING_LINEUPS_PATH.exists():
        print(f"\nError: upcoming lineups not found at {UPCOMING_LINEUPS_PATH}")
        return

    upcoming = pd.read_csv(UPCOMING_LINEUPS_PATH, low_memory=False)
    print(f"\nLoaded upcoming fixtures: {len(upcoming)} rows")

    if "game_id" not in upcoming.columns:
        upcoming = upcoming.copy()
        upcoming["game_id"] = (
            upcoming["Date"].astype(str).str.replace("-", "", regex=False)
            + "_" + upcoming["Home Team"].astype(str)
            + "_" + upcoming["Away Team"].astype(str)
        )

    # ------------------------------------------------------------------
    # Attach league/season/matchup context
    # ------------------------------------------------------------------
    upcoming = add_context(upcoming)

    # Report known vs unknown context categories
    unknown_leagues = set(upcoming.loc[~upcoming["league"].isin(league_vocab), "league"])
    unknown_seasons = set(upcoming.loc[~upcoming["season"].isin(season_vocab), "season"])
    unknown_matchups = set(upcoming.loc[~upcoming["matchup"].isin(matchup_vocab), "matchup"])

    if unknown_leagues:
        print(f"\n⚠ {len(unknown_leagues)} league(s) not in training vocab:")
        for l in sorted(unknown_leagues):
            print(f"    - {l}")
    if unknown_seasons:
        print(f"\n⚠ {len(unknown_seasons)} season(s) not in training vocab:")
        for s in sorted(unknown_seasons):
            print(f"    - {s}")
    if unknown_matchups:
        print(f"\n⚠ {len(unknown_matchups)} matchup(s) not in training vocab "
              f"(→ will map to __UNK__):")
        for m in sorted(unknown_matchups)[:15]:
            print(f"    - {m}")
        if len(unknown_matchups) > 15:
            print(f"    ... and {len(unknown_matchups) - 15} more")

    # ------------------------------------------------------------------
    # Encode indices
    # ------------------------------------------------------------------
    league_idx = encode_with_vocab(upcoming["league"], league_vocab)
    season_idx = encode_with_vocab(upcoming["season"], season_vocab)
    matchup_idx = encode_with_vocab(upcoming["matchup"], matchup_vocab)

    league_t = torch.tensor(league_idx, dtype=torch.long)
    season_t = torch.tensor(season_idx, dtype=torch.long)
    matchup_t = torch.tensor(matchup_idx, dtype=torch.long)

    # ------------------------------------------------------------------
    # Extract embeddings (no goal-rate prediction needed — we only want the
    # embedding vectors, matching what training exported)
    # ------------------------------------------------------------------
    with torch.no_grad():
        league_emb = model.league_emb(league_t).numpy()
        season_emb = model.season_emb(season_t).numpy()
        matchup_emb = model.matchup_emb(matchup_t).numpy()

    base_cols = {
        "game_id": upcoming["game_id"].astype(str).values,
        "Date": upcoming["Date"].values,
        "Home Team": upcoming["Home Team"].values,
        "Away Team": upcoming["Away Team"].values,
    }

    league_cols = [f"league_emb_{i+1}" for i in range(league_dim)]
    season_cols = [f"season_emb_{i+1}" for i in range(season_dim)]
    matchup_cols = [f"matchup_emb_{i+1}" for i in range(matchup_dim)]

    league_df = pd.DataFrame({**base_cols, **dict(zip(league_cols, league_emb.T))})
    season_df = pd.DataFrame({**base_cols, **dict(zip(season_cols, season_emb.T))})
    matchup_df = pd.DataFrame({**base_cols, **dict(zip(matchup_cols, matchup_emb.T))})

    # ------------------------------------------------------------------
    # Save (three separate files, matching training output naming)
    # ==================================================================
    league_df.to_csv(LEAGUE_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    season_df.to_csv(SEASON_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    matchup_df.to_csv(MATCHUP_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Fixtures encoded: {len(upcoming)}")
    print(f"League embeddings:  {league_df.shape}  → {LEAGUE_OUTPUT_PATH}")
    print(f"Season embeddings:  {season_df.shape}  → {SEASON_OUTPUT_PATH}")
    print(f"Matchup embeddings: {matchup_df.shape}  → {MATCHUP_OUTPUT_PATH}")


if __name__ == "__main__":
    main()