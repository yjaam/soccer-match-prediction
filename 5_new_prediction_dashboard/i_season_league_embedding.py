#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compute league / season / matchup embeddings for UPCOMING fixtures (dashboard).

Uses the OPTIMIZED Dixon-Coles style context model:
- Team attack/defence embeddings instead of a per-pair matchup embedding
- League and season share the head with the team offsets
- A match's "matchup" embedding = [home_attack, home_defence,
                                   away_attack, away_defence]
  i.e. 4 * team_dim columns, whereas league/season embeddings come from
  their own embedding tables.

Reuses the trained model and its vocabularies. Never retrains.

Reads:
- 5_new_prediction_dashboard/upcoming_lineups/upcoming_lineups_latest.csv
- result_data/context_embedding_model.pt   (state_dict + meta)
- src/game_id_context.py (for infer_league_and_season)

Writes:
- prediction_data/upcoming_league_embeddings.csv
- prediction_data/upcoming_season_embeddings.csv
- prediction_data/upcoming_matchup_embeddings.csv

IMPORTANT: result_data/context_embedding_model.pt must have been produced
by the matching OPTIMIZED training script. If it was produced by the older
per-pair matchup model, this script will fail with a clear error.
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
MODEL_PATH = PROJECT_DIR / "result_data" / "context_embedding_model.pt"
PREDICTION_DATA_DIR = SCRIPT_DIR / "prediction_data"

LEAGUE_OUTPUT_PATH = PREDICTION_DATA_DIR / "upcoming_league_embeddings.csv"
SEASON_OUTPUT_PATH = PREDICTION_DATA_DIR / "upcoming_season_embeddings.csv"
MATCHUP_OUTPUT_PATH = PREDICTION_DATA_DIR / "upcoming_matchup_embeddings.csv"

PREDICTION_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# MODEL DEFINITION (must exactly mirror the OPTIMIZED training script)
# ============================================================================

class ContextDixonColes(nn.Module):
    """Team attack/defence embeddings + league and season offsets.

        log(lambda_home) = log(home_prior) + head_h(home_off, league, season)
        log(lambda_away) = log(away_prior) + head_a(away_off, league, season)

    where
        home_off = attack(home) - defence(away)
        away_off = attack(away) - defence(home)

    Only the embedding tables are needed for the dashboard; the heads and
    priors are loaded so that state_dict keys match exactly.
    """

    def __init__(self, n_teams, n_leagues, n_seasons,
                 team_dim, league_dim, season_dim, head_hidden, dropout=0.1,
                 home_prior=1.4, away_prior=1.15):
        super().__init__()
        self.attack = nn.Embedding(n_teams, team_dim)
        self.defence = nn.Embedding(n_teams, team_dim)
        self.league = nn.Embedding(n_leagues, league_dim)
        self.season = nn.Embedding(n_seasons, season_dim)

        self.home_bias = nn.Parameter(torch.tensor(float(np.log(home_prior))))
        self.away_bias = nn.Parameter(torch.tensor(float(np.log(away_prior))))

        head_in = team_dim + league_dim + season_dim

        self.head_h = nn.Sequential(
            nn.Linear(head_in, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )
        self.head_a = nn.Sequential(
            nn.Linear(head_in, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, home_idx, away_idx, league_idx, season_idx):
        a_h = self.attack(home_idx)
        d_h = self.defence(home_idx)
        a_a = self.attack(away_idx)
        d_a = self.defence(away_idx)

        home_off = a_h - d_a
        away_off = a_a - d_h

        l_emb = self.league(league_idx)
        s_emb = self.season(season_idx)

        home_in = torch.cat([home_off, l_emb, s_emb], dim=-1)
        away_in = torch.cat([away_off, l_emb, s_emb], dim=-1)

        log_lam_h = self.home_bias + self.head_h(home_in).squeeze(-1)
        log_lam_a = self.away_bias + self.head_a(away_in).squeeze(-1)
        return torch.exp(log_lam_h), torch.exp(log_lam_a)


# ============================================================================
# HELPERS
# ============================================================================

def parse_teams(game_id: str):
    """Extract (home_team, away_team) from '<date>_<home>_<away>'.

    Mirrors the training-time extraction exactly. Falls back to __UNK__.
    """
    parts = str(game_id).split("_")
    if len(parts) >= 3:
        return parts[1], parts[2]
    return "__UNK__", "__UNK__"


def add_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute league / season / home_team / away_team from game_id."""
    ctx = (
        frame["game_id"].astype(str)
        .apply(infer_league_and_season)
        .apply(pd.Series)
        .reset_index(drop=True)
    )
    out = frame.copy().reset_index(drop=True)

    if "league" in ctx.columns:
        out["league"] = ctx["league"].fillna("__UNK__").values
    else:
        out["league"] = "__UNK__"
    if "season" in ctx.columns:
        out["season"] = ctx["season"].fillna("__UNK__").values
    else:
        out["season"] = "__UNK__"

    teams = out["game_id"].apply(parse_teams).apply(pd.Series)
    out["home_team"] = teams[0].fillna("__UNK__").values
    out["away_team"] = teams[1].fillna("__UNK__").values
    return out


def encode_with_vocab(values, vocab):
    """Map raw strings to vocab indices. Unknown → 0 (__UNK__)."""
    return values.astype(str).map(lambda x: vocab.get(x, 0)).astype(int).values


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 78)
    print("DASHBOARD: League / Season / Matchup embeddings (optimized context model)")
    print("=" * 78)

    # ------------------------------------------------------------------
    # Load trained model + metadata
    # ------------------------------------------------------------------
    if not MODEL_PATH.exists():
        print(f"Error: trained context model not found at {MODEL_PATH}")
        print("       Re-run the optimized context training script to generate it.")
        return

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    meta = checkpoint.get("meta", None)
    if meta is None:
        print(f"Error: {MODEL_PATH.name} is missing the 'meta' payload "
              f"(vocabularies + dims are required).")
        return

    required = ["team_vocab", "league_vocab", "season_vocab",
                "team_dim", "league_dim", "season_dim", "matchup_dim"]
    missing = [k for k in required if k not in meta]
    if missing:
        raise RuntimeError(
            f"Checkpoint meta is missing required keys: {missing}\n"
            f"  This usually means {MODEL_PATH.name} was produced by an "
            f"older training script.\n"
            f"  Re-run the optimized context training script to regenerate it."
        )

    team_vocab = meta["team_vocab"]
    league_vocab = meta["league_vocab"]
    season_vocab = meta["season_vocab"]

    team_dim = meta["team_dim"]
    league_dim = meta["league_dim"]
    season_dim = meta["season_dim"]
    matchup_dim = meta["matchup_dim"]  # = 4 * team_dim

    n_teams = max(team_vocab.values()) + 1
    n_leagues = max(league_vocab.values()) + 1
    n_seasons = max(season_vocab.values()) + 1

    # Infer head_hidden from the saved state dict
    #   head_h.0.weight has shape (head_hidden, team_dim + league_dim + season_dim)
    head_hidden = state_dict["head_h.0.weight"].shape[0]

    print(f"Loaded trained context model:")
    print(f"  Teams:    {n_teams} (dim {team_dim} each for attack and defence)")
    print(f"  Leagues:  {n_leagues} (dim {league_dim})")
    print(f"  Seasons:  {n_seasons} (dim {season_dim})")
    print(f"  Matchup:  {matchup_dim} dims (= 4 x team_dim)")
    print(f"  Head hidden: {head_hidden}")

    model = ContextDixonColes(
        n_teams=n_teams, n_leagues=n_leagues, n_seasons=n_seasons,
        team_dim=team_dim, league_dim=league_dim, season_dim=season_dim,
        head_hidden=head_hidden,
    )

    # Strict load: any architecture mismatch gives an explicit error.
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            "Context model architecture does not match the checkpoint.\n"
            "  This usually means the checkpoint was produced by an older "
            "training script (e.g. the per-pair matchup model).\n"
            "  Re-run the optimized training script to regenerate "
            "context_embedding_model.pt.\n"
            f"  PyTorch detail: {e}"
        ) from e

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
    # Attach league / season / teams
    # ------------------------------------------------------------------
    upcoming = add_context(upcoming)

    # Report out-of-vocabulary categories so the user knows if embeddings
    # will fall back to __UNK__.
    unknown_leagues = sorted(set(upcoming.loc[~upcoming["league"].isin(league_vocab), "league"]))
    unknown_seasons = sorted(set(upcoming.loc[~upcoming["season"].isin(season_vocab), "season"]))
    unknown_teams = sorted(
        set(upcoming.loc[~upcoming["home_team"].isin(team_vocab), "home_team"])
        | set(upcoming.loc[~upcoming["away_team"].isin(team_vocab), "away_team"])
    )

    if unknown_leagues:
        print(f"\n⚠ {len(unknown_leagues)} league(s) not in training vocab (→ __UNK__):")
        for l in unknown_leagues:
            print(f"    - {l}")
    if unknown_seasons:
        print(f"\n⚠ {len(unknown_seasons)} season(s) not in training vocab (→ __UNK__):")
        for s in unknown_seasons:
            print(f"    - {s}")
    if unknown_teams:
        print(f"\n⚠ {len(unknown_teams)} team(s) not in training vocab (→ __UNK__):")
        for t in unknown_teams[:15]:
            print(f"    - {t}")
        if len(unknown_teams) > 15:
            print(f"    ... and {len(unknown_teams) - 15} more")

    # ------------------------------------------------------------------
    # Encode indices
    # ------------------------------------------------------------------
    home_idx = encode_with_vocab(upcoming["home_team"], team_vocab)
    away_idx = encode_with_vocab(upcoming["away_team"], team_vocab)
    league_idx = encode_with_vocab(upcoming["league"], league_vocab)
    season_idx = encode_with_vocab(upcoming["season"], season_vocab)

    home_t = torch.tensor(home_idx, dtype=torch.long)
    away_t = torch.tensor(away_idx, dtype=torch.long)
    league_t = torch.tensor(league_idx, dtype=torch.long)
    season_t = torch.tensor(season_idx, dtype=torch.long)

    # ------------------------------------------------------------------
    # Extract embeddings
    # ------------------------------------------------------------------
    with torch.no_grad():
        league_emb = model.league(league_t).numpy()                       # (N, league_dim)
        season_emb = model.season(season_t).numpy()                       # (N, season_dim)

        home_attack = model.attack(home_t).numpy()                        # (N, team_dim)
        home_defence = model.defence(home_t).numpy()                      # (N, team_dim)
        away_attack = model.attack(away_t).numpy()                        # (N, team_dim)
        away_defence = model.defence(away_t).numpy()                      # (N, team_dim)

    # Matchup embedding = concatenation of the four team vectors
    matchup_emb = np.concatenate(
        [home_attack, home_defence, away_attack, away_defence], axis=1
    )                                                                     # (N, 4*team_dim)

    base_cols = {
        "game_id": upcoming["game_id"].astype(str).values,
        "Date": upcoming["Date"].values,
        "Home Team": upcoming["Home Team"].values,
        "Away Team": upcoming["Away Team"].values,
    }

    league_cols = [f"league_emb_{i+1}" for i in range(league_dim)]
    season_cols = [f"season_emb_{i+1}" for i in range(season_dim)]
    matchup_cols = [f"matchup_emb_{i+1}" for i in range(matchup_emb.shape[1])]

    league_df = pd.DataFrame({**base_cols, **dict(zip(league_cols, league_emb.T))})
    season_df = pd.DataFrame({**base_cols, **dict(zip(season_cols, season_emb.T))})
    matchup_df = pd.DataFrame({**base_cols, **dict(zip(matchup_cols, matchup_emb.T))})

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
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