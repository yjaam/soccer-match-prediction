#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compute lineup embeddings for UPCOMING fixtures (dashboard use).

Reuses:
- The trained 2-block transformer encoder (misc/lineup_encoder_state_dict.pt)
- The trained player vocabulary (misc/lineup_player_vocab.csv)

New lineup player names that are not in the vocabulary are resolved via
rapidfuzz against known players, falling back to <UNK> if nothing matches.

Reads:
- 5_new_prediction_dashboard/upcoming_lineups/upcoming_lineups_latest.csv
- misc/lineup_encoder_state_dict.pt  (read-only)
- misc/lineup_player_vocab.csv       (read-only)

Writes:
- 5_new_prediction_dashboard/prediction_data/upcoming_lineup_embeddings.csv
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from rapidfuzz import process, fuzz
    USING_RAPIDFUZZ = True
except ImportError:
    USING_RAPIDFUZZ = False
    print("Note: install 'rapidfuzz' for faster fuzzy matching: pip install rapidfuzz")


# ============================================================================
# PATHS
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

UPCOMING_LINEUPS_PATH = SCRIPT_DIR / "upcoming_lineups" / "upcoming_lineups_latest.csv"
STATE_DICT_PATH = PROJECT_DIR / "misc" / "lineup_encoder_state_dict.pt"
VOCAB_PATH = PROJECT_DIR / "misc" / "lineup_player_vocab.csv"
METRICS_PATH = PROJECT_DIR / "misc" / "lineup_encoder_training_metrics.json"
PREDICTION_DATA_DIR = SCRIPT_DIR / "prediction_data"
OUTPUT_PATH = PREDICTION_DATA_DIR / "upcoming_lineup_embeddings.csv"
UNMATCHED_PATH = PREDICTION_DATA_DIR / "upcoming_lineup_unmatched_players.csv"

PREDICTION_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CONFIG (must match training)
# ============================================================================

POSITION_TO_INDEX = {"GK": 0, "DEF": 1, "MID": 2, "ATT": 3}

FUZZY_MATCH_THRESHOLD = 82.0  # rapidfuzz ratio in [0, 100]


# ============================================================================
# MODEL DEFINITION (identical to training)
# ============================================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, attention_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_heads = attention_heads
        self.head_dim = hidden_dim // attention_heads
        self.attention_scale = self.head_dim ** 0.5
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.position_bias = nn.Parameter(torch.zeros(4))
        self.attention_dropout = nn.Dropout(dropout)

    def forward(self, x, position_indices, mask):
        batch_size, num_players, _ = x.shape
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        Q = Q.view(batch_size, num_players, self.attention_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_players, self.attention_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_players, self.attention_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.attention_scale
        pos_bias = self.position_bias[position_indices]
        pos_bias = pos_bias.unsqueeze(1).unsqueeze(2)
        attn_scores = attn_scores + pos_bias
        mask_expanded = mask.unsqueeze(1).unsqueeze(2)
        attn_scores = attn_scores.masked_fill(mask_expanded == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        attended = torch.matmul(attn_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, num_players, self.hidden_dim)
        return self.out_proj(attended)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, attention_heads, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_dim, attention_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, position_indices, mask):
        attn_out = self.attention(x, position_indices, mask)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim, attention_heads=4, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_dim, attention_heads, dropout)
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

    def forward(self, x, position_indices, mask):
        batch_size, num_players, _ = x.shape
        pool_query = self.pool_query.expand(batch_size, -1, -1)
        x_with_query = torch.cat([pool_query, x], dim=1)
        query_mask = torch.ones(batch_size, 1, device=mask.device)
        extended_mask = torch.cat([query_mask, mask], dim=1)
        query_pos = torch.zeros(batch_size, 1, dtype=position_indices.dtype, device=position_indices.device)
        extended_pos = torch.cat([query_pos, position_indices], dim=1)
        attended = self.attention(x_with_query, extended_pos, extended_mask)
        return attended[:, 0, :]


class ImprovedLineupEncoder(nn.Module):
    """Exactly the architecture used during training. Must be instantiated with
    matching config so state_dict keys and shapes line up."""

    def __init__(self, config: dict, vocab_size: int):
        super().__init__()
        self.config = config
        player_embedding_dim = config["player_embedding_dim"]
        position_embedding_dim = config["position_embedding_dim"]
        hidden_dim = config["hidden_dim"]
        output_dim = config["output_dim"]
        attention_heads = config["attention_heads"]
        num_blocks = config["num_transformer_blocks"]
        dropout = config["dropout"]

        self.player_embedding = nn.Embedding(vocab_size, player_embedding_dim)
        self.position_embedding = nn.Embedding(len(POSITION_TO_INDEX), position_embedding_dim)
        input_dim = player_embedding_dim + position_embedding_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, attention_heads, dropout)
            for _ in range(num_blocks)
        ])
        self.pooling = AttentionPooling(hidden_dim, attention_heads, dropout)
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, player_indices, position_indices, mask):
        player_emb = self.player_embedding(player_indices)
        position_emb = self.position_embedding(position_indices)
        x = torch.cat([player_emb, position_emb], dim=-1)
        x = self.input_dropout(self.input_proj(x))
        for block in self.blocks:
            x = block(x, position_indices, mask)
        pooled = self.pooling(x, position_indices, mask)
        return self.rho(pooled)


# ============================================================================
# LINEUP PARSING
# ============================================================================

def lineup_sort_key(column_name: str):
    if "Goalkeeper" in column_name:
        return (0, 0)
    if "Defender" in column_name:
        return (1, int(column_name.rsplit("_", 1)[-1]))
    if "Midfielder" in column_name:
        return (2, int(column_name.rsplit("_", 1)[-1]))
    if "Attacker" in column_name:
        return (3, int(column_name.rsplit("_", 1)[-1]))
    return (9, 0)


def lineup_position_from_column(column_name: str) -> str:
    if "Goalkeeper" in column_name:
        return "GK"
    if "Defender" in column_name:
        return "DEF"
    if "Midfielder" in column_name:
        return "MID"
    if "Attacker" in column_name:
        return "ATT"
    raise KeyError(f"Cannot infer position: {column_name}")


def extract_lineup_players(row, side: str):
    prefix = f"{side}_"
    lineup_columns = [
        col for col in row.index
        if col.startswith(prefix) and col not in {"Home Team", "Away Team"}
    ]
    lineup_columns = sorted(lineup_columns, key=lineup_sort_key)

    players, positions = [], []
    for column in lineup_columns:
        player_id = row[column]
        if pd.isna(player_id) or str(player_id).strip() == "":
            continue
        players.append(str(player_id).strip())
        positions.append(lineup_position_from_column(column))
    return players, positions


def normalize_lineup(players, positions, max_players):
    players = players[:max_players]
    positions = positions[:max_players]
    mask = [1.0] * len(players)
    while len(players) < max_players:
        players.append("<PAD>")
        positions.append("GK")
        mask.append(0.0)
    return players, positions, mask


# ============================================================================
# PLAYER NAME → VOCAB INDEX RESOLUTION
# ============================================================================

def load_vocabulary(path: Path):
    """Load the trained vocabulary. Returns (vocab_dict, list_of_known_players)."""
    df = pd.read_csv(path)
    if "player_id" not in df.columns or "index" not in df.columns:
        raise KeyError(f"{path.name} must contain player_id and index columns.")
    vocab = dict(zip(df["player_id"].astype(str), df["index"].astype(int)))

    # Collect known real players (exclude special tokens)
    known = [p for p in vocab.keys() if p not in {"<PAD>", "<UNK>"}]

    if "<PAD>" not in vocab or "<UNK>" not in vocab:
        raise KeyError(f"{path.name} is missing <PAD> or <UNK> tokens.")
    return vocab, known


def build_player_resolver(vocab: dict, known_players: list[str]):
    """
    Return a function that maps a name to a vocab index.

    Strategy:
      1. exact match (case-sensitive)
      2. exact match (case-insensitive)
      3. fuzzy match via rapidfuzz with a strict threshold
      4. fallback to <UNK>
    """
    vocab_lower = {k.lower(): v for k, v in vocab.items()}
    unk_idx = vocab["<UNK>"]

    if USING_RAPIDFUZZ and known_players:
        known_lower = [p.lower() for p in known_players]
        lower_to_original = {p.lower(): p for p in known_players}

    unmatched = []

    def resolve(name: str) -> int:
        if not name:
            return unk_idx
        name_stripped = name.strip()

        # 1. Exact
        if name_stripped in vocab:
            return vocab[name_stripped]

        # 2. Case-insensitive exact
        lowered = name_stripped.lower()
        if lowered in vocab_lower:
            return vocab_lower[lowered]

        # 3. Fuzzy
        if USING_RAPIDFUZZ and known_players:
            match = process.extractOne(lowered, known_lower, scorer=fuzz.ratio)
            if match is not None:
                candidate_lower, score, _ = match
                if score >= FUZZY_MATCH_THRESHOLD:
                    original = lower_to_original.get(candidate_lower)
                    if original is not None:
                        return vocab[original]

        # 4. Unknown
        unmatched.append(name_stripped)
        return unk_idx

    return resolve, unmatched


# ============================================================================
# EMBEDDING COMPUTATION
# ============================================================================

def build_side_tensors(rows, side, resolve, max_players):
    player_idx_rows, position_idx_rows, mask_rows = [], [], []
    for _, row in rows.iterrows():
        players, positions = extract_lineup_players(row, side)
        players, positions, mask = normalize_lineup(players, positions, max_players)
        player_idx_rows.append([resolve(p) for p in players])
        position_idx_rows.append([POSITION_TO_INDEX[pos] for pos in positions])
        mask_rows.append(mask)
    return (
        torch.tensor(player_idx_rows, dtype=torch.long),
        torch.tensor(position_idx_rows, dtype=torch.long),
        torch.tensor(mask_rows, dtype=torch.float32),
    )


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 78)
    print("DASHBOARD: Lineup embeddings for upcoming fixtures")
    print("=" * 78)

    # ------------------------------------------------------------------
    # Load trained artifacts (read-only)
    # ------------------------------------------------------------------
    if not STATE_DICT_PATH.exists():
        print(f"Error: trained encoder not found at {STATE_DICT_PATH}")
        return
    if not VOCAB_PATH.exists():
        print(f"Error: trained vocabulary not found at {VOCAB_PATH}")
        return

    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())
        config = {
            "player_embedding_dim": metrics.get("player_embedding_dim", 32),
            "position_embedding_dim": 8,
            "hidden_dim": metrics.get("hidden_dim", 64),
            "output_dim": metrics.get("output_dim", 32),
            "attention_heads": metrics.get("attention_heads", 4),
            "num_transformer_blocks": metrics.get("num_transformer_blocks", 2),
            "dropout": 0.15,
            "max_players_per_team": 11,
        }
        print(f"Loaded config from {METRICS_PATH.name}")
    else:
        # Fallback: use defaults matching the training script
        config = {
            "player_embedding_dim": 32,
            "position_embedding_dim": 8,
            "hidden_dim": 64,
            "output_dim": 32,
            "attention_heads": 4,
            "num_transformer_blocks": 2,
            "dropout": 0.15,
            "max_players_per_team": 11,
        }
        print(f"Metrics file not found; using default config")

    vocab, known_players = load_vocabulary(VOCAB_PATH)
    print(f"Loaded vocabulary: {len(vocab)} entries "
          f"({len(known_players)} players + special tokens)")

    # ------------------------------------------------------------------
    # Load upcoming fixtures
    # ------------------------------------------------------------------
    if not UPCOMING_LINEUPS_PATH.exists():
        print(f"Error: upcoming lineups not found at {UPCOMING_LINEUPS_PATH}")
        return

    upcoming = pd.read_csv(UPCOMING_LINEUPS_PATH, low_memory=False)
    print(f"Loaded upcoming fixtures: {len(upcoming)} rows")

    if "game_id" not in upcoming.columns:
        upcoming = upcoming.copy()
        upcoming["game_id"] = (
            upcoming["Date"].astype(str).str.replace("-", "", regex=False)
            + "_" + upcoming["Home Team"].astype(str)
            + "_" + upcoming["Away Team"].astype(str)
        )

    # ------------------------------------------------------------------
    # Build the resolver
    # ------------------------------------------------------------------
    resolve, unmatched = build_player_resolver(vocab, known_players)

    # ------------------------------------------------------------------
    # Load model and weights
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = ImprovedLineupEncoder(config, vocab_size=len(vocab)).to(device)
    state_dict = torch.load(STATE_DICT_PATH, map_location=device)
    # Support both raw state_dict and wrapped {"state_dict": ...}
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    encoder.load_state_dict(state_dict)
    encoder.eval()
    print(f"Loaded encoder weights from {STATE_DICT_PATH}")

    # ------------------------------------------------------------------
    # Encode home and away sides
    # ------------------------------------------------------------------
    print("\nEncoding upcoming lineups...")
    home_idx, home_pos, home_mask = build_side_tensors(
        upcoming, "Home", resolve, config["max_players_per_team"]
    )
    away_idx, away_pos, away_mask = build_side_tensors(
        upcoming, "Away", resolve, config["max_players_per_team"]
    )

    with torch.no_grad():
        home_emb = encoder(home_idx.to(device), home_pos.to(device), home_mask.to(device))
        away_emb = encoder(away_idx.to(device), away_pos.to(device), away_mask.to(device))
        home_emb = home_emb.cpu().numpy()
        away_emb = away_emb.cpu().numpy()

    # ------------------------------------------------------------------
    # Build output frame (same column names as training)
    # ------------------------------------------------------------------
    output = pd.DataFrame({
        "game_id": upcoming["game_id"].astype(str).values,
        "Date": upcoming["Date"].values,
        "Home Team": upcoming["Home Team"].values,
        "Away Team": upcoming["Away Team"].values,
    })
    for idx in range(config["output_dim"]):
        output[f"home_lineup_embedding_{idx + 1}"] = home_emb[:, idx]
        output[f"away_lineup_embedding_{idx + 1}"] = away_emb[:, idx]

    # ------------------------------------------------------------------
    # Report unmatched players (optional, for diagnostics)
    # ------------------------------------------------------------------
    if unmatched:
        unique_unmatched = sorted(set(unmatched))
        print(f"\n⚠ {len(unique_unmatched)} unique player names fell back to <UNK>:")
        for name in unique_unmatched[:15]:
            print(f"    - {name}")
        if len(unique_unmatched) > 15:
            print(f"    ... and {len(unique_unmatched) - 15} more")

        pd.DataFrame({"unmatched_player": unique_unmatched}).to_csv(
            UNMATCHED_PATH, index=False, encoding="utf-8-sig"
        )
        print(f"  Saved unmatched players to: {UNMATCHED_PATH}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Fixtures encoded:      {len(output)}")
    print(f"Output embedding dims: {config['output_dim']} per side")
    print(f"Total columns:         {output.shape[1]}")
    print(f"Saved to:              {OUTPUT_PATH}")


if __name__ == "__main__":
    main()