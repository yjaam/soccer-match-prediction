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
- misc/lineup_encoder_training_metrics.json  (read-only, for config)

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

# Fuzzy matching threshold. 88 is a good balance: tolerant of typos and
# abbreviations, but strict enough to avoid mismatching unrelated players.
FUZZY_MATCH_THRESHOLD = 80.0


# ============================================================================
# MODEL DEFINITION (must exactly match d_create_lineup_embeddings.py)
# ============================================================================

class MultiHeadAttention(nn.Module):
    """
    Mirrors the training-time attention:
      - Learnable positional bias (by position category)
      - Optional same-position structural bias
      - Padding mask
    """

    def __init__(self, hidden_dim, attention_heads=4, dropout=0.1,
                 same_position_bias=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_heads = attention_heads
        self.head_dim = hidden_dim // attention_heads
        self.attention_scale = self.head_dim ** 0.5
        self.same_position_bias = same_position_bias

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
        attn_scores = attn_scores + pos_bias.unsqueeze(1).unsqueeze(2)

        if self.same_position_bias != 0.0:
            pos_eq = position_indices.unsqueeze(2) == position_indices.unsqueeze(1)
            same_bias = pos_eq.float() * self.same_position_bias
            attn_scores = attn_scores + same_bias.unsqueeze(1)

        mask_expanded = mask.unsqueeze(1).unsqueeze(2)
        attn_scores = attn_scores.masked_fill(mask_expanded == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)

        attended = torch.matmul(attn_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, num_players, self.hidden_dim
        )
        return self.out_proj(attended)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm transformer block — matches the training script."""

    def __init__(self, hidden_dim, attention_heads, dropout, same_position_bias=0.5):
        super().__init__()
        self.attention = MultiHeadAttention(
            hidden_dim, attention_heads, dropout, same_position_bias
        )
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
        x = x + self.attention(self.norm1(x), position_indices, mask)
        x = x + self.ffn(self.norm2(x))
        return x


class AttentionPooling(nn.Module):
    """Attention pooling with a learnable query — matches the training script."""

    def __init__(self, hidden_dim, attention_heads=4, dropout=0.1,
                 same_position_bias=0.0):
        super().__init__()
        self.attention = MultiHeadAttention(
            hidden_dim, attention_heads, dropout, same_position_bias
        )
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

    def forward(self, x, position_indices, mask):
        batch_size, num_players, _ = x.shape

        pool_query = self.pool_query.expand(batch_size, -1, -1)
        x_with_query = torch.cat([pool_query, x], dim=1)

        query_mask = torch.ones(batch_size, 1, device=mask.device)
        extended_mask = torch.cat([query_mask, mask], dim=1)

        query_pos = torch.zeros(batch_size, 1, dtype=position_indices.dtype,
                                device=position_indices.device)
        extended_pos = torch.cat([query_pos, position_indices], dim=1)

        attended = self.attention(x_with_query, extended_pos, extended_mask)
        pooled = attended[:, 0, :]
        return pooled


class ImprovedLineupEncoder(nn.Module):
    """Must exactly mirror d_create_lineup_embeddings.ImprovedLineupEncoder."""

    def __init__(self, config: dict, vocab_size: int):
        super().__init__()
        self.config = config

        player_embedding_dim = config["player_embedding_dim"]
        position_embedding_dim = config["position_embedding_dim"]
        slot_embedding_dim = config["slot_embedding_dim"]
        hidden_dim = config["hidden_dim"]
        output_dim = config["output_dim"]
        attention_heads = config["attention_heads"]
        num_blocks = config["num_transformer_blocks"]
        dropout = config["dropout"]
        max_players = config["max_players_per_team"]
        same_position_bias = config["same_position_bias"]

        self.player_embedding = nn.Embedding(vocab_size, player_embedding_dim)
        self.position_embedding = nn.Embedding(len(POSITION_TO_INDEX), position_embedding_dim)
        self.slot_embedding = nn.Embedding(max_players, slot_embedding_dim)

        input_dim = player_embedding_dim + position_embedding_dim + slot_embedding_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, attention_heads, dropout,
                             same_position_bias=same_position_bias)
            for _ in range(num_blocks)
        ])

        self.pooling = AttentionPooling(hidden_dim, attention_heads, dropout)

        # Deeper final head — matches the training script.
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, player_indices, position_indices, mask):
        player_emb = self.player_embedding(player_indices)
        position_emb = self.position_embedding(position_indices)

        batch_size, num_players = player_indices.shape
        slot_indices = torch.arange(num_players, device=player_indices.device)
        slot_indices = slot_indices.unsqueeze(0).expand(batch_size, -1)
        slot_emb = self.slot_embedding(slot_indices)

        x = torch.cat([player_emb, position_emb, slot_emb], dim=-1)
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
    df = pd.read_csv(path)
    if "player_id" not in df.columns or "index" not in df.columns:
        raise KeyError(f"{path.name} must contain player_id and index columns.")
    vocab = dict(zip(df["player_id"].astype(str), df["index"].astype(int)))
    known = [p for p in vocab.keys() if p not in {"<PAD>", "<UNK>"}]
    if "<PAD>" not in vocab or "<UNK>" not in vocab:
        raise KeyError(f"{path.name} is missing <PAD> or <UNK> tokens.")
    return vocab, known


def _normalize_player_name(name: str) -> str:
    """
    Reduce a player name to a comparison key.

    - lowercases
    - strips accents
    - removes punctuation
    - collapses an initial ('A.' or 'A') followed by a surname into 'surname'
      only, so 'A. Isak' -> 'isak'

    This makes the resolver insensitive to whether the source uses
    'A. Isak', 'A Isak', 'Alexander Isak', or 'Isak'.
    """
    import unicodedata
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    # Remove punctuation (periods, hyphens, apostrophes)
    s = re.sub(r"[.\-'’]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    # If it's just an initial + surname, keep the surname
    tokens = s.split()
    if len(tokens) >= 2 and len(tokens[0]) <= 2:
        # Drop the leading initial token
        tokens = tokens[1:]
    return " ".join(tokens)


def _last_word_key(name: str) -> str:
    """Return the last word of a name, for surname-based fallback."""
    s = _normalize_player_name(name)
    tokens = s.split()
    return tokens[-1] if tokens else ""


def build_player_resolver(vocab: dict, known_players: list[str]):
    """
    Return (resolve, unmatched) where `resolve(name) -> vocab_index`.

    Resolution order:
      1. exact match (case-sensitive)
      2. exact match (case-insensitive)
      3. normalized-key exact match
      4. fuzzy match on normalized keys (rapidfuzz, threshold 82)
      5. surname-only fallback (exact or fuzzy at 85)
      6. <UNK>
    """
    vocab_lower = {k.lower(): v for k, v in vocab.items()}
    unk_idx = vocab["<UNK>"]

    # Precompute normalized keys for the known players
    if USING_RAPIDFUZZ and known_players:
        known_lower = [p.lower() for p in known_players]
        known_norm = [_normalize_player_name(p) for p in known_players]
        known_surname = [_last_word_key(p) for p in known_players]
        lower_to_original = {p.lower(): p for p in known_players}
        # Fast lookup tables for the exact-match stages
        norm_to_original = {}
        surname_to_original = {}
        for orig, norm, surname in zip(known_players, known_norm, known_surname):
            if norm and norm not in norm_to_original:
                norm_to_original[norm] = orig
            if surname and surname not in surname_to_original:
                surname_to_original[surname] = orig

    unmatched = []

    def resolve(name: str) -> int:
        if not name:
            return unk_idx
        name_stripped = name.strip()

        # 1. Exact (case-sensitive)
        if name_stripped in vocab:
            return vocab[name_stripped]

        # 2. Exact (case-insensitive)
        lowered = name_stripped.lower()
        if lowered in vocab_lower:
            return vocab_lower[lowered]

        # 3. Normalized-key exact
        norm = _normalize_player_name(name_stripped)
        if USING_RAPIDFUZZ and norm and norm in norm_to_original:
            return vocab[norm_to_original[norm]]

        # 4. Fuzzy on normalized keys
        if USING_RAPIDFUZZ and known_players and norm:
            match = process.extractOne(norm, known_norm, scorer=fuzz.ratio)
            if match is not None:
                _, score, idx = match
                if score >= 82.0:
                    return vocab[known_players[idx]]

        # 5. Surname fallback
        surname = _last_word_key(name_stripped)
        if USING_RAPIDFUZZ and surname:
            if surname in surname_to_original:
                return vocab[surname_to_original[surname]]
            match = process.extractOne(surname, known_surname, scorer=fuzz.ratio)
            if match is not None:
                _, score, idx = match
                if score >= 85.0:
                    return vocab[known_players[idx]]

        unmatched.append(name_stripped)
        return unk_idx

    return resolve, unmatched

    def resolve(name: str) -> int:
        if not name:
            return unk_idx
        name_stripped = name.strip()

        if name_stripped in vocab:
            return vocab[name_stripped]

        lowered = name_stripped.lower()
        if lowered in vocab_lower:
            return vocab_lower[lowered]

        if USING_RAPIDFUZZ and known_players:
            match = process.extractOne(lowered, known_lower, scorer=fuzz.ratio)
            if match is not None:
                candidate_lower, score, _ = match
                if score >= FUZZY_MATCH_THRESHOLD:
                    original = lower_to_original.get(candidate_lower)
                    if original is not None:
                        return vocab[original]

        unmatched.append(name_stripped)
        return unk_idx

    return resolve, unmatched


# ============================================================================
# CONFIG LOADING
# ============================================================================

def load_config_from_metrics(path: Path) -> dict:
    """
    Build the encoder config from the training metrics file.

    Falls back to hard-coded defaults matching d_create_lineup_embeddings.py
    if the file is missing or lacks a field.
    """
    defaults = {
        "player_embedding_dim": 32,
        "position_embedding_dim": 8,
        "slot_embedding_dim": 4,
        "hidden_dim": 64,
        "output_dim": 32,
        "attention_heads": 4,
        "num_transformer_blocks": 2,
        "dropout": 0.15,
        "max_players_per_team": 11,
        "same_position_bias": 0.5,
    }
    if not path.exists():
        print(f"Metrics file not found at {path}; using default config.")
        return defaults

    metrics = json.loads(path.read_text())
    cfg = {}
    for k, v in defaults.items():
        cfg[k] = metrics.get(k, v)
    return cfg


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

    if not STATE_DICT_PATH.exists():
        print(f"Error: trained encoder not found at {STATE_DICT_PATH}")
        return
    if not VOCAB_PATH.exists():
        print(f"Error: trained vocabulary not found at {VOCAB_PATH}")
        return

    config = load_config_from_metrics(METRICS_PATH)
    print(f"Encoder config: {config}")

    vocab, known_players = load_vocabulary(VOCAB_PATH)
    print(f"Loaded vocabulary: {len(vocab)} entries "
          f"({len(known_players)} players + special tokens)")

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

    resolve, unmatched = build_player_resolver(vocab, known_players)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = ImprovedLineupEncoder(config, vocab_size=len(vocab)).to(device)

    state_dict = torch.load(STATE_DICT_PATH, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Fail loudly if the architecture doesn't match the checkpoint — a clear
    # message beats the default PyTorch "Missing key" wall of text.
    model_keys = set(encoder.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys
    if missing or unexpected:
        raise RuntimeError(
            "Encoder architecture does not match the checkpoint.\n"
            f"  Missing keys in checkpoint: {sorted(missing)[:5]}"
            f"{' ...' if len(missing) > 5 else ''}\n"
            f"  Unexpected keys in checkpoint: {sorted(unexpected)[:5]}"
            f"{' ...' if len(unexpected) > 5 else ''}\n"
            "  Make sure d_create_lineup_embeddings.py and this script "
            "define the same encoder architecture."
        )

    encoder.load_state_dict(state_dict)
    encoder.eval()
    print(f"Loaded encoder weights from {STATE_DICT_PATH}")

    print("\nEncoding upcoming lineups...")
    home_idx, home_pos, home_mask = build_side_tensors(
        upcoming, "Home", resolve, config["max_players_per_team"]
    )
    away_idx, away_pos, away_mask = build_side_tensors(
        upcoming, "Away", resolve, config["max_players_per_team"]
    )

    with torch.no_grad():
        home_emb = encoder(home_idx.to(device), home_pos.to(device),
                           home_mask.to(device)).cpu().numpy()
        away_emb = encoder(away_idx.to(device), away_pos.to(device),
                           away_mask.to(device)).cpu().numpy()

    output = pd.DataFrame({
        "game_id": upcoming["game_id"].astype(str).values,
        "Date": upcoming["Date"].values,
        "Home Team": upcoming["Home Team"].values,
        "Away Team": upcoming["Away Team"].values,
    })
    for idx in range(config["output_dim"]):
        output[f"home_lineup_embedding_{idx + 1}"] = home_emb[:, idx]
        output[f"away_lineup_embedding_{idx + 1}"] = away_emb[:, idx]

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