#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compute lineup embeddings for UPCOMING fixtures (dashboard use).

Uses the OPTIMIZED cross-team transformer encoder:
- Intra-team pre-LN transformer blocks
- Cross-team attention blocks (home and away attend to each other)
- Team-ID embeddings
- Multi-query attention pooling

Reuses:
- misc/lineup_encoder_state_dict.pt
- misc/lineup_player_vocab.csv
- misc/lineup_team_vocab.csv          <-- NEW (produced by training script)
- misc/lineup_encoder_training_metrics.json

New lineup player names not in the vocabulary are resolved via rapidfuzz.

IMPORTANT: the checkpoint at misc/lineup_encoder_state_dict.pt must have
been produced by the matching training script (with cross-team blocks,
team embeddings, and multi-query pooling). If it was produced by an older
version, this script will raise a clear error at load_state_dict — re-run
training to regenerate the checkpoint.

Reads:
- 5_new_prediction_dashboard/upcoming_lineups/upcoming_lineups_latest.csv
- misc/lineup_encoder_state_dict.pt
- misc/lineup_player_vocab.csv
- misc/lineup_team_vocab.csv
- misc/lineup_encoder_training_metrics.json

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
TEAM_VOCAB_PATH = PROJECT_DIR / "misc" / "lineup_team_vocab.csv"
METRICS_PATH = PROJECT_DIR / "misc" / "lineup_encoder_training_metrics.json"
PREDICTION_DATA_DIR = SCRIPT_DIR / "prediction_data"
OUTPUT_PATH = PREDICTION_DATA_DIR / "upcoming_lineup_embeddings.csv"
UNMATCHED_PATH = PREDICTION_DATA_DIR / "upcoming_lineup_unmatched_players.csv"

PREDICTION_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CONFIG
# ============================================================================

POSITION_TO_INDEX = {"GK": 0, "DEF": 1, "MID": 2, "ATT": 3}
FUZZY_MATCH_THRESHOLD = 80.0


# ============================================================================
# MODEL DEFINITION (must match the optimized training script exactly)
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Self-attention with position bias and optional same-position bias
    (used by the intra-team blocks)."""

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

    def forward(self, x, position_indices, mask, team_ids=None):
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
            if team_ids is not None:
                team_eq = team_ids.unsqueeze(2) == team_ids.unsqueeze(1)
                same_bias = (pos_eq & team_eq).float() * self.same_position_bias
            else:
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
    """Pre-LayerNorm intra-team block."""

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

    def forward(self, x, position_indices, mask, team_ids=None):
        x = x + self.attention(self.norm1(x), position_indices, mask, team_ids)
        x = x + self.ffn(self.norm2(x))
        return x


class CrossTeamBlock(nn.Module):
    """Cross-team block: home and away concatenated and attended together."""

    def __init__(self, hidden_dim, attention_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_dim, attention_heads, dropout=dropout, batch_first=True
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

    def forward(self, home_x, away_x, home_mask, away_mask):
        n_home = home_x.size(1)
        x = torch.cat([home_x, away_x], dim=1)
        mask = torch.cat([home_mask, away_mask], dim=1)
        key_padding_mask = (mask == 0)

        attn_out, _ = self.attn(
            self.norm1(x), self.norm1(x), self.norm1(x),
            key_padding_mask=key_padding_mask,
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))

        return x[:, :n_home], x[:, n_home:]


class AttentionPooling(nn.Module):
    """Multi-query attention pooling."""

    def __init__(self, hidden_dim, attention_heads=4, dropout=0.1, num_queries=4):
        super().__init__()
        self.num_queries = num_queries
        self.attn = nn.MultiheadAttention(
            hidden_dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.pool_queries = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim * num_queries, hidden_dim)

    def forward(self, x, mask):
        batch_size = x.size(0)
        queries = self.pool_queries.unsqueeze(0).expand(batch_size, -1, -1)
        key_padding_mask = (mask == 0)

        # Safety: if a row is entirely masked, un-mask the first slot
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked, 0] = False

        pooled, _ = self.attn(
            queries, self.norm(x), self.norm(x),
            key_padding_mask=key_padding_mask,
        )
        pooled = pooled.reshape(batch_size, -1)
        return self.out_proj(pooled)


class ImprovedLineupEncoder(nn.Module):
    """Encoder with intra-team blocks, cross-team blocks, multi-query pooling.

    forward() takes BOTH sides at once and returns (home_emb, away_emb).
    """

    def __init__(self, config: dict, vocab_size: int, team_vocab_size: int):
        super().__init__()
        self.config = config

        pe = config["player_embedding_dim"]
        po = config["position_embedding_dim"]
        sl = config["slot_embedding_dim"]
        tm = config["team_embedding_dim"]
        hd = config["hidden_dim"]
        od = config["output_dim"]
        heads = config["attention_heads"]
        n_intra = config["num_transformer_blocks"]
        n_cross = config["num_cross_team_blocks"]
        drop = config["dropout"]
        max_players = config["max_players_per_team"]
        spb = config["same_position_bias"]

        self.player_embedding = nn.Embedding(vocab_size, pe)
        self.position_embedding = nn.Embedding(len(POSITION_TO_INDEX), po)
        self.slot_embedding = nn.Embedding(max_players, sl)
        self.team_embedding = nn.Embedding(team_vocab_size, tm)

        input_dim = pe + po + sl + tm
        self.input_proj = nn.Linear(input_dim, hd)
        self.input_dropout = nn.Dropout(drop)

        self.intra_blocks = nn.ModuleList([
            TransformerBlock(hd, heads, drop, same_position_bias=spb)
            for _ in range(n_intra)
        ])
        self.cross_blocks = nn.ModuleList([
            CrossTeamBlock(hd, heads, drop) for _ in range(n_cross)
        ])

        self.pooling = AttentionPooling(hd, heads, drop, num_queries=4)

        self.rho = nn.Sequential(
            nn.Linear(hd, hd),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hd, hd // 2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hd // 2, od),
        )

    def _embed_side(self, player_indices, position_indices, team_indices, mask):
        batch_size, num_players = player_indices.shape
        player_emb = self.player_embedding(player_indices)
        position_emb = self.position_embedding(position_indices)

        slot_indices = torch.arange(num_players, device=player_indices.device)
        slot_indices = slot_indices.unsqueeze(0).expand(batch_size, -1)
        slot_emb = self.slot_embedding(slot_indices)

        team_emb = self.team_embedding(team_indices).unsqueeze(1).expand(-1, num_players, -1)

        x = torch.cat([player_emb, position_emb, slot_emb, team_emb], dim=-1)
        x = self.input_dropout(self.input_proj(x))
        return x

    def forward(self, home_player_idx, home_pos_idx, home_team_idx, home_mask,
                away_player_idx, away_pos_idx, away_team_idx, away_mask):
        home_x = self._embed_side(home_player_idx, home_pos_idx, home_team_idx, home_mask)
        away_x = self._embed_side(away_player_idx, away_pos_idx, away_team_idx, away_mask)

        for block in self.intra_blocks:
            home_x = block(home_x, home_pos_idx, home_mask)
            away_x = block(away_x, away_pos_idx, away_mask)

        for block in self.cross_blocks:
            home_x, away_x = block(home_x, away_x, home_mask, away_mask)

        home_pooled = self.pooling(home_x, home_mask)
        away_pooled = self.pooling(away_x, away_mask)

        return self.rho(home_pooled), self.rho(away_pooled)


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
    # Ensure at least one valid slot (avoids all-masked pooling NaN)
    if len(players) == 0:
        players = ["<UNK>"]
        positions = ["GK"]
        mask = [1.0]
    while len(players) < max_players:
        players.append("<PAD>")
        positions.append("GK")
        mask.append(0.0)
    return players, positions, mask


def extract_team_name(row, side: str) -> str:
    col = f"{side} Team"
    if col in row.index:
        v = row[col]
        if pd.notna(v):
            return str(v).strip()
    return "<UNK>"


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


def load_team_vocabulary(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Team vocabulary not found at {path}. "
            "Re-run the optimized training script to generate it."
        )
    df = pd.read_csv(path)
    if "team_id" not in df.columns or "index" not in df.columns:
        raise KeyError(f"{path.name} must contain team_id and index columns.")
    vocab = dict(zip(df["team_id"].astype(str), df["index"].astype(int)))
    if "<UNK>" not in vocab:
        raise KeyError(f"{path.name} is missing the <UNK> token.")
    return vocab


def _normalize_player_name(name: str) -> str:
    """Lowercase, strip accents, remove punctuation, drop leading initial."""
    import unicodedata
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    s = re.sub(r"[.\-'’]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    tokens = s.split()
    if len(tokens) >= 2 and len(tokens[0]) <= 2:
        tokens = tokens[1:]
    return " ".join(tokens)


def _last_word_key(name: str) -> str:
    s = _normalize_player_name(name)
    tokens = s.split()
    return tokens[-1] if tokens else ""


def build_player_resolver(vocab: dict, known_players: list[str]):
    vocab_lower = {k.lower(): v for k, v in vocab.items()}
    unk_idx = vocab["<UNK>"]

    if USING_RAPIDFUZZ and known_players:
        known_lower = [p.lower() for p in known_players]
        known_norm = [_normalize_player_name(p) for p in known_players]
        known_surname = [_last_word_key(p) for p in known_players]

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

        # 1. Exact
        if name_stripped in vocab:
            return vocab[name_stripped]

        # 2. Case-insensitive
        lowered = name_stripped.lower()
        if lowered in vocab_lower:
            return vocab_lower[lowered]

        # 3. Normalized exact
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


# ============================================================================
# CONFIG LOADING
# ============================================================================

def load_config_from_metrics(path: Path) -> dict:
    """Load config matching the OPTIMIZED training script. Falls back to
    the same defaults if the metrics file is missing fields."""
    defaults = {
        "player_embedding_dim": 48,
        "position_embedding_dim": 12,
        "slot_embedding_dim": 8,
        "team_embedding_dim": 8,
        "hidden_dim": 128,
        "output_dim": 64,
        "attention_heads": 8,
        "num_transformer_blocks": 3,
        "num_cross_team_blocks": 2,
        "dropout": 0.2,
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


def build_team_tensor(rows, side, team_vocab):
    unk = team_vocab["<UNK>"]
    return torch.tensor(
        [team_vocab.get(extract_team_name(row, side), unk) for _, row in rows.iterrows()],
        dtype=torch.long,
    )


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 78)
    print("DASHBOARD: Lineup embeddings for upcoming fixtures (optimized encoder)")
    print("=" * 78)

    if not STATE_DICT_PATH.exists():
        print(f"Error: trained encoder not found at {STATE_DICT_PATH}")
        return
    if not VOCAB_PATH.exists():
        print(f"Error: trained vocabulary not found at {VOCAB_PATH}")
        return
    if not TEAM_VOCAB_PATH.exists():
        print(f"Error: team vocabulary not found at {TEAM_VOCAB_PATH}")
        print("       Re-run the optimized training script to generate it.")
        return

    config = load_config_from_metrics(METRICS_PATH)
    print(f"Encoder config: {config}")

    vocab, known_players = load_vocabulary(VOCAB_PATH)
    team_vocab = load_team_vocabulary(TEAM_VOCAB_PATH)
    print(f"Loaded player vocabulary: {len(vocab)} entries "
          f"({len(known_players)} players + special tokens)")
    print(f"Loaded team vocabulary:   {len(team_vocab)} entries")

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
    encoder = ImprovedLineupEncoder(
        config, vocab_size=len(vocab), team_vocab_size=len(team_vocab)
    ).to(device)

    state_dict = torch.load(STATE_DICT_PATH, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Fail loudly if the architecture doesn't match the checkpoint.
    model_keys = set(encoder.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys
    if missing or unexpected:
        raise RuntimeError(
            "Encoder architecture does not match the checkpoint.\n"
            f"  Missing keys in checkpoint: {sorted(missing)[:6]}"
            f"{' ...' if len(missing) > 6 else ''}\n"
            f"  Unexpected keys in checkpoint: {sorted(unexpected)[:6]}"
            f"{' ...' if len(unexpected) > 6 else ''}\n"
            "  This usually means the checkpoint was produced by an older "
            "training script.\n"
            "  Re-run the optimized training script to regenerate "
            "lineup_encoder_state_dict.pt and lineup_team_vocab.csv."
        )

    encoder.load_state_dict(state_dict)
    encoder.eval()
    print(f"Loaded encoder weights from {STATE_DICT_PATH}")

    print("\nEncoding upcoming lineups...")
    max_p = config["max_players_per_team"]
    home_idx, home_pos, home_mask = build_side_tensors(upcoming, "Home", resolve, max_p)
    away_idx, away_pos, away_mask = build_side_tensors(upcoming, "Away", resolve, max_p)
    home_team = build_team_tensor(upcoming, "Home", team_vocab)
    away_team = build_team_tensor(upcoming, "Away", team_vocab)

    with torch.no_grad():
        home_emb, away_emb = encoder(
            home_idx.to(device), home_pos.to(device),
            home_team.to(device), home_mask.to(device),
            away_idx.to(device), away_pos.to(device),
            away_team.to(device), away_mask.to(device),
        )
        home_emb = home_emb.cpu().numpy()
        away_emb = away_emb.cpu().numpy()

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