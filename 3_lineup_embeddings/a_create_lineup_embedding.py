#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Create train/test lineup embeddings with an IMPROVED 2-Block Transformer Encoder.

Key improvements over the previous version:
- Proper train/validation split (no test-set leakage for early stopping / LR)
- Cross-team interaction blocks: home and away players attend to each other
  so the encoder can learn matchups and chemistry across both teams
- Team-ID embeddings to capture club-level playing styles
- Auxiliary match-outcome prediction head (win/draw/loss) for a richer
  training signal, weighted against the Poisson goal NLL
- Cosine LR schedule with linear warmup
- Larger model capacity (3 intra-team blocks, 128 hidden dim, 64 output dim)
- Multiple attention-pooling queries for a richer team summary
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
INPUT_PATH = PROJECT_DIR / "data" / "lineups.csv"
GOALS_SOURCE_PATH = PROJECT_DIR / "data" / "big5_matches_fixed.csv"
FINAL_DATA_DIR = PROJECT_DIR / "final_data"
MISC_DIR = PROJECT_DIR / "misc"

TRAIN_OUTPUT_PATH = FINAL_DATA_DIR / "lineup_embeddings_TRAIN.csv"
TEST_OUTPUT_PATH = FINAL_DATA_DIR / "lineup_embeddings_TEST.csv"
STATE_DICT_PATH = MISC_DIR / "lineup_encoder_state_dict.pt"
VOCAB_PATH = MISC_DIR / "lineup_player_vocab.csv"
TEAM_VOCAB_PATH = MISC_DIR / "lineup_team_vocab.csv"
METRICS_PATH = MISC_DIR / "lineup_encoder_training_metrics.json"

POSITION_TO_INDEX = {"GK": 0, "DEF": 1, "MID": 2, "ATT": 3}


@dataclass
class LineupEncoderConfig:
    # --- Embedding dimensions ---
    player_embedding_dim: int = 48
    position_embedding_dim: int = 12
    slot_embedding_dim: int = 8
    team_embedding_dim: int = 8
    hidden_dim: int = 128
    output_dim: int = 64
    max_players_per_team: int = 11

    # --- Transformer ---
    attention_heads: int = 8
    num_transformer_blocks: int = 3          # intra-team blocks
    num_cross_team_blocks: int = 2           # cross-team blocks
    same_position_bias: float = 0.5

    # --- Training ---
    epochs: int = 80
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.2
    seed: int = 42
    val_ratio: float = 0.1
    early_stopping_patience: int = 15
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.05

    # --- Auxiliary loss ---
    aux_outcome_weight: float = 0.3

    # --- Misc ---
    grad_clip: float = 3.0
    max_goals: int = 12


# ============================================================================
# HELPERS
# ============================================================================

def lineup_sort_key(column_name: str) -> tuple[int, int]:
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
    raise KeyError(f"Cannot infer position from column name: {column_name}")


def extract_lineup_players(row: pd.Series, side: str) -> tuple[list[str], list[str]]:
    prefix = f"{side}_"
    lineup_columns = [
        col for col in row.index
        if col.startswith(prefix) and col not in {"Home Team", "Away Team"}
    ]
    lineup_columns = sorted(lineup_columns, key=lineup_sort_key)

    player_ids, positions = [], []
    for column in lineup_columns:
        player_id = row[column]
        if pd.isna(player_id) or str(player_id).strip() == "":
            continue
        player_ids.append(str(player_id).strip())
        positions.append(lineup_position_from_column(column))

    return player_ids, positions


def extract_team_id(row: pd.Series, side: str) -> str:
    col = f"{side} Team"
    if col in row.index:
        return str(row[col]).strip()
    return "<UNK>"


def normalize_lineup(player_ids, positions, max_players):
    player_ids = player_ids[:max_players]
    positions = positions[:max_players]
    mask = [1.0] * len(player_ids)

    # FIX: guarantee at least one valid slot so attention pooling never sees
    # an all-masked sequence (which would produce NaN in softmax).
    if len(player_ids) == 0:
        player_ids = ["<UNK>"]
        positions = ["GK"]
        mask = [1.0]

    while len(player_ids) < max_players:
        player_ids.append("<PAD>")
        positions.append("GK")
        mask.append(0.0)
    return player_ids, positions, mask


def season_start_year_from_date(date_value) -> float:
    date_value = pd.to_datetime(date_value, errors="coerce")
    if pd.isna(date_value):
        return np.nan
    return float(date_value.year if date_value.month >= 7 else date_value.year - 1)


def split_train_test(frame):
    if "Date" not in frame.columns:
        raise KeyError("Date column is required.")
    season_start_year = frame["Date"].apply(season_start_year_from_date)
    train_frame = frame.loc[season_start_year < 2025].copy()
    test_frame = frame.loc[season_start_year == 2025].copy()
    if train_frame.empty or test_frame.empty:
        raise ValueError("Empty split detected.")
    return train_frame, test_frame


def build_player_vocab(frame):
    players = []
    for side in ("Home", "Away"):
        for column in frame.columns:
            if column.startswith(f"{side}_") and column not in {"Home Team", "Away Team"}:
                players.extend(str(v).strip() for v in frame[column].dropna().tolist() if str(v).strip())
    unique_players = sorted(set(players))
    vocab = {player: idx for idx, player in enumerate(unique_players)}
    vocab["<PAD>"] = len(vocab)
    vocab["<UNK>"] = len(vocab)
    return vocab


def build_team_vocab(frame):
    teams = set()
    for side in ("Home", "Away"):
        for v in frame[f"{side} Team"].dropna().tolist():
            if str(v).strip():
                teams.add(str(v).strip())
    unique_teams = sorted(teams)
    vocab = {team: idx for idx, team in enumerate(unique_teams)}
    vocab["<UNK>"] = len(vocab)
    return vocab


def load_match_results(goals_path: Path) -> pd.DataFrame:
    if not goals_path.exists():
        raise FileNotFoundError(f"Goals source file not found: {goals_path}")

    raw = pd.read_csv(goals_path, low_memory=False)

    home_candidates = ["HG", "FTHG", "Home_goals", "home_goals"]
    away_candidates = ["AG", "FTAG", "Away_goals", "away_goals"]

    hg_col = next((c for c in home_candidates if c in raw.columns), None)
    ag_col = next((c for c in away_candidates if c in raw.columns), None)

    if "game_id" not in raw.columns or hg_col is None or ag_col is None:
        raise KeyError(
            f"Could not locate game_id + goal columns in {goals_path}. "
            f"Available columns: {list(raw.columns)}"
        )

    out = (
        raw[["game_id", hg_col, ag_col]]
        .rename(columns={hg_col: "HG", ag_col: "AG"})
        .drop_duplicates(subset=["game_id"], keep="first")
    )
    out["HG"] = pd.to_numeric(out["HG"], errors="coerce")
    out["AG"] = pd.to_numeric(out["AG"], errors="coerce")
    out = out.dropna(subset=["HG", "AG"])
    return out


def side_has_players(row: pd.Series, side: str) -> bool:
    ids, _ = extract_lineup_players(row, side)
    return len(ids) > 0


# ============================================================================
# ATTENTION MODULES
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with position bias and optional same-position bias."""

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

        # Position bias
        pos_bias = self.position_bias[position_indices]
        attn_scores = attn_scores + pos_bias.unsqueeze(1).unsqueeze(2)

        # Same-position structural bias
        if self.same_position_bias != 0.0:
            pos_eq = position_indices.unsqueeze(2) == position_indices.unsqueeze(1)
            if team_ids is not None:
                team_eq = team_ids.unsqueeze(2) == team_ids.unsqueeze(1)
                same_bias = (pos_eq & team_eq).float() * self.same_position_bias
            else:
                same_bias = pos_eq.float() * self.same_position_bias
            attn_scores = attn_scores + same_bias.unsqueeze(1)

        # Padding mask
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
    """Pre-LayerNorm transformer block with residual connections."""

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
    """
    Cross-team transformer block: home and away players are concatenated
    into a single sequence and self-attention is applied across all 22 players.
    This lets the encoder learn direct matchups (e.g., striker vs defender).
    Standard self-attention (no same-position bias) is used here.
    """

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
        key_padding_mask = (mask == 0)  # True means ignore

        attn_out, _ = self.attn(
            self.norm1(x), self.norm1(x), self.norm1(x),
            key_padding_mask=key_padding_mask,
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))

        return x[:, :n_home], x[:, n_home:]


class AttentionPooling(nn.Module):
    """
    Attention pooling with multiple learnable queries.
    Each query attends to the player sequence and produces one vector.
    The outputs are concatenated and projected back to hidden_dim.
    """

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

        # Safety: if a row is entirely masked, un-mask the first slot so
        # softmax over keys stays well-defined.
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked, 0] = False

        pooled, _ = self.attn(
            queries, self.norm(x), self.norm(x),
            key_padding_mask=key_padding_mask,
        )  # (B, num_queries, D)
        pooled = pooled.reshape(batch_size, -1)
        return self.out_proj(pooled)


# ============================================================================
# ENCODER
# ============================================================================

class ImprovedLineupEncoder(nn.Module):
    """Encoder with intra-team blocks, cross-team blocks, and multi-query pooling."""

    def __init__(self, player_vocab, team_vocab, config):
        super().__init__()
        self.config = config
        self.player_vocab = player_vocab
        self.team_vocab = team_vocab

        self.player_embedding = nn.Embedding(len(player_vocab), config.player_embedding_dim)
        self.position_embedding = nn.Embedding(len(POSITION_TO_INDEX), config.position_embedding_dim)
        self.slot_embedding = nn.Embedding(config.max_players_per_team, config.slot_embedding_dim)
        self.team_embedding = nn.Embedding(len(team_vocab), config.team_embedding_dim)

        input_dim = (
            config.player_embedding_dim
            + config.position_embedding_dim
            + config.slot_embedding_dim
            + config.team_embedding_dim
        )
        self.input_proj = nn.Linear(input_dim, config.hidden_dim)
        self.input_dropout = nn.Dropout(config.dropout)

        # Intra-team blocks
        self.intra_blocks = nn.ModuleList([
            TransformerBlock(
                config.hidden_dim, config.attention_heads, config.dropout,
                same_position_bias=config.same_position_bias,
            )
            for _ in range(config.num_transformer_blocks)
        ])

        # Cross-team blocks
        self.cross_blocks = nn.ModuleList([
            CrossTeamBlock(config.hidden_dim, config.attention_heads, config.dropout)
            for _ in range(config.num_cross_team_blocks)
        ])

        # Pooling
        self.pooling = AttentionPooling(
            config.hidden_dim, config.attention_heads, config.dropout,
            num_queries=4,
        )

        # Final projection head
        self.rho = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.output_dim),
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

        # Intra-team blocks (separately)
        for block in self.intra_blocks:
            home_x = block(home_x, home_pos_idx, home_mask)
            away_x = block(away_x, away_pos_idx, away_mask)

        # Cross-team blocks
        for block in self.cross_blocks:
            home_x, away_x = block(home_x, away_x, home_mask, away_mask)

        # Pool each team separately
        home_pooled = self.pooling(home_x, home_mask)
        away_pooled = self.pooling(away_x, away_mask)

        # Final projection
        home_emb = self.rho(home_pooled)
        away_emb = self.rho(away_pooled)

        return home_emb, away_emb


class LineupGoalPredictor(nn.Module):
    """
    Predicts home/away goals from the concatenated team embeddings.
    Also outputs an auxiliary 3-class match-outcome logit.
    """

    def __init__(self, encoder, hidden_dim=128):
        super().__init__()
        self.encoder = encoder
        in_dim = encoder.config.output_dim * 2  # concatenated home + away

        self.goal_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, 2),
        )
        self.outcome_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, batch):
        home_emb, away_emb = self.encoder(
            batch["home_player_idx"], batch["home_pos_idx"],
            batch["home_team_idx"], batch["home_mask"],
            batch["away_player_idx"], batch["away_pos_idx"],
            batch["away_team_idx"], batch["away_mask"],
        )
        x = torch.cat([home_emb, away_emb], dim=1)
        rates = F.softplus(self.goal_head(x)) + 1e-6
        outcome_logits = self.outcome_head(x)
        return rates[:, 0], rates[:, 1], outcome_logits


# ============================================================================
# DATASET
# ============================================================================

class LineupSupervisedDataset(Dataset):
    def __init__(self, frame, player_vocab, team_vocab, config):
        self.frame = frame.reset_index(drop=True)
        self.player_vocab = player_vocab
        self.team_vocab = team_vocab
        self.config = config

        self.home_player_idx, self.home_pos_idx, self.home_mask = self._build_side_tensors("Home")
        self.away_player_idx, self.away_pos_idx, self.away_mask = self._build_side_tensors("Away")

        unk_team = team_vocab["<UNK>"]
        self.home_team_idx = torch.tensor(
            [team_vocab.get(extract_team_id(row, "Home"), unk_team)
             for _, row in self.frame.iterrows()], dtype=torch.long
        )
        self.away_team_idx = torch.tensor(
            [team_vocab.get(extract_team_id(row, "Away"), unk_team)
             for _, row in self.frame.iterrows()], dtype=torch.long
        )

        self.hg = torch.tensor(self.frame["HG"].astype(float).values, dtype=torch.float32)
        self.ag = torch.tensor(self.frame["AG"].astype(float).values, dtype=torch.float32)

        hg_np = self.frame["HG"].values
        ag_np = self.frame["AG"].values
        self.outcome = torch.tensor(
            np.where(hg_np > ag_np, 0, np.where(hg_np == ag_np, 1, 2)),
            dtype=torch.long,
        )

    def _build_side_tensors(self, side):
        player_idx_rows, position_idx_rows, mask_rows = [], [], []
        unk_idx = self.player_vocab["<UNK>"]
        for _, row in self.frame.iterrows():
            player_ids, positions = extract_lineup_players(row, side)
            player_ids, positions, mask = normalize_lineup(
                player_ids, positions, self.config.max_players_per_team
            )
            player_idx_rows.append([self.player_vocab.get(pid, unk_idx) for pid in player_ids])
            position_idx_rows.append([POSITION_TO_INDEX[pos] for pos in positions])
            mask_rows.append(mask)
        return (
            torch.tensor(player_idx_rows, dtype=torch.long),
            torch.tensor(position_idx_rows, dtype=torch.long),
            torch.tensor(mask_rows, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        return {
            "home_player_idx": self.home_player_idx[idx],
            "home_pos_idx": self.home_pos_idx[idx],
            "home_team_idx": self.home_team_idx[idx],
            "home_mask": self.home_mask[idx],
            "away_player_idx": self.away_player_idx[idx],
            "away_pos_idx": self.away_pos_idx[idx],
            "away_team_idx": self.away_team_idx[idx],
            "away_mask": self.away_mask[idx],
            "HG": self.hg[idx],
            "AG": self.ag[idx],
            "outcome": self.outcome[idx],
        }


def collate_lineup_batch(batch):
    return {k: torch.stack([x[k] for x in batch]) for k in batch[0]}


def encode_dataframe(encoder, frame, player_vocab, team_vocab, config):
    encoder.eval()
    dataset = LineupSupervisedDataset(
        frame.assign(HG=0.0, AG=0.0), player_vocab, team_vocab, config
    )
    loader = DataLoader(dataset, batch_size=1024, shuffle=False,
                        collate_fn=collate_lineup_batch)

    home_chunks, away_chunks = [], []
    with torch.no_grad():
        for batch in loader:
            home_emb, away_emb = encoder(
                batch["home_player_idx"], batch["home_pos_idx"],
                batch["home_team_idx"], batch["home_mask"],
                batch["away_player_idx"], batch["away_pos_idx"],
                batch["away_team_idx"], batch["away_mask"],
            )
            home_chunks.append(home_emb.cpu().numpy())
            away_chunks.append(away_emb.cpu().numpy())

    home_array = np.vstack(home_chunks)
    away_array = np.vstack(away_chunks)

    output = pd.DataFrame({"game_id": frame["game_id"].astype(str).values})
    for idx in range(config.output_dim):
        output[f"home_lineup_embedding_{idx + 1}"] = home_array[:, idx]
        output[f"away_lineup_embedding_{idx + 1}"] = away_array[:, idx]
    return output


# ============================================================================
# TRAINING
# ============================================================================

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.05):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_epoch_loss(model, loader, device, aux_weight):
    """Returns mean total loss, mean goal NLL, mean outcome CE, accuracy."""
    model.eval()
    total_losses, goal_losses, outcome_losses = [], [], []
    correct, n_samples = 0, 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            lam_h, lam_a, outcome_logits = model(batch)

            goal_loss = (
                F.poisson_nll_loss(lam_h, batch["HG"], log_input=False, full=True)
                + F.poisson_nll_loss(lam_a, batch["AG"], log_input=False, full=True)
            ).mean()

            outcome_loss = F.cross_entropy(outcome_logits, batch["outcome"])
            total_loss = goal_loss + aux_weight * outcome_loss

            preds = outcome_logits.argmax(dim=1)
            correct += (preds == batch["outcome"]).sum().item()
            n_samples += batch["outcome"].size(0)

            total_losses.append(float(total_loss.item()))
            goal_losses.append(float(goal_loss.item()))
            outcome_losses.append(float(outcome_loss.item()))

    acc = correct / max(1, n_samples)
    return (
        float(np.mean(total_losses)),
        float(np.mean(goal_losses)),
        float(np.mean(outcome_losses)),
        acc,
    )


def train_encoder(train_frame, val_frame, test_frame, player_vocab, team_vocab, config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = LineupSupervisedDataset(train_frame, player_vocab, team_vocab, config)
    val_ds = LineupSupervisedDataset(val_frame, player_vocab, team_vocab, config)
    test_ds = LineupSupervisedDataset(test_frame, player_vocab, team_vocab, config)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=True, collate_fn=collate_lineup_batch)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size,
                            shuffle=False, collate_fn=collate_lineup_batch)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size,
                             shuffle=False, collate_fn=collate_lineup_batch)

    encoder = ImprovedLineupEncoder(player_vocab, team_vocab, config).to(device)
    model = LineupGoalPredictor(encoder=encoder, hidden_dim=config.hidden_dim).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    total_steps = config.epochs * len(train_loader)
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps, config.min_lr_ratio
    )

    best_val_loss = float("inf")
    best_encoder_state = None
    best_epoch = 0
    epochs_no_improve = 0

    print("=" * 70)
    print("TRAINING IMPROVED TRANSFORMER LINEUP ENCODER")
    print("=" * 70)
    print(f"Intra blocks: {config.num_transformer_blocks} | "
          f"Cross blocks: {config.num_cross_team_blocks} | "
          f"Heads: {config.attention_heads} | Hidden: {config.hidden_dim} | "
          f"Output: {config.output_dim} | "
          f"LR: {config.learning_rate} | Aux weight: {config.aux_outcome_weight}")
    print(f"Train rows: {len(train_ds)} | Val rows: {len(val_ds)} | Test rows: {len(test_ds)}")

    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            lam_h, lam_a, outcome_logits = model(batch)

            goal_loss = (
                F.poisson_nll_loss(lam_h, batch["HG"], log_input=False, full=True)
                + F.poisson_nll_loss(lam_a, batch["AG"], log_input=False, full=True)
            ).mean()

            outcome_loss = F.cross_entropy(outcome_logits, batch["outcome"])
            loss = goal_loss + config.aux_outcome_weight * outcome_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
            optimizer.step()
            scheduler.step()

        val_total, val_goal, val_outcome, val_acc = compute_epoch_loss(
            model, val_loader, device, config.aux_outcome_weight
        )

        if val_total < best_val_loss - 1e-4:
            best_val_loss = val_total
            best_epoch = epoch
            epochs_no_improve = 0
            best_encoder_state = {k: v.detach().cpu().clone()
                                  for k, v in encoder.state_dict().items()}
        else:
            epochs_no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"Ep {epoch:03d}/{config.epochs} | "
                  f"Val Goal {val_goal:.4f} | Val Out {val_outcome:.4f} | "
                  f"Val Acc {val_acc:.3f} | LR {lr_now:.2e}")

        if epochs_no_improve >= config.early_stopping_patience:
            print(f"  ⏹ Early stop at epoch {epoch} (best ep {best_epoch})")
            break

    if best_encoder_state is not None:
        encoder.load_state_dict(best_encoder_state)
        print(f"  ✓ Restored best encoder from epoch {best_epoch}")

    # Final test evaluation (once)
    test_total, test_goal, test_outcome, test_acc = compute_epoch_loss(
        model, test_loader, device, config.aux_outcome_weight
    )
    print(f"\n  Test | Goal NLL {test_goal:.4f} | Outcome CE {test_outcome:.4f} | "
          f"Outcome Acc {test_acc:.3f}")

    metrics = {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "epochs_trained": epoch,
        "test_goal_nll": test_goal,
        "test_outcome_ce": test_outcome,
        "test_outcome_acc": test_acc,
        "architecture": "cross_team_transformer_encoder_with_aux_outcome",
        "num_transformer_blocks": config.num_transformer_blocks,
        "num_cross_team_blocks": config.num_cross_team_blocks,
        "attention_heads": config.attention_heads,
        "hidden_dim": config.hidden_dim,
        "output_dim": config.output_dim,
        "player_embedding_dim": config.player_embedding_dim,
        "position_embedding_dim": config.position_embedding_dim,
        "slot_embedding_dim": config.slot_embedding_dim,
        "team_embedding_dim": config.team_embedding_dim,
        "aux_outcome_weight": config.aux_outcome_weight,
        "trained": True,
    }
    return encoder, metrics


# ============================================================================
# MAIN
# ============================================================================

def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    frame = pd.read_csv(INPUT_PATH, low_memory=False)

    if "game_id" not in frame.columns or "Date" not in frame.columns:
        raise KeyError("lineups.csv must contain game_id and Date columns.")

    # --- Load goals ---
    targets = load_match_results(GOALS_SOURCE_PATH)
    print(f"Loaded {len(targets)} match results from {GOALS_SOURCE_PATH.name}")

    frame = frame.merge(targets, on="game_id", how="left")
    before = len(frame)
    frame = frame.dropna(subset=["HG", "AG"]).copy()
    print(f"Dropped {before - len(frame)} rows without HG/AG labels.")

    # --- Drop rows with an empty home or away lineup ---
    before = len(frame)
    frame = frame[
        frame.apply(
            lambda r: side_has_players(r, "Home") and side_has_players(r, "Away"),
            axis=1,
        )
    ].copy()
    print(f"Dropped {before - len(frame)} rows with empty lineups.")

    # --- Split train / test by season ---
    train_full, test_frame = split_train_test(frame)
    print(f"Train rows (full): {len(train_full)} | Test rows: {len(test_frame)}")

    # --- Split train into train / validation (random) ---
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(train_full))
    n_val = int(len(train_full) * 0.1)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    train_frame = train_full.iloc[train_idx].reset_index(drop=True)
    val_frame = train_full.iloc[val_idx].reset_index(drop=True)
    print(f"Train rows: {len(train_frame)} | Val rows: {len(val_frame)}")

    # --- Vocabularies (built ONLY from training data) ---
    player_vocab = build_player_vocab(train_frame)
    team_vocab = build_team_vocab(train_frame)
    print(f"Player vocab size: {len(player_vocab)} | Team vocab size: {len(team_vocab)}")

    config = LineupEncoderConfig()
    encoder, metrics = train_encoder(
        train_frame, val_frame, test_frame, player_vocab, team_vocab, config
    )

    # --- Encode train and test ---
    train_output = encode_dataframe(encoder, train_full, player_vocab, team_vocab, config)
    test_output = encode_dataframe(encoder, test_frame, player_vocab, team_vocab, config)

    FINAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MISC_DIR.mkdir(parents=True, exist_ok=True)

    train_output.to_csv(TRAIN_OUTPUT_PATH, index=False)
    test_output.to_csv(TEST_OUTPUT_PATH, index=False)

    pd.DataFrame({
        "player_id": list(player_vocab.keys()),
        "index": list(player_vocab.values()),
    }).to_csv(VOCAB_PATH, index=False)

    pd.DataFrame({
        "team_id": list(team_vocab.keys()),
        "index": list(team_vocab.values()),
    }).to_csv(TEAM_VOCAB_PATH, index=False)

    torch.save(encoder.state_dict(), STATE_DICT_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nSaved lineup encoder to {STATE_DICT_PATH}")
    print(f"Saved training embeddings to {TRAIN_OUTPUT_PATH} ({train_output.shape})")
    print(f"Saved test embeddings to {TEST_OUTPUT_PATH} ({test_output.shape})")
    print(f"Saved player vocab to {VOCAB_PATH}")
    print(f"Saved team vocab to {TEAM_VOCAB_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()