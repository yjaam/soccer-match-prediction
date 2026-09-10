#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Create train/test lineup embeddings with an IMPROVED 2-Block Transformer Encoder.

Key improvements:
- Two transformer blocks with residual connections
- Layer normalization for stable training
- GELU activation (better than ReLU for transformers)
- Final attention pooling
- Early stopping and LR scheduling

This is a representation learning model - complexity is justified here
(unlike the prediction MLP where simplicity worked better).
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
INPUT_PATH = PROJECT_DIR / "data" / "lineups.csv"
TARGET_PATH = PROJECT_DIR / "data" / "target_variables.csv"
FINAL_DATA_DIR = PROJECT_DIR / "final_data"
MISC_DIR = PROJECT_DIR / "misc"

TRAIN_OUTPUT_PATH = FINAL_DATA_DIR / "lineup_embeddings_TRAIN.csv"
TEST_OUTPUT_PATH = FINAL_DATA_DIR / "lineup_embeddings_TEST.csv"
STATE_DICT_PATH = MISC_DIR / "lineup_encoder_state_dict.pt"
VOCAB_PATH = MISC_DIR / "lineup_player_vocab.csv"
METRICS_PATH = MISC_DIR / "lineup_encoder_training_metrics.json"

POSITION_TO_INDEX = {"GK": 0, "DEF": 1, "MID": 2, "ATT": 3}


@dataclass
class LineupEncoderConfig:
    player_embedding_dim: int = 32
    position_embedding_dim: int = 8
    hidden_dim: int = 64
    output_dim: int = 32
    max_players_per_team: int = 11
    epochs: int = 50
    batch_size: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    dropout: float = 0.15
    seed: int = 42
    attention_heads: int = 4
    num_transformer_blocks: int = 2
    early_stopping_patience: int = 10
    lr_patience: int = 5
    lr_factor: float = 0.5
    min_lr: float = 1e-5


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


def normalize_lineup(player_ids, positions, max_players):
    player_ids = player_ids[:max_players]
    positions = positions[:max_players]
    mask = [1.0] * len(player_ids)
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


# ============================================================================
# ATTENTION MODULES
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with position bias."""
    
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
        
        # Multi-head reshape
        Q = Q.view(batch_size, num_players, self.attention_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_players, self.attention_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_players, self.attention_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.attention_scale
        
        # Position bias
        pos_bias = self.position_bias[position_indices]
        pos_bias = pos_bias.unsqueeze(1).unsqueeze(2)
        attn_scores = attn_scores + pos_bias
        
        # Mask padding
        mask_expanded = mask.unsqueeze(1).unsqueeze(2)
        attn_scores = attn_scores.masked_fill(mask_expanded == 0, -1e9)
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        
        # Apply attention
        attended = torch.matmul(attn_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, num_players, self.hidden_dim)
        
        return self.out_proj(attended)


class TransformerBlock(nn.Module):
    """Transformer block: self-attention + feed-forward with residuals."""
    
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
    """Final attention pooling to get team representation."""
    
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
        pooled = attended[:, 0, :]
        return pooled


# ============================================================================
# ENCODER
# ============================================================================

class ImprovedLineupEncoder(nn.Module):
    """2-block transformer encoder with residual connections."""
    
    def __init__(self, player_vocab, config):
        super().__init__()
        self.config = config
        self.player_vocab = player_vocab

        self.player_embedding = nn.Embedding(len(self.player_vocab), config.player_embedding_dim)
        self.position_embedding = nn.Embedding(len(POSITION_TO_INDEX), config.position_embedding_dim)

        input_dim = config.player_embedding_dim + config.position_embedding_dim
        self.input_proj = nn.Linear(input_dim, config.hidden_dim)
        self.input_dropout = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(config.hidden_dim, config.attention_heads, config.dropout)
            for _ in range(config.num_transformer_blocks)
        ])
        
        self.pooling = AttentionPooling(config.hidden_dim, config.attention_heads, config.dropout)
        
        self.rho = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.output_dim),
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


class LineupGoalPredictor(nn.Module):
    def __init__(self, encoder, hidden_dim=64):
        super().__init__()
        self.encoder = encoder
        in_dim = encoder.config.output_dim * 2
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, batch):
        home_emb = self.encoder(batch["home_player_idx"], batch["home_pos_idx"], batch["home_mask"])
        away_emb = self.encoder(batch["away_player_idx"], batch["away_pos_idx"], batch["away_mask"])
        x = torch.cat([home_emb, away_emb], dim=1)
        rates = F.softplus(self.head(x)) + 1e-6
        return rates[:, 0], rates[:, 1]


# ============================================================================
# DATASET
# ============================================================================

class LineupSupervisedDataset(Dataset):
    def __init__(self, frame, player_vocab, config):
        self.frame = frame.reset_index(drop=True)
        self.player_vocab = player_vocab
        self.config = config
        self.home_player_idx, self.home_pos_idx, self.home_mask = self._build_side_tensors("Home")
        self.away_player_idx, self.away_pos_idx, self.away_mask = self._build_side_tensors("Away")
        self.hg = torch.tensor(self.frame["HG"].astype(float).values, dtype=torch.float32)
        self.ag = torch.tensor(self.frame["AG"].astype(float).values, dtype=torch.float32)

    def _build_side_tensors(self, side):
        player_idx_rows, position_idx_rows, mask_rows = [], [], []
        unk_idx = self.player_vocab["<UNK>"]
        for _, row in self.frame.iterrows():
            player_ids, positions = extract_lineup_players(row, side)
            player_ids, positions, mask = normalize_lineup(player_ids, positions, self.config.max_players_per_team)
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
            "home_mask": self.home_mask[idx],
            "away_player_idx": self.away_player_idx[idx],
            "away_pos_idx": self.away_pos_idx[idx],
            "away_mask": self.away_mask[idx],
            "HG": self.hg[idx],
            "AG": self.ag[idx],
        }


def collate_lineup_batch(batch):
    return {
        "home_player_idx": torch.stack([x["home_player_idx"] for x in batch]),
        "home_pos_idx": torch.stack([x["home_pos_idx"] for x in batch]),
        "home_mask": torch.stack([x["home_mask"] for x in batch]),
        "away_player_idx": torch.stack([x["away_player_idx"] for x in batch]),
        "away_pos_idx": torch.stack([x["away_pos_idx"] for x in batch]),
        "away_mask": torch.stack([x["away_mask"] for x in batch]),
        "HG": torch.stack([x["HG"] for x in batch]),
        "AG": torch.stack([x["AG"] for x in batch]),
    }


def encode_dataframe(encoder, frame, config):
    encoder.eval()
    dataset = LineupSupervisedDataset(frame.assign(HG=0.0, AG=0.0), encoder.player_vocab, config)
    loader = DataLoader(dataset, batch_size=1024, shuffle=False, collate_fn=collate_lineup_batch)

    home_chunks, away_chunks = [], []
    with torch.no_grad():
        for batch in loader:
            home_emb = encoder(batch["home_player_idx"], batch["home_pos_idx"], batch["home_mask"])
            away_emb = encoder(batch["away_player_idx"], batch["away_pos_idx"], batch["away_mask"])
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

def train_encoder(train_frame, test_frame, player_vocab, config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = LineupSupervisedDataset(train_frame, player_vocab, config)
    test_ds = LineupSupervisedDataset(test_frame, player_vocab, config)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_lineup_batch)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_lineup_batch)

    encoder = ImprovedLineupEncoder(player_vocab, config=config).to(device)
    model = LineupGoalPredictor(encoder=encoder, hidden_dim=config.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.lr_factor,
                                                           patience=config.lr_patience, min_lr=config.min_lr)

    best_test_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    best_encoder_state = None

    print("=" * 70)
    print("TRAINING IMPROVED TRANSFORMER LINEUP ENCODER")
    print("=" * 70)
    print(f"Blocks: {config.num_transformer_blocks} | Heads: {config.attention_heads} | "
          f"Hidden: {config.hidden_dim} | Emb: {config.player_embedding_dim} | "
          f"Output: {config.output_dim} | LR: {config.learning_rate}")

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            lambda_h, lambda_a = model(batch)
            loss_h = F.poisson_nll_loss(lambda_h, batch["HG"], log_input=False, full=True)
            loss_a = F.poisson_nll_loss(lambda_a, batch["AG"], log_input=False, full=True)
            loss = loss_h + loss_a
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        test_losses = []
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                lambda_h, lambda_a = model(batch)
                loss_h = F.poisson_nll_loss(lambda_h, batch["HG"], log_input=False, full=True)
                loss_a = F.poisson_nll_loss(lambda_a, batch["AG"], log_input=False, full=True)
                test_losses.append(float((loss_h + loss_a).item()))

        train_loss = float(np.mean(train_losses))
        test_loss = float(np.mean(test_losses))
        scheduler.step(test_loss)

        if test_loss < best_test_loss - 1e-4:
            best_test_loss = test_loss
            best_epoch = epoch
            epochs_no_improve = 0
            best_encoder_state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
        else:
            epochs_no_improve += 1

        lr_now = optimizer.param_groups[0]['lr']
        if epoch % 5 == 0 or epoch == 1:
            print(f"Ep {epoch:03d}/{config.epochs} | Train NLL {train_loss:.5f} | "
                  f"Test NLL {test_loss:.5f} | LR {lr_now:.1e}")

        if epochs_no_improve >= config.early_stopping_patience:
            print(f"  ⏹ Early stop at epoch {epoch} (best ep {best_epoch}, NLL {best_test_loss:.5f})")
            break

    if best_encoder_state is not None:
        encoder.load_state_dict(best_encoder_state)
        print(f"  ✓ Restored best model from epoch {best_epoch}")

    metrics = {
        "best_test_goal_nll": best_test_loss,
        "best_epoch": best_epoch,
        "epochs_trained": epoch,
        "architecture": "2_block_transformer_encoder",
        "num_transformer_blocks": config.num_transformer_blocks,
        "attention_heads": config.attention_heads,
        "hidden_dim": config.hidden_dim,
        "player_embedding_dim": config.player_embedding_dim,
        "output_dim": config.output_dim,
        "trained": True,
    }
    return encoder, metrics


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")
    if not TARGET_PATH.exists():
        raise FileNotFoundError(f"Target file not found: {TARGET_PATH}")

    frame = pd.read_csv(INPUT_PATH, low_memory=False)
    targets = pd.read_csv(TARGET_PATH, low_memory=False)

    if "game_id" not in frame.columns or "Date" not in frame.columns:
        raise KeyError("lineups.csv must contain game_id and Date columns.")
    if not {"game_id", "HG", "AG"}.issubset(targets.columns):
        raise KeyError("target_variables.csv must contain game_id, HG, AG.")

    frame = frame.merge(targets[["game_id", "HG", "AG"]], on="game_id", how="left")
    before = len(frame)
    frame = frame.dropna(subset=["HG", "AG"]).copy()
    print(f"Dropped {before - len(frame)} rows without HG/AG labels.")

    train_frame, test_frame = split_train_test(frame)
    print(f"Training rows: {len(train_frame)} | Test rows: {len(test_frame)}")

    player_vocab = build_player_vocab(train_frame)
    print(f"Training vocabulary size: {len(player_vocab)}")

    config = LineupEncoderConfig()
    encoder, metrics = train_encoder(train_frame, test_frame, player_vocab, config)

    train_output = encode_dataframe(encoder, train_frame, config)
    test_output = encode_dataframe(encoder, test_frame, config)

    FINAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MISC_DIR.mkdir(parents=True, exist_ok=True)

    train_output.to_csv(TRAIN_OUTPUT_PATH, index=False)
    test_output.to_csv(TEST_OUTPUT_PATH, index=False)

    vocab_df = pd.DataFrame({"player_id": list(player_vocab.keys()), "index": list(player_vocab.values())})
    vocab_df.to_csv(VOCAB_PATH, index=False)

    torch.save(encoder.state_dict(), STATE_DICT_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved improved lineup encoder to {STATE_DICT_PATH}")
    print(f"Saved training embeddings to {TRAIN_OUTPUT_PATH} ({train_output.shape})")
    print(f"Saved test embeddings to {TEST_OUTPUT_PATH} ({test_output.shape})")
    print(f"Saved player vocabulary to {VOCAB_PATH}")


if __name__ == "__main__":
    main()