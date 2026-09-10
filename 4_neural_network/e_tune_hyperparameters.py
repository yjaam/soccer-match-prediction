#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Random hyperparameter search for d_train_RNN_LSTM.py.

Reuses the exact model/dataset classes and train_module/train_fusion
functions from d_train_RNN_LSTM.py (the architecture itself is never
touched here) and searches over optimization/regularization/capacity
hyperparameters, running short (reduced-epoch) trials to keep the search
tractable. The winning fusion validation accuracy determines the best trial.

Usage:
    python 4_neural_network/e_tune_hyperparameters.py [--trials N]

After running, copy the printed/saved best hyperparameters into
RNNFusionConfig's defaults in d_train_RNN_LSTM.py.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict
from pathlib import Path

import torch

import d_train_RNN_LSTM as rnn_lstm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MISC_DIR = PROJECT_DIR / "misc"
BEST_PARAMS_PATH = MISC_DIR / "rnn_lstm_best_hyperparams.json"
SEARCH_LOG_PATH = MISC_DIR / "rnn_lstm_hyperparam_search_log.json"

# Search space: sampled uniformly at random per trial.
SEARCH_SPACE = {
    "learning_rate": [1e-4, 3e-4, 5e-4, 1e-3],
    "weight_decay": [1e-5, 1e-4, 1e-3],
    "dropout": [0.1, 0.15, 0.2, 0.3],
    "module_hidden_dim": [16, 32, 48],
    "lineup_hidden_dim": [64, 128, 192],
    "fusion_hidden_dim": [16, 32, 48],
    "rate_head_hidden_dim": [8, 16, 32],
    "outcome_weight": [0.5, 1.0, 1.5],
}

# Reduced-epoch budget so each trial stays fast; final training script uses the full budget.
TRIAL_MODULE_EPOCHS = 15
TRIAL_FUSION_EPOCHS = 15
TRIAL_EARLY_STOPPING_PATIENCE = 5


def sample_config(base_config: rnn_lstm.RNNFusionConfig, rng: random.Random) -> rnn_lstm.RNNFusionConfig:
    config = copy.deepcopy(base_config)
    for field, choices in SEARCH_SPACE.items():
        setattr(config, field, rng.choice(choices))
    config.module_epochs = TRIAL_MODULE_EPOCHS
    config.fusion_epochs = TRIAL_FUSION_EPOCHS
    config.early_stopping_patience = TRIAL_EARLY_STOPPING_PATIENCE
    return config


def run_trial(config: rnn_lstm.RNNFusionConfig, device: torch.device) -> dict:
    rnn_lstm.set_seed(config.seed)
    for module_name in rnn_lstm.MODULES:
        rnn_lstm.train_module(module_name, config, device, verbose=False)
    fusion_result = rnn_lstm.train_fusion(config, device, verbose=False)
    return fusion_result


def main():
    parser = argparse.ArgumentParser(description="Random hyperparameter search for the RNN/LSTM multimodal model.")
    parser.add_argument("--trials", type=int, default=12, help="Number of random trials to run (default: 12).")
    parser.add_argument("--search-seed", type=int, default=0, help="Seed for sampling hyperparameter combinations.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_config = rnn_lstm.RNNFusionConfig()
    rng = random.Random(args.search_seed)

    print("=" * 70)
    print(f"HYPERPARAMETER SEARCH ({args.trials} trials, device={device})")
    print(f"Trial budget: module_epochs={TRIAL_MODULE_EPOCHS}, fusion_epochs={TRIAL_FUSION_EPOCHS}, "
          f"early_stopping_patience={TRIAL_EARLY_STOPPING_PATIENCE}")
    print("=" * 70)

    trial_log = []
    best_trial = None
    best_val_acc = -1.0

    for trial_idx in range(1, args.trials + 1):
        config = sample_config(base_config, rng)
        sampled = {field: getattr(config, field) for field in SEARCH_SPACE}
        print(f"\nTrial {trial_idx:02d}/{args.trials} | {sampled}")

        try:
            result = run_trial(config, device)
        except Exception as e:
            print(f"  ❌ Trial failed: {e}")
            trial_log.append({"trial": trial_idx, "hyperparameters": sampled, "error": str(e)})
            continue

        val_acc = result["best_val_accuracy"]
        test_acc = result["best_test_accuracy"]
        print(f"  Fusion ValAcc {val_acc:.3f} | Fusion TestAcc {test_acc:.3f}")

        trial_log.append({
            "trial": trial_idx,
            "hyperparameters": sampled,
            "fusion_val_accuracy": val_acc,
            "fusion_test_accuracy": test_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_trial = {"hyperparameters": sampled, "fusion_val_accuracy": val_acc, "fusion_test_accuracy": test_acc}
            print(f"  ✓ New best (val acc {best_val_acc:.3f})")

    print("\n" + "=" * 70)
    print("SEARCH COMPLETE")
    print("=" * 70)
    if best_trial is None:
        print("No successful trials.")
        return

    print(f"Best hyperparameters (fusion val acc {best_trial['fusion_val_accuracy']:.3f}, "
          f"test acc {best_trial['fusion_test_accuracy']:.3f}):")
    for field, value in best_trial["hyperparameters"].items():
        print(f"  {field} = {value}")

    MISC_DIR.mkdir(parents=True, exist_ok=True)
    BEST_PARAMS_PATH.write_text(json.dumps(best_trial, indent=2), encoding="utf-8")
    SEARCH_LOG_PATH.write_text(json.dumps({"trials": trial_log, "base_config": asdict(base_config)}, indent=2), encoding="utf-8")
    print(f"\nSaved best hyperparameters to {BEST_PARAMS_PATH}")
    print(f"Saved full search log to {SEARCH_LOG_PATH}")
    print("\nNext step: copy these values into RNNFusionConfig's defaults in d_train_RNN_LSTM.py, "
          "then rerun the full training script with its full epoch budget.")


if __name__ == "__main__":
    main()
