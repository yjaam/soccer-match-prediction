#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train an XGBoost baseline on the existing prepared module database.

This script uses the already-generated `final_data/nn_input_*_{TRAIN,TEST}.csv`
files, merges them on `game_id`, and trains two Poisson XGBoost regressors:
one for home goals and one for away goals.

The model stays intentionally shallow/small to avoid overfitting and includes a
validation split for early stopping plus draw-margin tuning.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
FINAL_DATA_DIR = PROJECT_DIR / "final_data"
RESULT_DATA_DIR = PROJECT_DIR / "result_data"
MISC_DIR = PROJECT_DIR / "misc"
META_PATH = MISC_DIR / "nn_module_metadata.json"

SUMMARY_PATH = RESULT_DATA_DIR / "xgboost_summary.json"
MODEL_HOME_PATH = RESULT_DATA_DIR / "xgboost_home_model.json"
MODEL_AWAY_PATH = RESULT_DATA_DIR / "xgboost_away_model.json"
PRED_PATH = RESULT_DATA_DIR / "xgboost_test_predictions.csv"

DEFAULT_MODULES = ["lineup", "player_stats", "skill_matchups", "match_stats", "league", "season"]
OPTIONAL_MATCHUP_MODULE = "matchup"


@dataclass
class Config:
	modules: list[str]
	include_matchup: bool = False
	val_fraction: float = 0.15
	seed: int = 42
	learning_rate: float = 0.05
	max_depth: int = 3
	min_child_weight: float = 4.0
	subsample: float = 0.8
	colsample_bytree: float = 0.8
	reg_alpha: float = 0.0
	reg_lambda: float = 1.0
	gamma: float = 0.0
	n_estimators: int = 3000
	early_stopping_rounds: int = 75
	draw_margin_grid: tuple[float, ...] = tuple(np.round(np.linspace(0.0, 0.8, 41), 3))


def parse_args() -> Config:
	parser = argparse.ArgumentParser(description="Train an XGBoost baseline on prepared match features.")
	parser.add_argument(
		"--include-matchup",
		action="store_true",
		help="Include the matchup embedding module in addition to the default core/context modules.",
	)
	parser.add_argument(
		"--val-fraction",
		type=float,
		default=0.15,
		help="Fraction of the training set used for validation and draw-margin tuning.",
	)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--learning-rate", type=float, default=0.05)
	parser.add_argument("--max-depth", type=int, default=3)
	parser.add_argument("--min-child-weight", type=float, default=4.0)
	parser.add_argument("--subsample", type=float, default=0.8)
	parser.add_argument("--colsample-bytree", type=float, default=0.8)
	parser.add_argument("--reg-alpha", type=float, default=0.0)
	parser.add_argument("--reg-lambda", type=float, default=1.0)
	parser.add_argument("--gamma", type=float, default=0.0)
	parser.add_argument("--n-estimators", type=int, default=3000)
	parser.add_argument("--early-stopping-rounds", type=int, default=75)
	args = parser.parse_args()

	modules = list(DEFAULT_MODULES)
	if args.include_matchup:
		modules.append(OPTIONAL_MATCHUP_MODULE)

	return Config(
		modules=modules,
		include_matchup=args.include_matchup,
		val_fraction=args.val_fraction,
		seed=args.seed,
		learning_rate=args.learning_rate,
		max_depth=args.max_depth,
		min_child_weight=args.min_child_weight,
		subsample=args.subsample,
		colsample_bytree=args.colsample_bytree,
		reg_alpha=args.reg_alpha,
		reg_lambda=args.reg_lambda,
		gamma=args.gamma,
		n_estimators=args.n_estimators,
		early_stopping_rounds=args.early_stopping_rounds,
	)


def set_seed(seed: int) -> None:
	np.random.seed(seed)


def load_module_metadata() -> dict:
	if not META_PATH.exists():
		raise FileNotFoundError(f"Missing metadata file: {META_PATH}")
	return json.loads(META_PATH.read_text(encoding="utf-8"))


def _load_split_frame(module_name: str, split: str) -> pd.DataFrame:
	path = FINAL_DATA_DIR / f"nn_input_{module_name}_{split}.csv"
	if not path.exists():
		raise FileNotFoundError(f"Missing prepared input file: {path}")
	return pd.read_csv(path, low_memory=False)


def _feature_columns(frame: pd.DataFrame) -> list[str]:
	exclude = {"game_id", "HG", "AG", "match_id"}
	return [col for col in frame.columns if col not in exclude]


def load_merged_split(split: str, modules: list[str]) -> pd.DataFrame:
	merged = None

	for module_name in modules:
		frame = _load_split_frame(module_name, split)
		if "game_id" not in frame.columns:
			raise KeyError(f"{module_name} {split} file is missing game_id")
		if "HG" not in frame.columns or "AG" not in frame.columns:
			raise KeyError(f"{module_name} {split} file is missing HG/AG")

		frame = frame.copy().drop_duplicates(subset=["game_id"], keep="first")
		feature_cols = _feature_columns(frame)
		keep_cols = ["game_id", "HG", "AG"] + feature_cols
		frame = frame[keep_cols]

		if merged is None:
			merged = frame
			continue

		merged = merged.merge(frame, on=["game_id", "HG", "AG"], how="inner")

	if merged is None or merged.empty:
		raise ValueError("No merged rows available after combining module inputs.")

	merged = merged.replace([np.inf, -np.inf], np.nan)
	merged = merged.dropna(subset=["HG", "AG"]).copy()
	merged = merged.sort_values("game_id").reset_index(drop=True)
	return merged


def split_train_val(frame: pd.DataFrame, val_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
	if not 0 < val_fraction < 1:
		raise ValueError("val_fraction must be between 0 and 1.")

	frame = frame.sort_values("game_id").reset_index(drop=True)
	split_idx = int(len(frame) * (1 - val_fraction))
	split_idx = min(max(split_idx, 1), len(frame) - 1)
	return frame.iloc[:split_idx].copy(), frame.iloc[split_idx:].copy()


def to_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
	feature_cols = [col for col in frame.columns if col not in {"game_id", "HG", "AG"}]
	matrix = frame.copy()
	for col in feature_cols:
		matrix[col] = pd.to_numeric(matrix[col], errors="coerce")
	return matrix[feature_cols], feature_cols


def fit_poisson_regressor(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    config: Config,
) -> XGBRegressor:
    model = XGBRegressor(
        objective="count:poisson",
        eval_metric="poisson-nloglik",
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        min_child_weight=config.min_child_weight,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        gamma=config.gamma,
        tree_method="hist",
        random_state=config.seed,
        n_jobs=-1,
        early_stopping_rounds=config.early_stopping_rounds,  # MOVED HERE
    )

    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        verbose=False,
    )
    return model


def poisson_nll(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	y_pred = np.clip(y_pred, 1e-6, None)
	return float(np.mean(y_pred - y_true * np.log(y_pred)))


def tune_draw_margin(
	lambda_home: np.ndarray,
	lambda_away: np.ndarray,
	true_home: np.ndarray,
	true_away: np.ndarray,
	margins: tuple[float, ...],
) -> tuple[float, float]:
	true_outcome = np.where(true_home > true_away, 0, np.where(true_home == true_away, 1, 2))
	best_margin = 0.0
	best_accuracy = -1.0

	for margin in margins:
		pred_outcome = np.where(
			lambda_home > lambda_away + margin,
			0,
			np.where(lambda_away > lambda_home + margin, 2, 1),
		)
		acc = accuracy_score(true_outcome, pred_outcome)
		if acc > best_accuracy:
			best_accuracy = acc
			best_margin = float(margin)

	return best_margin, best_accuracy


def evaluate_predictions(
	frame: pd.DataFrame,
	lambda_home: np.ndarray,
	lambda_away: np.ndarray,
	draw_margin: float,
) -> dict:
	true_outcome = np.where(frame["HG"].values > frame["AG"].values, 0, np.where(frame["HG"].values == frame["AG"].values, 1, 2))
	pred_outcome = np.where(
		lambda_home > lambda_away + draw_margin,
		0,
		np.where(lambda_away > lambda_home + draw_margin, 2, 1),
	)

	return {
		"accuracy": float(accuracy_score(true_outcome, pred_outcome)),
		"confusion_matrix": confusion_matrix(true_outcome, pred_outcome, labels=[0, 1, 2]).tolist(),
		"mae_home": float(mean_absolute_error(frame["HG"].values, lambda_home)),
		"mae_away": float(mean_absolute_error(frame["AG"].values, lambda_away)),
		"mse_home": float(mean_squared_error(frame["HG"].values, lambda_home)),
		"mse_away": float(mean_squared_error(frame["AG"].values, lambda_away)),
		"poisson_nll_home": poisson_nll(frame["HG"].values, lambda_home),
		"poisson_nll_away": poisson_nll(frame["AG"].values, lambda_away),
	}


def main() -> None:
	config = parse_args()
	set_seed(config.seed)
	RESULT_DATA_DIR.mkdir(parents=True, exist_ok=True)

	print("=" * 70)
	print("TRAINING XGBOOST BASELINE")
	print("=" * 70)
	print(f"Modules: {config.modules}")
	print(f"Validation fraction: {config.val_fraction:.0%}")
	print(f"Objective: count:poisson for home/away goals")

	load_module_metadata()

	train_full = load_merged_split("TRAIN", config.modules)
	test_df = load_merged_split("TEST", config.modules)
	train_df, val_df = split_train_val(train_full, config.val_fraction)

	x_train, feature_cols = to_matrix(train_df)
	x_val, _ = to_matrix(val_df)
	x_test, _ = to_matrix(test_df)

	train_y_home = pd.to_numeric(train_df["HG"], errors="coerce")
	train_y_away = pd.to_numeric(train_df["AG"], errors="coerce")
	val_y_home = pd.to_numeric(val_df["HG"], errors="coerce")
	val_y_away = pd.to_numeric(val_df["AG"], errors="coerce")

	print(f"Train rows: {len(train_df)} | Val rows: {len(val_df)} | Test rows: {len(test_df)}")
	print(f"Feature count: {len(feature_cols)}")

	home_model = fit_poisson_regressor(x_train, train_y_home, x_val, val_y_home, config)
	away_model = fit_poisson_regressor(x_train, train_y_away, x_val, val_y_away, config)

	val_home_pred = home_model.predict(x_val)
	val_away_pred = away_model.predict(x_val)
	draw_margin, val_accuracy = tune_draw_margin(
		val_home_pred,
		val_away_pred,
		val_df["HG"].values,
		val_df["AG"].values,
		config.draw_margin_grid,
	)

	test_home_pred = np.clip(home_model.predict(x_test), 1e-6, None)
	test_away_pred = np.clip(away_model.predict(x_test), 1e-6, None)
	metrics = evaluate_predictions(test_df, test_home_pred, test_away_pred, draw_margin)

	pred_frame = pd.DataFrame(
		{
			"game_id": test_df["game_id"].values,
			"HG": test_df["HG"].values,
			"AG": test_df["AG"].values,
			"lambda_home": test_home_pred,
			"lambda_away": test_away_pred,
			"expected_goal_diff": test_home_pred - test_away_pred,
		}
	)
	pred_frame.to_csv(PRED_PATH, index=False)

	home_model.save_model(MODEL_HOME_PATH)
	away_model.save_model(MODEL_AWAY_PATH)

	summary = {
		"config": {
			"modules": config.modules,
			"include_matchup": config.include_matchup,
			"val_fraction": config.val_fraction,
			"seed": config.seed,
			"learning_rate": config.learning_rate,
			"max_depth": config.max_depth,
			"min_child_weight": config.min_child_weight,
			"subsample": config.subsample,
			"colsample_bytree": config.colsample_bytree,
			"reg_alpha": config.reg_alpha,
			"reg_lambda": config.reg_lambda,
			"gamma": config.gamma,
			"n_estimators": config.n_estimators,
			"early_stopping_rounds": config.early_stopping_rounds,
		},
		"validation_accuracy": val_accuracy,
		"draw_margin": draw_margin,
		"test_metrics": metrics,
		"feature_count": len(feature_cols),
		"train_rows": len(train_df),
		"val_rows": len(val_df),
		"test_rows": len(test_df),
	}
	SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

	print("\nValidation draw margin tuned to:", draw_margin)
	print(f"Validation accuracy: {val_accuracy:.3f}")
	print(f"Test accuracy: {metrics['accuracy']:.3f}")
	print("Confusion matrix (rows=true, cols=predicted; order home/draw/away):")
	print(np.array(metrics["confusion_matrix"]))
	print(f"MAE home/away: {metrics['mae_home']:.3f} / {metrics['mae_away']:.3f}")
	print(f"Saved predictions to: {PRED_PATH}")
	print(f"Saved models to: {MODEL_HOME_PATH} and {MODEL_AWAY_PATH}")
	print(f"Saved summary to: {SUMMARY_PATH}")


if __name__ == "__main__":
	main()
