#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build a leakage-safe match-statistics dataset from data/big5_matches.csv.

For each numeric match-stat feature, values are derived as the mean over the
past k matches of the same team on the same side (home or away), excluding the
current match. This ensures inputs are available before kickoff.

Feature filtering:
    A feature is only kept if it is populated in every training season
    (season != "2025-2026") at a rate >= MIN_SEASON_COVERAGE (default 50%).
    This drops features that FBref stopped reporting in recent seasons,
    preventing the pipeline from relying on imputed historical means.

After this historical transformation, the script imputes remaining gaps. With
PERFORM_PCA = False the raw features are saved as the TRAIN/TEST datasets; with
PERFORM_PCA = True a PCA fit on the training split reduces them first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# True: reduce the match statistics with PCA (fit on training data only).
# False: save the raw imputed features; folder 4 then uses PCsInit for this module.
PERFORM_PCA = False

# Variance retained when PERFORM_PCA is True.
PCA_VARIANCE_THRESHOLD = 0.95


# Number of past side-specific matches used to compute pre-match means.
K_PAST_MATCHES = 5

# Minimum fraction of non-null cells required in EVERY training season for a
# feature to be kept. Features that fail in even one season are dropped, so the
# model never sees features that were silently imputed across whole seasons.
MIN_SEASON_COVERAGE = 0.50

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
INPUT_PATH = PROJECT_DIR / "data" / "big5_matches_fixed.csv"
FINAL_DATA_DIR = PROJECT_DIR / "final_data"
MISC_DIR = PROJECT_DIR / "misc"

TRAIN_OUTPUT_PATH = FINAL_DATA_DIR / "match_statistics_pca_TRAIN.csv"
TEST_OUTPUT_PATH = FINAL_DATA_DIR / "match_statistics_pca_TEST.csv"
FEATURE_MATRIX_PATH = PROJECT_DIR / "data" / "match_statistics_before_pca.csv"
TEST_FEATURE_MATRIX_PATH = PROJECT_DIR / "data" / "match_statistics_before_pca_TEST.csv"
PCA_WEIGHTS_PATH = MISC_DIR / "match_statistics_pca_weights.json"

# Tells folder 4 whether this module's inputs were already PCA-reduced (PCsInit is used if not).
PCA_STATUS_PATH = MISC_DIR / "pca_status.json"


def update_pca_status(module_name: str, pca_applied: bool) -> None:
	status = {}
	if PCA_STATUS_PATH.exists():
		try:
			status = json.loads(PCA_STATUS_PATH.read_text(encoding="utf-8"))
		except json.JSONDecodeError:
			status = {}
	status[module_name] = {"pca_applied": bool(pca_applied)}
	PCA_STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")


EXCLUDE_EXACT = {
	"Home_goals",
	"Away_goals",
}

EXCLUDE_SUBSTRINGS = (
	"shirtnumber",
	"_age",
	"_minutes",
	"nationality",
	"position",
)


def is_feature_column(column_name: str) -> bool:
	if column_name in EXCLUDE_EXACT:
		return False
	lowered = column_name.lower()
	return not any(token in lowered for token in EXCLUDE_SUBSTRINGS)


def infer_team_group_column(feature_name: str) -> str:
	lowered = feature_name.lower()
	if lowered.startswith("home_"):
		return "home_team"
	if lowered.startswith("away_"):
		return "away_team"
	return "away_team"


def parse_game_date_from_id(game_id: str) -> pd.Timestamp:
	parts = str(game_id).split("_", 2)
	if len(parts) < 3:
		return pd.NaT
	return pd.to_datetime(parts[0], format="%Y%m%d", errors="coerce")


def filter_features_by_season_coverage(
	frame: pd.DataFrame,
	feature_columns: list[str],
	min_coverage: float = MIN_SEASON_COVERAGE,
) -> tuple[list[str], pd.DataFrame]:
	"""Keep only features that are populated (>= min_coverage) in every season.

	Only seasons other than the held-out test season are considered. Returns
	(kept_features, coverage_report) where coverage_report summarises per-season
	coverage for every candidate feature.
	"""

	if "season" not in frame.columns:
		raise KeyError("season column is required for per-season coverage filtering.")

	train_frame = frame[frame["season"] != "2025-2026"].copy()
	if train_frame.empty:
		raise ValueError("No training seasons available for coverage filtering.")

	seasons = sorted(train_frame["season"].dropna().unique().tolist())

	report_rows = []
	kept_features = []

	for column in feature_columns:
		per_season = {}
		col_numeric = pd.to_numeric(train_frame[column], errors="coerce")
		train_frame_with_col = train_frame.assign(_value=col_numeric)

		for season in seasons:
			mask = train_frame_with_col["season"] == season
			season_values = train_frame_with_col.loc[mask, "_value"]
			if len(season_values) == 0:
				per_season[season] = float("nan")
				continue
			per_season[season] = float(season_values.notna().mean())

		# Keep feature only if every season clears the threshold
		season_coverages = [v for v in per_season.values() if pd.notna(v)]
		min_season_coverage_observed = min(season_coverages) if season_coverages else 0.0
		passes = min_season_coverage_observed >= min_coverage

		if passes:
			kept_features.append(column)

		row = {
			"feature": column,
			"min_season_coverage": min_season_coverage_observed,
			"passes_filter": passes,
		}
		for season in seasons:
			row[f"coverage_{season}"] = per_season.get(season)
		report_rows.append(row)

	return kept_features, pd.DataFrame(report_rows)


def derive_historical_features(frame: pd.DataFrame, feature_columns: list[str], k: int) -> pd.DataFrame:
	"""Create leakage-safe pre-match features from past-k side-specific matches."""

	if k < 1:
		raise ValueError("k must be >= 1")

	derived = frame.copy()
	derived["_match_date"] = derived["game_id"].apply(parse_game_date_from_id)
	derived = derived.sort_values(["_match_date", "game_id"]).reset_index(drop=True)

	for column in feature_columns:
		team_column = infer_team_group_column(column)
		derived[column] = pd.to_numeric(derived[column], errors="coerce")

		# Mean over the previous k matches for this team on this side (home/away).
		derived[column] = derived.groupby(team_column, sort=False)[column].transform(
			lambda s: s.shift(1).rolling(window=k, min_periods=1).mean()
		)

	return derived.drop(columns=["_match_date"])


def team_mean_impute(frame: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Impute missing values with the mean of the respective team."""

	imputed = frame[feature_columns].copy()
	missing_report = []

	for column in feature_columns:
		team_column = infer_team_group_column(column)
		imputed[column] = pd.to_numeric(imputed[column], errors="coerce")

		missing_before = int(imputed[column].isna().sum())
		team_means = frame.groupby(team_column)[column].transform(lambda s: pd.to_numeric(s, errors="coerce").mean())
		imputed[column] = imputed[column].fillna(team_means)

		global_mean = imputed[column].mean()
		imputed[column] = imputed[column].fillna(global_mean)
		missing_after = int(imputed[column].isna().sum())

		missing_report.append(
			{
				"feature": column,
				"missing_before": missing_before,
				"missing_after_imputation": missing_after,
				"team_group": team_column,
				"global_mean_used": pd.notna(global_mean),
			}
		)

	return imputed, pd.DataFrame(missing_report)


def team_mean_impute_with_reference(
	frame: pd.DataFrame,
	feature_columns: list[str],
	reference_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Impute missing values using team means learned from a reference frame."""

	imputed = frame[feature_columns].copy()
	missing_report = []

	for column in feature_columns:
		team_column = infer_team_group_column(column)
		imputed[column] = pd.to_numeric(imputed[column], errors="coerce")

		missing_before = int(imputed[column].isna().sum())

		team_means_lookup = (
			reference_frame[[team_column, column]]
			.assign(_value=lambda x: pd.to_numeric(x[column], errors="coerce"))
			.groupby(team_column)["_value"]
			.mean()
		)
		team_means = frame[team_column].map(team_means_lookup)
		imputed[column] = imputed[column].fillna(team_means)

		global_mean = pd.to_numeric(reference_frame[column], errors="coerce").mean()
		imputed[column] = imputed[column].fillna(global_mean)
		missing_after = int(imputed[column].isna().sum())

		missing_report.append(
			{
				"feature": column,
				"missing_before": missing_before,
				"missing_after_imputation": missing_after,
				"team_group": team_column,
				"global_mean_used": pd.notna(global_mean),
			}
		)

	return imputed, pd.DataFrame(missing_report)


def pca_reduce(
	train_matrix: pd.DataFrame,
	test_matrix: pd.DataFrame,
	variance_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
	"""Standardise and project onto the leading principal components (fit on train only)."""

	means = train_matrix.mean(axis=0)
	stds = train_matrix.std(axis=0, ddof=0).replace(0, np.nan)

	train_scaled = ((train_matrix - means) / stds).fillna(0.0).to_numpy(dtype=float)
	test_scaled = ((test_matrix - means) / stds).fillna(0.0).to_numpy(dtype=float)

	if train_scaled.shape[0] < 2:
		raise ValueError("At least two training rows are required to fit PCA.")

	_, singular_values, vt = np.linalg.svd(train_scaled, full_matrices=False)
	variance = singular_values ** 2
	total = variance.sum()
	n_components = 1
	if total > 0:
		n_components = int(np.searchsorted(np.cumsum(variance) / total, variance_threshold)) + 1
	n_components = min(n_components, vt.shape[0])
	components = vt[:n_components]

	columns = [f"match_stats_pc_{i + 1}" for i in range(n_components)]
	train_scores = pd.DataFrame(train_scaled @ components.T, columns=columns)
	test_scores = pd.DataFrame(test_scaled @ components.T, columns=columns)

	weights = {
		"feature_columns": list(train_matrix.columns),
		"means": {c: float(means[c]) for c in train_matrix.columns},
		"stds": {c: (float(stds[c]) if pd.notna(stds[c]) else 1.0) for c in train_matrix.columns},
		"components": components.tolist(),
		"explained_variance_ratio": (variance[:n_components] / total).tolist() if total > 0 else [],
	}
	return train_scores, test_scores, weights


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	train_df = df[df["season"] != "2025-2026"].copy()
	test_df = df[df["season"] == "2025-2026"].copy()
	if train_df.empty:
		raise ValueError("Training split is empty. Check the season labels in big5_matches.csv.")
	if test_df.empty:
		raise ValueError("Test split is empty. No 2025-2026 rows were found in big5_matches.csv.")
	return train_df, test_df


def main() -> None:
	parser = argparse.ArgumentParser(description="Build leakage-safe match-statistics datasets.")
	parser.add_argument(
		"--k",
		type=int,
		default=K_PAST_MATCHES,
		help=f"Number of past side-specific matches used to compute pre-match means (default: {K_PAST_MATCHES}).",
	)
	parser.add_argument(
		"--min-season-coverage",
		type=float,
		default=MIN_SEASON_COVERAGE,
		help=f"Minimum per-season coverage required for a feature to be kept (default: {MIN_SEASON_COVERAGE}).",
	)
	args = parser.parse_args()

	if not INPUT_PATH.exists():
		print(f"Error: input file not found at {INPUT_PATH}")
		return

	FINAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
	MISC_DIR.mkdir(parents=True, exist_ok=True)

	df = pd.read_csv(INPUT_PATH, low_memory=False)
	print(f"Loaded {len(df)} rows and {df.shape[1]} columns from {INPUT_PATH.name}.")

	if "match_id" not in df.columns:
		raise KeyError("match_id column is required.")
	if "game_id" not in df.columns:
		raise KeyError("game_id column is required. Run 2_match_statistics/b_id_matches.py first.")

	rows_before_id_filter = len(df)
	df = df.dropna(subset=["game_id"]).copy()
	print(f"Dropped {rows_before_id_filter - len(df)} rows without game_id.")
	if "Home_goals" not in df.columns or "Away_goals" not in df.columns:
		raise KeyError("Home_goals and Away_goals are required target columns.")

	feature_columns = [column for column in df.columns if is_feature_column(column)]
	feature_columns = [
		column
		for column in feature_columns
		if column not in {"league", "season", "url", "match_id", "game_id", "home_team", "away_team"}
	]

	if not feature_columns:
		raise ValueError("No usable numeric match-stat columns were found.")

	# ------------------------------------------------------------------
	# Per-season coverage filter (drops features abandoned in recent seasons)
	# ------------------------------------------------------------------
	print(f"\nApplying per-season coverage filter (min per-season coverage: "
	      f"{args.min_season_coverage:.0%})...")

	kept_features, coverage_report = filter_features_by_season_coverage(
		df,
		feature_columns,
		min_coverage=args.min_season_coverage,
	)

	dropped_features = sorted(set(feature_columns) - set(kept_features))

	print(f"  Candidates:        {len(feature_columns)}")
	print(f"  Kept:              {len(kept_features)}")
	print(f"  Dropped:           {len(dropped_features)}")
	if dropped_features:
		print(f"  Dropped features:")
		for column in dropped_features:
			row = coverage_report.loc[coverage_report["feature"] == column].iloc[0]
			print(f"    - {column}  (min per-season coverage: {row['min_season_coverage']:.2%})")

	coverage_report.sort_values("min_season_coverage", ascending=True).to_csv(
		MISC_DIR / "match_statistics_season_coverage_report.csv", index=False
	)
	print(f"  Saved coverage report to: {MISC_DIR / 'match_statistics_season_coverage_report.csv'}")

	if not kept_features:
		raise ValueError("All candidate features were dropped by the coverage filter.")

	feature_columns = kept_features

	# Replace raw match statistics with leakage-safe historical means.
	df = derive_historical_features(df, feature_columns, k=args.k)
	print(f"\nDerived leakage-safe features using past k={args.k} home/away matches per team.")

	train_df, test_df = split_train_test(df)
	print(f"Training rows: {len(train_df)} | Test rows: {len(test_df)}")

	numeric_frame = df.copy()
	for column in feature_columns:
		numeric_frame[column] = pd.to_numeric(numeric_frame[column], errors="coerce")

	print(f"Using {len(feature_columns)} numeric match-stat features.")
	print("Excluding identity/demographic columns such as nationality, position, age, minutes, and shirtnumber.")

	train_numeric = train_df.copy()
	test_numeric = test_df.copy()
	for column in feature_columns:
		train_numeric[column] = pd.to_numeric(train_numeric[column], errors="coerce")
		test_numeric[column] = pd.to_numeric(test_numeric[column], errors="coerce")

	train_feature_matrix, missing_report = team_mean_impute(train_numeric, feature_columns)
	test_feature_matrix, _ = team_mean_impute_with_reference(test_numeric, feature_columns, train_numeric)

	FEATURE_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
	train_feature_matrix.to_csv(FEATURE_MATRIX_PATH, index=False)
	test_feature_matrix.to_csv(TEST_FEATURE_MATRIX_PATH, index=False)
	missing_report.to_csv(MISC_DIR / "match_statistics_missingness_report.csv", index=False)

	if PERFORM_PCA:
		train_features_out, test_features_out, pca_weights = pca_reduce(
			train_feature_matrix, test_feature_matrix, PCA_VARIANCE_THRESHOLD
		)
		PCA_WEIGHTS_PATH.write_text(json.dumps(pca_weights, indent=2), encoding="utf-8")
		print(f"PCA reduced {len(feature_columns)} features to {train_features_out.shape[1]} components "
		      f"(>= {PCA_VARIANCE_THRESHOLD:.0%} variance). Weights: {PCA_WEIGHTS_PATH}")
	else:
		train_features_out, test_features_out = train_feature_matrix, test_feature_matrix

	train_output_df = pd.concat(
		[
			train_df[["match_id"]].reset_index(drop=True),
			train_df[["game_id"]].reset_index(drop=True),
			train_df[["Home_goals"]].rename(columns={"Home_goals": "HG"}).reset_index(drop=True),
			train_df[["Away_goals"]].rename(columns={"Away_goals": "AG"}).reset_index(drop=True),
			train_features_out.reset_index(drop=True),
		],
		axis=1,
	)
	test_output_df = pd.concat(
		[
			test_df[["match_id"]].reset_index(drop=True),
			test_df[["game_id"]].reset_index(drop=True),
			test_df[["Home_goals"]].rename(columns={"Home_goals": "HG"}).reset_index(drop=True),
			test_df[["Away_goals"]].rename(columns={"Away_goals": "AG"}).reset_index(drop=True),
			test_features_out.reset_index(drop=True),
		],
		axis=1,
	)

	train_output_df.to_csv(TRAIN_OUTPUT_PATH, index=False)
	test_output_df.to_csv(TEST_OUTPUT_PATH, index=False)
	update_pca_status("match_stats", PERFORM_PCA)

	print(f"\nSaved training dataset to {TRAIN_OUTPUT_PATH}")
	print(f"Saved test dataset to {TEST_OUTPUT_PATH}")
	print(f"Saved imputed training feature matrix to {FEATURE_MATRIX_PATH}")
	print(f"Saved imputed test feature matrix to {TEST_FEATURE_MATRIX_PATH}")
	if PERFORM_PCA:
		print(f"Saved {train_features_out.shape[1]} PCA components (fit on training data).")
	else:
		print(f"Saved {len(feature_columns)} raw match-stat features (no PCA applied).")


if __name__ == "__main__":
	main()