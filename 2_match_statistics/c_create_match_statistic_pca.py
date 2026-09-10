#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build a leakage-safe match-statistics dataset from data/big5_matches.csv.

For each numeric match-stat feature, values are derived as the mean over the
past k matches of the same team on the same side (home or away), excluding the
current match. This ensures inputs are available before kickoff.

After this historical transformation, the script imputes remaining gaps and
saves the raw (non-PCA) features directly as the TRAIN/TEST datasets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# Number of past side-specific matches used to compute pre-match means.
K_PAST_MATCHES = 5

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
INPUT_PATH = PROJECT_DIR / "data" / "big5_matches_fixed.csv"
FINAL_DATA_DIR = PROJECT_DIR / "final_data"
MISC_DIR = PROJECT_DIR / "misc"

TRAIN_OUTPUT_PATH = FINAL_DATA_DIR / "match_statistics_pca_TRAIN.csv"
TEST_OUTPUT_PATH = FINAL_DATA_DIR / "match_statistics_pca_TEST.csv"
FEATURE_MATRIX_PATH = PROJECT_DIR / "data" / "match_statistics_before_pca.csv"
TEST_FEATURE_MATRIX_PATH = PROJECT_DIR / "data" / "match_statistics_before_pca_TEST.csv"


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
	"""Impute missing values with the mean of the respective team.

	Home-side columns are grouped by home_team, away-side columns by away_team.
	Returns the imputed feature frame and a small report frame with missing counts.
	"""

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
	"""Impute missing values using team means learned from a reference frame.

	The reference frame should be the training data. If a team has no training
	values for a feature, the training global mean is used as fallback.
	"""

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

	# Replace raw match statistics with leakage-safe historical means.
	df = derive_historical_features(df, feature_columns, k=args.k)
	print(f"Derived leakage-safe features using past k={args.k} home/away matches per team.")

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

	train_output_df = pd.concat(
		[
			train_df[["match_id"]].reset_index(drop=True),
			train_df[["game_id"]].reset_index(drop=True),
			train_df[["Home_goals"]].rename(columns={"Home_goals": "HG"}).reset_index(drop=True),
			train_df[["Away_goals"]].rename(columns={"Away_goals": "AG"}).reset_index(drop=True),
			train_feature_matrix.reset_index(drop=True),
		],
		axis=1,
	)
	test_output_df = pd.concat(
		[
			test_df[["match_id"]].reset_index(drop=True),
			test_df[["game_id"]].reset_index(drop=True),
			test_df[["Home_goals"]].rename(columns={"Home_goals": "HG"}).reset_index(drop=True),
			test_df[["Away_goals"]].rename(columns={"Away_goals": "AG"}).reset_index(drop=True),
			test_feature_matrix.reset_index(drop=True),
		],
		axis=1,
	)

	train_output_df.to_csv(TRAIN_OUTPUT_PATH, index=False)
	test_output_df.to_csv(TEST_OUTPUT_PATH, index=False)

	print(f"Saved training dataset to {TRAIN_OUTPUT_PATH}")
	print(f"Saved test dataset to {TEST_OUTPUT_PATH}")
	print(f"Saved imputed training feature matrix to {FEATURE_MATRIX_PATH}")
	print(f"Saved imputed test feature matrix to {TEST_FEATURE_MATRIX_PATH}")
	print(f"Saved {len(feature_columns)} raw match-stat features (no PCA applied).")


if __name__ == "__main__":
	main()
