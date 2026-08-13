#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd


def safe_log_ratio(numerator, denominator, eps=1e-6):
	numerator = np.clip(pd.to_numeric(numerator, errors="coerce"), eps, None)
	denominator = np.clip(pd.to_numeric(denominator, errors="coerce"), eps, None)
	return np.log(numerator / denominator)


def build_feature(df, numerator_col, denominator_col):
	return safe_log_ratio(df[numerator_col], df[denominator_col])


def parse_game_id(game_id):
	parts = str(game_id).split("_")
	if len(parts) != 3:
		return pd.NaT, None
	date_part, home_team, away_team = parts[0], parts[1], parts[2]
	match_date = pd.to_datetime(date_part, format="%Y%m%d", errors="coerce")
	if pd.isna(match_date):
		return pd.NaT, None
	fixture_key = f"{home_team}_{away_team}"
	return match_date, fixture_key


def season_start_year_from_date(date_value):
	date_value = pd.to_datetime(date_value, errors="coerce")
	if pd.isna(date_value):
		return np.nan
	return date_value.year if date_value.month >= 7 else date_value.year - 1


def split_train_test(frame):
	season_start_year = frame["Date"].apply(season_start_year_from_date)
	train_mask = season_start_year < 2025
	test_mask = season_start_year == 2025
	train_df = frame.loc[train_mask].copy()
	test_df = frame.loc[test_mask].copy()
	if train_df.empty:
		raise ValueError("Training split is empty. Check the Date values in player_group_before_pca.csv.")
	if test_df.empty:
		raise ValueError("Test split is empty. No 2025-2026 matches were found.")
	return train_df, test_df


def main():
	script_dir = os.path.dirname(os.path.abspath(__file__))
	project_dir = os.path.dirname(script_dir)

	input_path = os.path.join(project_dir, "data", "player_group_before_pca.csv")
	target_vars_path = os.path.join(project_dir, "data", "target_variables.csv")
	final_data_dir = os.path.join(project_dir, "final_data")
	train_output_path = os.path.join(final_data_dir, "skill_matchups_TRAIN.csv")
	test_output_path = os.path.join(final_data_dir, "skill_matchups_TEST.csv")

	if not os.path.exists(input_path):
		print(f"Error: input file not found at {input_path}")
		return

	os.makedirs(final_data_dir, exist_ok=True)

	df = pd.read_csv(input_path)
	print(f"Loaded player_group_before_pca with {len(df)} rows and {df.shape[1]} columns.")

	if os.path.exists(target_vars_path):
		targets = pd.read_csv(target_vars_path)
		target_cols = [col for col in ["game_id", "HG", "AG"] if col in targets.columns]
		targets = targets[target_cols].copy()
		if {"game_id", "HG", "AG"}.issubset(targets.columns):
			df = df.merge(targets, on="game_id", how="left", suffixes=("", "_target"))
			if "HG_target" in df.columns:
				df["HG"] = df["HG"].combine_first(df["HG_target"]) if "HG" in df.columns else df["HG_target"]
				df = df.drop(columns=["HG_target"])
			if "AG_target" in df.columns:
				df["AG"] = df["AG"].combine_first(df["AG_target"]) if "AG" in df.columns else df["AG_target"]
				df = df.drop(columns=["AG_target"])
			print(f"Merged HG/AG from target_variables.csv with {len(targets)} rows.")
		else:
			print("Warning: target_variables.csv does not contain game_id, HG, and AG. Using columns already present in input if available.")
	else:
		print("Warning: target_variables.csv not found. Using HG/AG from input if available.")

	required_columns = [
		"HomeAttacker_shooting_Avg",
		"HomeAttacker_pace_Avg",
		"HomeAttacker_dribbling_Avg",
		"HomeMidfielder_overall_Avg",
		"HomeMidfielder_passing_Avg",
		"AwayAttacker_shooting_Avg",
		"AwayAttacker_pace_Avg",
		"AwayAttacker_dribbling_Avg",
		"AwayMidfielder_overall_Avg",
		"AwayMidfielder_passing_Avg",
		"HomeDefender_defending_Avg",
		"HomeDefender_pace_Avg",
		"HomeDefender_physic_Avg",
		"AwayDefender_defending_Avg",
		"AwayDefender_pace_Avg",
		"AwayDefender_physic_Avg",
		"HomeGoalkeeper_goalkeeping_reflexes_Avg",
		"AwayGoalkeeper_goalkeeping_reflexes_Avg",
	]

	missing_columns = [col for col in required_columns if col not in df.columns]
	if missing_columns:
		raise KeyError(f"Missing required columns: {missing_columns}")

	feature_df = pd.DataFrame(index=df.index)

	feature_df["home_attack_vs_away_defense"] = build_feature(
		df, "HomeAttacker_shooting_Avg", "AwayDefender_defending_Avg"
	)
	feature_df["away_attack_vs_home_defense"] = build_feature(
		df, "AwayAttacker_shooting_Avg", "HomeDefender_defending_Avg"
	)

	feature_df["home_attack_vs_away_goalkeeper"] = build_feature(
		df, "HomeAttacker_shooting_Avg", "AwayGoalkeeper_goalkeeping_reflexes_Avg"
	)
	feature_df["away_attack_vs_home_goalkeeper"] = build_feature(
		df, "AwayAttacker_shooting_Avg", "HomeGoalkeeper_goalkeeping_reflexes_Avg"
	)

	feature_df["home_midfield_vs_away_midfield"] = build_feature(
		df, "HomeMidfielder_overall_Avg", "AwayMidfielder_overall_Avg"
	)
	feature_df["away_midfield_vs_home_midfield"] = build_feature(
		df, "AwayMidfielder_overall_Avg", "HomeMidfielder_overall_Avg"
	)

	feature_df["home_midfield_vs_away_defense"] = build_feature(
		df, "HomeMidfielder_passing_Avg", "AwayDefender_defending_Avg"
	)
	feature_df["away_midfield_vs_home_defense"] = build_feature(
		df, "AwayMidfielder_passing_Avg", "HomeDefender_defending_Avg"
	)

	feature_df["home_attack_pace_vs_away_defense_pace"] = build_feature(
		df, "HomeAttacker_pace_Avg", "AwayDefender_pace_Avg"
	)
	feature_df["away_attack_pace_vs_home_defense_pace"] = build_feature(
		df, "AwayAttacker_pace_Avg", "HomeDefender_pace_Avg"
	)

	feature_df["home_attack_dribbling_vs_away_defense_physicality"] = build_feature(
		df, "HomeAttacker_dribbling_Avg", "AwayDefender_physic_Avg"
	)
	feature_df["away_attack_dribbling_vs_home_defense_physicality"] = build_feature(
		df, "AwayAttacker_dribbling_Avg", "HomeDefender_physic_Avg"
	)
	feature_columns = list(feature_df.columns)

	output_columns = ["game_id", "Date"]
	if "HG" in df.columns:
		output_columns.append("HG")
	if "AG" in df.columns:
		output_columns.append("AG")

	df_out = pd.concat([df[output_columns].copy(), feature_df], axis=1)

	if "HG" not in df_out.columns or "AG" not in df_out.columns:
		raise KeyError("HG and AG must be available after merging target variables or from the input file.")

	before_drop = len(df_out)
	df_out = df_out.dropna(subset=["HG", "AG"])
	dropped_goals = before_drop - len(df_out)
	if dropped_goals > 0:
		print(f"Dropped {dropped_goals} rows with missing HG or AG.")

	# Impute missing matchup features from the previous match of the same fixture pair.
	# The fixture pair is recovered from game_id, which encodes YYYYMMDD_HOME_AWAY.
	# We sort chronologically and forward-fill within each exact home/away pairing.
	match_meta = df_out["game_id"].apply(parse_game_id)
	df_out["_match_date"] = match_meta.apply(lambda item: item[0])
	df_out["_fixture_key"] = match_meta.apply(lambda item: item[1])

	feature_missing_before = df_out[feature_columns].isna().any(axis=1).sum()
	df_out = df_out.sort_values(["_fixture_key", "_match_date", "game_id"]).reset_index(drop=True)
	df_out[feature_columns] = df_out.groupby("_fixture_key", sort=False)[feature_columns].ffill()
	feature_missing_after_impute = df_out[feature_columns].isna().any(axis=1).sum()
	rows_imputed = feature_missing_before - feature_missing_after_impute
	print(f"Rows with at least one missing matchup feature before imputation: {feature_missing_before}")
	print(f"Rows recovered from the previous same-fixture match: {rows_imputed}")
	print(f"Rows still missing at least one matchup feature after imputation: {feature_missing_after_impute}")

	before_feature_drop = len(df_out)
	df_out = df_out.dropna(subset=feature_columns)
	dropped_features = before_feature_drop - len(df_out)
	if dropped_features > 0:
		print(f"Dropped {dropped_features} rows with missing matchup features after imputation.")

	df_out = df_out.drop(columns=["_match_date", "_fixture_key"])

	train_df, test_df = split_train_test(df_out)
	print(f"Training rows: {len(train_df)} | Test rows: {len(test_df)}")

	def build_output(frame):
		output = frame.dropna(subset=["HG", "AG"]).copy()
		before_feature_drop = len(output)
		output = output.dropna(subset=feature_columns)
		dropped = before_feature_drop - len(output)
		if "Date" in output.columns:
			output = output.drop(columns=["Date"])
		return output, dropped

	train_output, train_dropped_features = build_output(train_df)
	test_output, test_dropped_features = build_output(test_df)

	if train_dropped_features > 0:
		print(f"Training rows dropped after feature filtering: {train_dropped_features}")
	if test_dropped_features > 0:
		print(f"Test rows dropped after feature filtering: {test_dropped_features}")

	train_output.to_csv(train_output_path, index=False)
	test_output.to_csv(test_output_path, index=False)

	print(f"Rows remaining after complete-case filtering: {len(df_out)}")
	print(f"Saved training matchup dataset to {train_output_path}")
	print(f"Saved test matchup dataset to {test_output_path}")
	print(f"Training final shape: {train_output.shape}")
	print(f"Test final shape: {test_output.shape}")
	print(f"Columns: {list(train_output.columns)}")


if __name__ == "__main__":
	main()
