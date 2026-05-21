#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


CORE_COLUMNS = [
	"Date",
	"Home Team",
	"Away Team",
	"attendance",
	"home_club_position_before_game",
	"away_club_position_before_game",
	"home_form",
	"away_form",
]

GROUP_COLUMNS: Dict[str, List[str]] = {
	"home_Goalkeeper": [
		"Home_Goalkeeper_overall_Avg",
		"Home_Goalkeeper_goalkeeping_diving_Avg",
		"Home_Goalkeeper_goalkeeping_handling_Avg",
		"Home_Goalkeeper_goalkeeping_kicking_Avg",
		"Home_Goalkeeper_goalkeeping_positioning_Avg",
		"Home_Goalkeeper_goalkeeping_reflexes_Avg",
		"Home_Goalkeeper_goalkeeping_speed_Avg",
	],
	"away_Goalkeeper": [
		"Away_Goalkeeper_overall_Avg",
		"Away_Goalkeeper_goalkeeping_diving_Avg",
		"Away_Goalkeeper_goalkeeping_handling_Avg",
		"Away_Goalkeeper_goalkeeping_kicking_Avg",
		"Away_Goalkeeper_goalkeeping_positioning_Avg",
		"Away_Goalkeeper_goalkeeping_reflexes_Avg",
		"Away_Goalkeeper_goalkeeping_speed_Avg",
	],
	"home_Defenders": [
		"Home_Defender_overall_Avg",
		"Home_Defender_pace_Avg",
		"Home_Defender_shooting_Avg",
		"Home_Defender_passing_Avg",
		"Home_Defender_dribbling_Avg",
		"Home_Defender_defending_Avg",
		"Home_Defender_physic_Avg",
	],
	"away_Defenders": [
		"Away_Defender_overall_Avg",
		"Away_Defender_pace_Avg",
		"Away_Defender_shooting_Avg",
		"Away_Defender_passing_Avg",
		"Away_Defender_dribbling_Avg",
		"Away_Defender_defending_Avg",
		"Away_Defender_physic_Avg",
	],
	"home_Midfielders": [
		"Home_Midfielder_overall_Avg",
		"Home_Midfielder_pace_Avg",
		"Home_Midfielder_shooting_Avg",
		"Home_Midfielder_passing_Avg",
		"Home_Midfielder_dribbling_Avg",
		"Home_Midfielder_defending_Avg",
		"Home_Midfielder_physic_Avg",
	],
	"away_Midfielders": [
		"Away_Midfielder_overall_Avg",
		"Away_Midfielder_pace_Avg",
		"Away_Midfielder_shooting_Avg",
		"Away_Midfielder_passing_Avg",
		"Away_Midfielder_dribbling_Avg",
		"Away_Midfielder_defending_Avg",
		"Away_Midfielder_physic_Avg",
	],
	"home_Attackers": [
		"Home_Attacker_overall_Avg",
		"Home_Attacker_pace_Avg",
		"Home_Attacker_shooting_Avg",
		"Home_Attacker_passing_Avg",
		"Home_Attacker_dribbling_Avg",
		"Home_Attacker_defending_Avg",
		"Home_Attacker_physic_Avg",
	],
	"away_Attackers": [
		"Away_Attacker_overall_Avg",
		"Away_Attacker_pace_Avg",
		"Away_Attacker_shooting_Avg",
		"Away_Attacker_passing_Avg",
		"Away_Attacker_dribbling_Avg",
		"Away_Attacker_defending_Avg",
		"Away_Attacker_physic_Avg",
	],
}


def impute_group_horizontally(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
	"""Fill missing values within a position group using the row mean first.

	If an entire row-group is missing, fall back to the column mean for that
	position group, then to 0 only if a column is entirely empty.
	"""
	group = df[group_cols].apply(pd.to_numeric, errors="coerce").copy()

	row_means = group.mean(axis=1, skipna=True)
	group = group.T.fillna(row_means).T

	col_means = group.mean(axis=0, skipna=True)
	group = group.fillna(col_means)

	return group.fillna(0.0)


def build_group_pc(df: pd.DataFrame, group_name: str, group_cols: List[str]) -> pd.Series:
	"""Create a single PCA component for a position group."""
	imputed = impute_group_horizontally(df, group_cols)

	scaler = StandardScaler()
	scaled = scaler.fit_transform(imputed)

	pca = PCA(n_components=1, random_state=42)
	component = pca.fit_transform(scaled).ravel()

	return pd.Series(component, index=df.index, name=group_name)


def main() -> None:
	script_dir = os.path.dirname(os.path.abspath(__file__))
	forms_path = os.path.join(script_dir, "data/lineups_values_ratings_games_forms.csv")
	out_path = os.path.join(script_dir, "data/data_pca.csv")

	df = pd.read_csv(forms_path)
	df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

	required_columns = CORE_COLUMNS + [col for cols in GROUP_COLUMNS.values() for col in cols]
	missing_columns = [col for col in required_columns if col not in df.columns]
	if missing_columns:
		raise KeyError(f"Missing required columns in forms file: {missing_columns}")

	# Keep only rows with complete match-level data; player-rating gaps are handled per group.
	df = df.dropna(subset=CORE_COLUMNS).copy()

	output = df[CORE_COLUMNS].copy()

	for group_name, group_cols in GROUP_COLUMNS.items():
		output[group_name] = build_group_pc(df, group_name, group_cols)

	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	output.to_csv(out_path, index=False, encoding="utf-8-sig")

	print(f"Saved PCA-ready CSV: {out_path}")
	print(f"Rows: {len(output)}")
	print(f"Missing cells: {int(output.isna().sum().sum())}")


if __name__ == "__main__":
	main()
