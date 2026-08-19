#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build a leakage-safe player-group PCA dataset from data_before_pca.csv.

LOCF is used as the primary imputation strategy because the inputs are ordered
time series of market values and player ratings. Any residual gaps that LOCF
cannot fill are completed with training-set means so the PCA inputs stay fully
numeric without using future information.

PCA is fit only on seasons up to 2024-2025. The 2025-2026 season is held out as
test data and transformed with the training PCA weights.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
INPUT_PATH = PROJECT_DIR / "data" / "player_group_before_pca.csv"
TARGET_VARS_PATH = PROJECT_DIR / "data" / "target_variables.csv"
FINAL_DATA_DIR = PROJECT_DIR / "final_data"
TRAIN_OUTPUT_PATH = FINAL_DATA_DIR / "player_statistics_pca_TRAIN.csv"
TEST_OUTPUT_PATH = FINAL_DATA_DIR / "player_statistics_pca_TEST.csv"
MISC_DIR = PROJECT_DIR / "misc"


GROUP_CONFIGS = {
    "home_Goalkeeper": [
        "Home_Goalkeeper_Avg_Value",
        "HomeGoalkeeper_overall_Avg",
        "HomeGoalkeeper_goalkeeping_diving_Avg",
        "HomeGoalkeeper_goalkeeping_handling_Avg",
        "HomeGoalkeeper_goalkeeping_kicking_Avg",
        "HomeGoalkeeper_goalkeeping_positioning_Avg",
        "HomeGoalkeeper_goalkeeping_reflexes_Avg",
        "HomeGoalkeeper_goalkeeping_speed_Avg",
    ],
    "away_Goalkeeper": [
        "Away_Goalkeeper_Avg_Value",
        "AwayGoalkeeper_overall_Avg",
        "AwayGoalkeeper_goalkeeping_diving_Avg",
        "AwayGoalkeeper_goalkeeping_handling_Avg",
        "AwayGoalkeeper_goalkeeping_kicking_Avg",
        "AwayGoalkeeper_goalkeeping_positioning_Avg",
        "AwayGoalkeeper_goalkeeping_reflexes_Avg",
        "AwayGoalkeeper_goalkeeping_speed_Avg",
    ],
    "home_Defenders": [
        "Home_Defender_Avg_Value",
        "HomeDefender_overall_Avg",
        "HomeDefender_pace_Avg",
        "HomeDefender_shooting_Avg",
        "HomeDefender_passing_Avg",
        "HomeDefender_dribbling_Avg",
        "HomeDefender_defending_Avg",
        "HomeDefender_physic_Avg",
    ],
    "away_Defenders": [
        "Away_Defender_Avg_Value",
        "AwayDefender_overall_Avg",
        "AwayDefender_pace_Avg",
        "AwayDefender_shooting_Avg",
        "AwayDefender_passing_Avg",
        "AwayDefender_dribbling_Avg",
        "AwayDefender_defending_Avg",
        "AwayDefender_physic_Avg",
    ],
    "home_Midfielders": [
        "Home_Midfielder_Avg_Value",
        "HomeMidfielder_overall_Avg",
        "HomeMidfielder_pace_Avg",
        "HomeMidfielder_shooting_Avg",
        "HomeMidfielder_passing_Avg",
        "HomeMidfielder_dribbling_Avg",
        "HomeMidfielder_defending_Avg",
        "HomeMidfielder_physic_Avg",
    ],
    "away_Midfielders": [
        "Away_Midfielder_Avg_Value",
        "AwayMidfielder_overall_Avg",
        "AwayMidfielder_pace_Avg",
        "AwayMidfielder_shooting_Avg",
        "AwayMidfielder_passing_Avg",
        "AwayMidfielder_dribbling_Avg",
        "AwayMidfielder_defending_Avg",
        "AwayMidfielder_physic_Avg",
    ],
    "home_Attackers": [
        "Home_Attacker_Avg_Value",
        "HomeAttacker_overall_Avg",
        "HomeAttacker_pace_Avg",
        "HomeAttacker_shooting_Avg",
        "HomeAttacker_passing_Avg",
        "HomeAttacker_dribbling_Avg",
        "HomeAttacker_defending_Avg",
        "HomeAttacker_physic_Avg",
    ],
    "away_Attackers": [
        "Away_Attacker_Avg_Value",
        "AwayAttacker_overall_Avg",
        "AwayAttacker_pace_Avg",
        "AwayAttacker_shooting_Avg",
        "AwayAttacker_passing_Avg",
        "AwayAttacker_dribbling_Avg",
        "AwayAttacker_defending_Avg",
        "AwayAttacker_physic_Avg",
    ],
}


def resolve_date_column(frame: pd.DataFrame) -> str:
    for candidate in ("Date_x", "Date", "date"):
        if candidate in frame.columns:
            return candidate
    raise KeyError("No date column found. Expected one of Date_x, Date, or date.")


def resolve_team_columns(frame: pd.DataFrame) -> dict[str, str]:
    home_candidates = ["home_team_code", "Home Team", "home_club_id", "HomeTeam"]
    away_candidates = ["away_team_code", "Away Team", "away_club_id", "AwayTeam"]

    home_column = next((col for col in home_candidates if col in frame.columns), None)
    away_column = next((col for col in away_candidates if col in frame.columns), None)

    if home_column is None or away_column is None:
        raise KeyError("Could not resolve home/away team columns for LOCF imputation.")

    return {"home": home_column, "away": away_column}


def split_train_test(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "season_start_year" in frame.columns:
        season_year = pd.to_numeric(frame["season_start_year"], errors="coerce")
    else:
        date_col = resolve_date_column(frame)
        dt = pd.to_datetime(frame[date_col], errors="coerce")
        season_year = np.where(dt.dt.month >= 7, dt.dt.year, dt.dt.year - 1)
        season_year = pd.Series(season_year, index=frame.index, dtype="float64")

    train_df = frame[season_year < 2025].copy()
    test_df = frame[season_year == 2025].copy()

    if train_df.empty:
        raise ValueError("Training split is empty. Check the season_start_year values.")
    if test_df.empty:
        raise ValueError("Test split is empty. No 2025-2026 rows were found.")

    return train_df, test_df


def resolve_goal_series(frame: pd.DataFrame, targets_path: Path | None = None) -> tuple[pd.Series, pd.Series]:
    if "FTHG" in frame.columns:
        hg = pd.to_numeric(frame["FTHG"], errors="coerce")
    elif "HG" in frame.columns:
        hg = pd.to_numeric(frame["HG"], errors="coerce")
    else:
        hg = pd.Series(np.nan, index=frame.index, dtype=float)

    if "FTAG" in frame.columns:
        ag = pd.to_numeric(frame["FTAG"], errors="coerce")
    elif "AG" in frame.columns:
        ag = pd.to_numeric(frame["AG"], errors="coerce")
    else:
        ag = pd.Series(np.nan, index=frame.index, dtype=float)

    if targets_path is not None and targets_path.exists() and "game_id" in frame.columns:
        targets = pd.read_csv(targets_path)
        if {"game_id", "HG", "AG"}.issubset(targets.columns):
            merged = frame[["game_id"]].merge(targets[["game_id", "HG", "AG"]], on="game_id", how="left")
            hg = hg.combine_first(pd.to_numeric(merged["HG"], errors="coerce"))
            ag = ag.combine_first(pd.to_numeric(merged["AG"], errors="coerce"))

    return hg, ag


def build_feature_groups(frame: pd.DataFrame) -> dict[str, list[str]]:
    filtered = {}
    print("\n" + "=" * 60)
    print("CHECKING AVAILABLE FEATURES PER GROUP")
    print("=" * 60)

    for group_key, group_cols in GROUP_CONFIGS.items():
        available = [col for col in group_cols if col in frame.columns]
        if available:
            filtered[group_key] = available
            if len(available) < len(group_cols):
                missing = sorted(set(group_cols) - set(available))
                print(f"{group_key}: {len(available)}/{len(group_cols)} features")
                print(f"  Missing: {missing}")
            else:
                print(f"{group_key}: {len(available)}/{len(group_cols)} features ✓")
        else:
            print(f"{group_key}: 0/{len(group_cols)} features - SKIPPED (no features available)")

    if not filtered:
        raise ValueError("No matching features found. Please check the column names in data_before_pca.csv.")

    print(f"\nTotal groups with features: {len(filtered)}")
    print(f"Total features to process: {sum(len(cols) for cols in filtered.values())}")
    return filtered


def locf_impute(
    frame: pd.DataFrame,
    feature_columns: list[str],
    team_columns: dict[str, str],
    date_column: str,
    training_reference: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply LOCF within each team-side series, then fill residual gaps with training means."""

    imputed = frame.copy()
    imputed[date_column] = pd.to_datetime(imputed[date_column], errors="coerce")
    sort_columns = [date_column]
    if "game_id" in imputed.columns:
        sort_columns.append("game_id")
    imputed = imputed.sort_values(sort_columns).reset_index(drop=True)

    report_rows = []

    for column in feature_columns:
        side = "home" if column.startswith("Home") else "away"
        team_column = team_columns[side]

        imputed[column] = pd.to_numeric(imputed[column], errors="coerce")
        missing_before = int(imputed[column].isna().sum())

        imputed[column] = imputed.groupby(team_column, sort=False)[column].ffill()
        missing_after_locf = int(imputed[column].isna().sum())

        fallback_mean = pd.to_numeric(training_reference[column], errors="coerce").mean()
        imputed[column] = imputed[column].fillna(fallback_mean)
        if pd.isna(imputed[column]).any():
            imputed[column] = imputed[column].fillna(0.0)

        missing_after = int(imputed[column].isna().sum())

        report_rows.append(
            {
                "feature": column,
                "team_column": team_column,
                "missing_before": missing_before,
                "missing_after_locf": missing_after_locf,
                "missing_after_final": missing_after,
                "fallback_mean_used": pd.notna(fallback_mean),
            }
        )

    return imputed, pd.DataFrame(report_rows)


def fit_group_pca(train_matrix: pd.DataFrame, test_matrix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a one-component PCA on the training matrix and transform train/test."""

    train_means = train_matrix.mean(axis=0)
    train_stds = train_matrix.std(axis=0, ddof=0).replace(0, np.nan)

    train_scaled = ((train_matrix - train_means) / train_stds).fillna(0.0).to_numpy(dtype=float)
    test_scaled = ((test_matrix - train_means) / train_stds).fillna(0.0).to_numpy(dtype=float)

    if train_scaled.shape[0] < 2:
        raise ValueError("At least two training rows are required to fit PCA.")

    _, singular_values, vt = np.linalg.svd(train_scaled, full_matrices=False)
    component = vt[0]

    total_variance = float(((singular_values**2) / (train_scaled.shape[0] - 1)).sum())
    pc1_variance = float((singular_values[0] ** 2) / (train_scaled.shape[0] - 1))
    pc1_explained_variance = pc1_variance / total_variance if total_variance > 0 else 0.0

    train_scores = train_scaled @ component.T
    test_scores = test_scaled @ component.T

    return train_scores, test_scores, pc1_explained_variance


def main() -> None:
    if not INPUT_PATH.exists():
        print(f"Error: input file not found at {INPUT_PATH}")
        return

    FINAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MISC_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH, low_memory=False)
    print(f"Loaded PCA data with {len(df)} rows and {df.shape[1]} columns.")

    if "game_id" not in df.columns:
        raise KeyError("game_id column is required.")

    date_column = resolve_date_column(df)
    team_columns = resolve_team_columns(df)
    hg_series, ag_series = resolve_goal_series(df, TARGET_VARS_PATH)
    df = df.copy()
    df["HG"] = hg_series.values
    df["AG"] = ag_series.values

    train_raw, test_raw = split_train_test(df)

    print(f"Training rows: {len(train_raw)} | Test rows: {len(test_raw)}")
    print(f"Using date column: {date_column}")
    print(f"Using home team column: {team_columns['home']}")
    print(f"Using away team column: {team_columns['away']}")

    feature_groups = build_feature_groups(df)
    feature_columns = sorted({col for cols in feature_groups.values() for col in cols})

    # LOCF is applied across the full chronology so later seasons can safely
    # inherit the last observed pre-test value for each team.
    combined = pd.concat([train_raw.assign(_split="train"), test_raw.assign(_split="test")], ignore_index=True)
    combined[date_column] = pd.to_datetime(combined[date_column], errors="coerce")
    combined = combined.sort_values([date_column, "game_id"]).reset_index(drop=True)

    train_reference = train_raw.copy()
    train_reference[date_column] = pd.to_datetime(train_reference[date_column], errors="coerce")

    for column in feature_columns:
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
        train_reference[column] = pd.to_numeric(train_reference[column], errors="coerce")

    imputed, missing_report = locf_impute(
        combined,
        feature_columns=feature_columns,
        team_columns=team_columns,
        date_column=date_column,
        training_reference=train_reference,
    )

    missing_report.to_csv(MISC_DIR / "player_statistics_pca_missingness_report.csv", index=False)

    train_imputed = imputed[imputed["_split"] == "train"].copy()
    test_imputed = imputed[imputed["_split"] == "test"].copy()

    sort_columns = [date_column]
    if "game_id" in train_imputed.columns:
        sort_columns.append("game_id")
    train_imputed = train_imputed.sort_values(sort_columns).reset_index(drop=True)
    test_imputed = test_imputed.sort_values(sort_columns).reset_index(drop=True)

    summary_rows = []
    loadings_rows = []
    top_loading_rows = []
    train_scores_map: dict[str, np.ndarray] = {}
    test_scores_map: dict[str, np.ndarray] = {}

    print("\n" + "=" * 60)
    print("PERFORMING PCA BY POSITION GROUP")
    print("=" * 60)

    for group_key, group_cols in feature_groups.items():
        print(f"\nProcessing {group_key}...")
        print(f"  Features ({len(group_cols)}): {group_cols}")

        train_matrix = train_imputed[group_cols].copy()
        test_matrix = test_imputed[group_cols].copy()

        train_scores, test_scores, pc1_explained_variance = fit_group_pca(train_matrix, test_matrix)

        train_scores_map[group_key] = train_scores
        test_scores_map[group_key] = test_scores

        summary_rows.append(
            {
                "Group": group_key,
                "Features_Count": len(group_cols),
                "Features": ", ".join(group_cols),
                "PC1_Explained_Variance": pc1_explained_variance,
                "PC1_Explained_Variance_Pct": pc1_explained_variance * 100,
            }
        )

        component = np.linalg.svd(
            ((train_matrix - train_matrix.mean(axis=0)) / train_matrix.std(axis=0, ddof=0).replace(0, np.nan))
            .fillna(0.0)
            .to_numpy(dtype=float),
            full_matrices=False,
        )[2][0]
        loadings = pd.Series(component, index=group_cols)

        top_features = loadings.abs().sort_values(ascending=False).head(3)
        print(f"  PC1 explained variance: {pc1_explained_variance:.3f} ({pc1_explained_variance * 100:.1f}%)")
        print("  Top 3 features for PC1:")
        for feat in top_features.index:
            loading = float(loadings[feat])
            direction = "+" if loading > 0 else "-"
            print(f"    {feat}: {loading:.3f} ({direction})")

        for feature in group_cols:
            loadings_rows.append(
                {
                    "Group": group_key,
                    "Feature": feature,
                    "Loading": float(loadings[feature]),
                    "Abs_Loading": float(abs(loadings[feature])),
                }
            )

        for feature in top_features.index:
            top_loading_rows.append(
                {
                    "Group": group_key,
                    "Feature": feature,
                    "Loading": float(loadings[feature]),
                    "Abs_Loading": float(abs(loadings[feature])),
                }
            )

    train_output = pd.DataFrame({"game_id": train_imputed["game_id"].values})
    test_output = pd.DataFrame({"game_id": test_imputed["game_id"].values})

    for group_key in feature_groups:
        train_output[group_key] = train_scores_map[group_key]
        test_output[group_key] = test_scores_map[group_key]

    train_output["HG"] = pd.to_numeric(train_imputed["HG"], errors="coerce").values
    train_output["AG"] = pd.to_numeric(train_imputed["AG"], errors="coerce").values
    test_output["HG"] = pd.to_numeric(test_imputed["HG"], errors="coerce").values
    test_output["AG"] = pd.to_numeric(test_imputed["AG"], errors="coerce").values

    before_train_drop = len(train_output)
    before_test_drop = len(test_output)
    train_output = train_output.dropna(subset=["HG", "AG"])
    test_output = test_output.dropna(subset=["HG", "AG"])

    train_output.to_csv(TRAIN_OUTPUT_PATH, index=False)
    test_output.to_csv(TEST_OUTPUT_PATH, index=False)

    summary_df = pd.DataFrame(summary_rows)
    loadings_df = pd.DataFrame(loadings_rows)
    top_loadings_df = pd.DataFrame(top_loading_rows)

    summary_df.to_csv(MISC_DIR / "player_statistics_pca_summary.csv", index=False)
    loadings_df.to_csv(MISC_DIR / "player_statistics_pca_loadings.csv", index=False)
    top_loadings_df.to_csv(MISC_DIR / "player_statistics_pca_top_loadings.csv", index=False)

    goals_report = pd.DataFrame(
        {
            "Metric": [
                "Training rows before goal filtering",
                "Training rows removed (missing goals)",
                "Training final rows",
                "Test rows before goal filtering",
                "Test rows removed (missing goals)",
                "Test final rows",
            ],
            "Value": [
                before_train_drop,
                before_train_drop - len(train_output),
                len(train_output),
                before_test_drop,
                before_test_drop - len(test_output),
                len(test_output),
            ],
        }
    )
    goals_report.to_csv(MISC_DIR / "player_statistics_pca_goals_report.csv", index=False)

    print(f"\nSaved training PCA dataset to {TRAIN_OUTPUT_PATH}")
    print(f"Saved test PCA dataset to {TEST_OUTPUT_PATH}")
    print(f"Saved missingness report to {MISC_DIR / 'player_statistics_pca_missingness_report.csv'}")
    print(f"Saved summary and loading reports to {MISC_DIR}")
    print(f"\nTraining final shape: {train_output.shape}")
    print(f"Test final shape: {test_output.shape}")

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total groups processed: {len(feature_groups)}")
    print(f"Total input features used in PCA: {sum(len(cols) for cols in feature_groups.values())}")
    print(f"Training output rows: {len(train_output)}")
    print(f"Test output rows: {len(test_output)}")


if __name__ == "__main__":
    main()