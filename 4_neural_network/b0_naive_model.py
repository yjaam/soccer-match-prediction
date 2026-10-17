#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""NAIVE BASELINE: Predict HOME_WIN / DRAW / AWAY_WIN from market values.

Rule
----
    value_gap = home_value - away_value
    predicted = HOME_WIN  if value_gap >  +tau
              = AWAY_WIN  if value_gap <  -tau
              = DRAW      otherwise

    tau is a single scalar tuned on a held-out slice of the training
    seasons to maximize balanced accuracy.

No training, no optimizer, no model — the "model" is a scalar threshold
plus the raw market-value comparison.

Split (matches the rest of the historical pipeline)
---------------------------------------------------
    train: seasons before 2025-2026
    test:  2025-2026 season

    tau is tuned on a random val_fraction slice of the training seasons,
    then frozen and applied to the test season.

Reads
-----
- ../data/player_group_before_pca.csv         (raw market values + labels)
- ../data/lineups_values_ratings_games.csv    (fallback)
- ../data/big5_matches_fixed.csv              (labels if missing)

Writes
------
Each saved prediction CSV contains exactly four columns:
    game_id, HG, AG, predicted_result

`predicted_result` is coded as:
    +1  -> HOME_WIN
     0  -> DRAW
    -1  -> AWAY_WIN

This matches the coding used by the fusion model's predictions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)


# ============================================================================
# PATHS
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULT_DATA_DIR = PROJECT_DIR / "result_data"
DATA_DIR = PROJECT_DIR / "data"

NAIVE_SUMMARY_PATH = RESULT_DATA_DIR / "baseline_naive_summary.json"
NAIVE_TEST_PRED_PATH = RESULT_DATA_DIR / "baseline_naive_test_predictions.csv"
NAIVE_TRAIN_PRED_PATH = RESULT_DATA_DIR / "baseline_naive_train_predictions.csv"

RAW_HISTORICAL_CANDIDATES = [
    DATA_DIR / "player_group_before_pca.csv",
    DATA_DIR / "lineups_values_ratings_games.csv",
]

GOALS_SOURCE = DATA_DIR / "big5_matches_fixed.csv"


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class NaiveConfig:
    test_season: int = 2025          # season_start_year == 2025 → test
    val_fraction: float = 0.15       # hold-out slice of train for tuning tau
    threshold_max: float = 4.0       # upper bound for the tau sweep
    threshold_steps: int = 401       # resolution of the tau sweep
    seed: int = 42
    use_pca: bool = True             # score a team by PC1 of its position-group values, not their sum


VALUE_COLS_HOME = [
    "Home_Goalkeeper_Avg_Value",
    "Home_Defender_Avg_Value",
    "Home_Midfielder_Avg_Value",
    "Home_Attacker_Avg_Value",
]
VALUE_COLS_AWAY = [
    "Away_Goalkeeper_Avg_Value",
    "Away_Defender_Avg_Value",
    "Away_Midfielder_Avg_Value",
    "Away_Attacker_Avg_Value",
]
VALUE_COLS = VALUE_COLS_HOME + VALUE_COLS_AWAY

OUTCOME_NAMES = {0: "HOME", 1: "DRAW", 2: "AWAY"}
OUTCOME_LABEL = {0: "HOME_WIN", 1: "DRAW", 2: "AWAY_WIN"}
LABEL_TO_CODE = {"HOME_WIN": 0, "DRAW": 1, "AWAY_WIN": 2}

# Internal 0/1/2 coding → signed coding used in the saved CSVs
INT_TO_SIGNED = np.array([1, 0, -1])

# Columns written to the output CSVs
SAVED_COLUMNS = ["game_id", "HG", "AG", "predicted_result"]


# ============================================================================
# HELPERS
# ============================================================================

def set_seed(seed: int) -> None:
    np.random.seed(seed)


def outcome_labels(hg: np.ndarray, ag: np.ndarray) -> np.ndarray:
    return np.where(hg > ag, 0, np.where(hg == ag, 1, 2))


def season_start_year(date_value) -> float:
    d = pd.to_datetime(date_value, errors="coerce")
    if pd.isna(d):
        return np.nan
    return float(d.year if d.month >= 7 else d.year - 1)


def _find_goal_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    home_candidates = ["HG", "FTHG", "Home_goals", "home_goals", "home_score"]
    away_candidates = ["AG", "FTAG", "Away_goals", "away_goals", "away_score"]
    hg = next((c for c in home_candidates if c in df.columns), None)
    ag = next((c for c in away_candidates if c in df.columns), None)
    return hg, ag


def ensure_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add HG/AG to df if missing, by merging from big5_matches_fixed.csv."""
    if "HG" in df.columns and "AG" in df.columns:
        return df
    if "game_id" not in df.columns or not GOALS_SOURCE.exists():
        return df

    goals = pd.read_csv(GOALS_SOURCE, low_memory=False)
    hg_col, ag_col = _find_goal_columns(goals)
    if hg_col is None or ag_col is None:
        return df

    goals = (
        goals[["game_id", hg_col, ag_col]]
        .rename(columns={hg_col: "HG", ag_col: "AG"})
    )
    df = df.drop(columns=[c for c in ("HG", "AG") if c in df.columns])
    return df.merge(goals, on="game_id", how="left")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_historical() -> tuple[pd.DataFrame, str]:
    for path in RAW_HISTORICAL_CANDIDATES:
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        missing = [c for c in VALUE_COLS if c not in df.columns]
        if missing:
            print(f"  Skipping {path.name}: missing {len(missing)} value cols.")
            continue
        print(f"  Loaded {path.name}  ({len(df)} rows)")
        return df, path.name

    raise FileNotFoundError(
        "No historical raw-value source found. Tried:\n"
        + "\n".join(f"    {p}" for p in RAW_HISTORICAL_CANDIDATES)
    )


# ============================================================================
# PREDICTION RULE
# ============================================================================

def fit_value_pca(train_df: pd.DataFrame) -> dict:
    """PC1 of the four position-group values, shared by home and away teams.

    Fit on the stacked home/away rows of the training data (no labels), standardised.
    The sign is set so a higher score means a more valuable team.
    """
    home = train_df[VALUE_COLS_HOME].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    away = train_df[VALUE_COLS_AWAY].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    stacked = np.vstack([home, away])

    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std[std == 0] = 1.0
    _, singular_values, vt = np.linalg.svd((stacked - mean) / std, full_matrices=False)
    component = vt[0] if vt[0].sum() >= 0 else -vt[0]
    explained = float(singular_values[0] ** 2 / (singular_values ** 2).sum())
    return {"mean": mean, "std": std, "component": component, "explained": explained}


def compute_team_scores(df: pd.DataFrame, pca: dict | None = None) -> tuple[pd.Series, pd.Series]:
    if pca is None:
        home = df[VALUE_COLS_HOME].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        away = df[VALUE_COLS_AWAY].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        return home, away

    def score(cols):
        x = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        return pd.Series(((x - pca["mean"]) / pca["std"]) @ pca["component"], index=df.index)

    return score(VALUE_COLS_HOME), score(VALUE_COLS_AWAY)


def predict_with_threshold(df: pd.DataFrame, tau: float, pca: dict | None = None) -> pd.DataFrame:
    """
    Apply the decision rule with the given threshold tau.

    Returns a copy of df with diagnostic columns plus two encodings of the
    prediction:
        predicted_code    : 0 = HOME, 1 = DRAW, 2 = AWAY   (internal)
        predicted_result  : +1 = HOME, 0 = DRAW, -1 = AWAY (for CSV output)
    """
    home_score, away_score = compute_team_scores(df, pca)
    gap = (home_score - away_score).astype(float)

    labels = np.where(
        gap >  tau, "HOME_WIN",
        np.where(gap < -tau, "AWAY_WIN", "DRAW"),
    )
    codes = np.array([LABEL_TO_CODE[l] for l in labels], dtype=int)

    out = df.copy()
    out["home_value_score"] = home_score.astype(float)
    out["away_value_score"] = away_score.astype(float)
    out["value_gap"] = gap
    out["predicted_label"] = labels
    out["predicted_code"] = codes
    out["predicted_result"] = INT_TO_SIGNED[codes]
    return out


def tune_threshold(gap: np.ndarray, hg: np.ndarray, ag: np.ndarray,
                    tau_max: float, n_steps: int) -> tuple[float, float]:
    """
    Sweep tau over a grid and pick the value that maximizes balanced
    accuracy on (gap, hg, ag). Returns (best_tau, best_balanced_accuracy).
    """
    true = outcome_labels(hg, ag)
    grid = np.linspace(0.0, tau_max, n_steps)

    best_tau, best_bal = 0.0, -1.0
    for tau in grid:
        pred = np.where(gap > tau, 0, np.where(gap < -tau, 2, 1))
        bal = balanced_accuracy_score(true, pred)
        if bal > best_bal:
            best_bal, best_tau = bal, float(tau)
    return best_tau, best_bal


# ============================================================================
# METRICS
# ============================================================================

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                    zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred,
                                              labels=[0, 1, 2]).tolist(),
        "n_rows": int(len(y_true)),
    }


def _print_metrics(title: str, m: dict) -> None:
    print(f"\n  {title}")
    print(f"    Accuracy:           {m['accuracy']:.4f}")
    print(f"    Balanced accuracy:  {m['balanced_accuracy']:.4f}")
    print(f"    Macro F1:           {m['macro_f1']:.4f}")
    print(f"    Rows:               {m['n_rows']}")
    cm = np.array(m["confusion_matrix"])
    print(f"    Confusion matrix (rows=true, cols=pred):")
    print(f"      {'':>10}  {'Pred H':>7}  {'Pred D':>7}  {'Pred A':>7}")
    for i, name in enumerate(["Actual H", "Actual D", "Actual A"]):
        print(f"      {name:>10}  {cm[i, 0]:>7}  {cm[i, 1]:>7}  {cm[i, 2]:>7}")


def _distribution(y: np.ndarray) -> dict:
    return {name: int((y == code).sum()) for code, name in OUTCOME_NAMES.items()}


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    config = NaiveConfig()
    set_seed(config.seed)

    RESULT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("NAIVE BASELINE: MARKET-VALUE GAP vs. TUNED DRAW THRESHOLD")
    print("=" * 74)
    print(f"Rule: HOME_WIN if gap >  tau")
    print(f"      AWAY_WIN if gap < -tau")
    print(f"      DRAW     otherwise")
    print(f"tau:  tuned on a {int(config.val_fraction*100)}% slice of the "
          f"training seasons")
    print(f"Train: seasons < {config.test_season}   "
          f"Test: {config.test_season}-{config.test_season + 1}")
    print(f"Output columns: {SAVED_COLUMNS}")
    print(f"predicted_result coding: +1 = HOME_WIN, 0 = DRAW, -1 = AWAY_WIN")
    print("Value columns summed (both sides):")
    for c in VALUE_COLS_HOME:
        print(f"    + {c}")

    # ------------------------------------------------------------------
    # Load + clean
    # ------------------------------------------------------------------
    print(f"\n{'-' * 74}")
    print("LOADING DATA")
    print(f"{'-' * 74}")

    hist, source_name = load_historical()
    hist = ensure_labels(hist)

    hist["HG"] = pd.to_numeric(hist["HG"], errors="coerce")
    hist["AG"] = pd.to_numeric(hist["AG"], errors="coerce")

    if "season_start_year" in hist.columns:
        hist["season_start_year"] = pd.to_numeric(
            hist["season_start_year"], errors="coerce"
        )
    else:
        date_col = next((c for c in ("Date", "date", "Date_x")
                         if c in hist.columns), None)
        if date_col is None:
            raise KeyError("No season_start_year and no Date column.")
        hist["season_start_year"] = hist[date_col].apply(season_start_year)

    before = len(hist)
    hist = hist.dropna(
        subset=["HG", "AG", "season_start_year"] + VALUE_COLS
    ).copy()
    if len(hist) < before:
        print(f"  Dropped {before - len(hist)} rows with missing "
              f"labels, season, or values.")
    print(f"  Usable rows: {len(hist)}")

    if hist.empty:
        print("  ⚠ No usable rows. Aborting.")
        return

    # ------------------------------------------------------------------
    # Season-based split (matches the rest of the pipeline)
    # ------------------------------------------------------------------
    train_full = hist.loc[hist["season_start_year"] < config.test_season].copy()
    test_df = hist.loc[hist["season_start_year"] == config.test_season].copy()

    print(f"\n  Train rows (season < {config.test_season}):    {len(train_full)}")
    print(f"  Test rows  (season == {config.test_season}-{config.test_season + 1}):  {len(test_df)}")

    if train_full.empty or test_df.empty:
        print("  ⚠ Empty train or test split — check season_start_year.")
        return

    # ------------------------------------------------------------------
    # Split train into train / threshold-tuning slices
    # ------------------------------------------------------------------
    rng = np.random.default_rng(config.seed)
    idx = rng.permutation(len(train_full))
    n_val = int(len(train_full) * config.val_fraction)
    tune_idx, fit_idx = idx[:n_val], idx[n_val:]

    tune_df = train_full.iloc[tune_idx].reset_index(drop=True)

    print(f"\n  Threshold-tuning slice: {len(tune_df)} rows "
          f"({100.0 * len(tune_df) / len(train_full):.1f}% of train)")

    # Compute the value gap on the tuning slice
    pca = fit_value_pca(train_full) if config.use_pca else None
    if pca is not None:
        print(f"  Team value score: PC1 of the position-group values "
              f"({100 * pca['explained']:.1f}% variance)")
    home_tune, away_tune = compute_team_scores(tune_df, pca)
    gap_tune = (home_tune - away_tune).values.astype(float)

    # ------------------------------------------------------------------
    # Tune tau
    # ------------------------------------------------------------------
    print(f"\n{'-' * 74}")
    print("TUNING DRAW THRESHOLD")
    print(f"{'-' * 74}")

    tau, tune_bal = tune_threshold(
        gap_tune,
        tune_df["HG"].values,
        tune_df["AG"].values,
        config.threshold_max,
        config.threshold_steps,
    )
    print(f"  Best tau:           {tau:.4f}")
    print(f"  Tuning balanced:    {tune_bal:.4f}")

    # A second reference point: tau = 0 (equivalent to the previous
    # never-predict-draw version, up to the tie-break)
    pred_at_zero = np.where(gap_tune > 0, 0, np.where(gap_tune < 0, 2, 0))
    bal_at_zero = balanced_accuracy_score(
        outcome_labels(tune_df["HG"].values, tune_df["AG"].values),
        pred_at_zero,
    )
    print(f"  tau=0 baseline bal: {bal_at_zero:.4f}  "
          f"(gain from tuning: {tune_bal - bal_at_zero:+.4f})")

    # ------------------------------------------------------------------
    # Apply to full train and test
    # ------------------------------------------------------------------
    train_pred = predict_with_threshold(train_full, tau, pca)
    test_pred = predict_with_threshold(test_df, tau, pca)

    print(f"\n{'-' * 74}")
    print("EVALUATION")
    print(f"{'-' * 74}")

    y_train_true = outcome_labels(train_pred["HG"].values, train_pred["AG"].values)
    y_train_pred = train_pred["predicted_code"].values
    y_test_true = outcome_labels(test_pred["HG"].values, test_pred["AG"].values)
    y_test_pred = test_pred["predicted_code"].values

    train_metrics = evaluate(y_train_true, y_train_pred)
    test_metrics = evaluate(y_test_true, y_test_pred)

    _print_metrics(f"Train metrics (tau = {tau:.4f})", train_metrics)
    _print_metrics(f"Test metrics  (tau = {tau:.4f})", test_metrics)

    train_pred_dist = _distribution(y_train_pred)
    train_actual_dist = _distribution(y_train_true)
    test_pred_dist = _distribution(y_test_pred)
    test_actual_dist = _distribution(y_test_true)

    def _show_dist(label, pred_dist, actual_dist, n):
        print(f"\n  {label}")
        print(f"    {'':>6}  {'Predicted':>12}  {'Actual':>12}")
        for code, name in OUTCOME_NAMES.items():
            p = pred_dist[name]
            a = actual_dist[name]
            pp = 100.0 * p / n if n else 0.0
            ap = 100.0 * a / n if n else 0.0
            print(f"    {name:>6}  {p:>6} ({pp:>5.1f}%)  "
                  f"{a:>6} ({ap:>5.1f}%)")

    _show_dist("Distribution — train", train_pred_dist,
               train_actual_dist, len(y_train_true))
    _show_dist("Distribution — test", test_pred_dist,
               test_actual_dist, len(y_test_true))

    # Sanity: biggest value gaps in test
    test_pred["abs_gap"] = test_pred["value_gap"].abs()
    top = test_pred.sort_values("abs_gap", ascending=False).head(5)
    print(f"\n  Top-5 largest value gaps in test:")
    for _, r in top.iterrows():
        actual = OUTCOME_LABEL[int(outcome_labels(
            np.array([r["HG"]]), np.array([r["AG"]]))[0])]
        mark = "✓" if actual == r["predicted_label"] else "✗"
        print(f"    {mark} home={r['home_value_score']:>6.1f}  "
              f"away={r['away_value_score']:>6.1f}  "
              f"pred={r['predicted_label']:<9} actual={actual}")

    # How many draws does the threshold actually capture on test?
    n_test_draws_pred = int((test_pred["predicted_code"] == 1).sum())
    n_test_draws_actual = int((y_test_true == 1).sum())
    print(f"\n  Draws on test: predicted {n_test_draws_pred}, "
          f"actual {n_test_draws_actual}")

    # ------------------------------------------------------------------
    # Save outputs — exactly four columns
    # ------------------------------------------------------------------
    train_out = train_pred[SAVED_COLUMNS].copy()
    test_out = test_pred[SAVED_COLUMNS].copy()

    train_out.to_csv(NAIVE_TRAIN_PRED_PATH, index=False)
    test_out.to_csv(NAIVE_TEST_PRED_PATH, index=False)

    # Sanity check: report the coding distribution of saved columns
    print(f"\n  Saved column layout: {SAVED_COLUMNS}")
    for label, df_ in (("train", train_out), ("test", test_out)):
        vc = df_["predicted_result"].value_counts().sort_index()
        parts = ", ".join(f"{int(k)}={int(v)}" for k, v in vc.items())
        print(f"    {label}: {parts}")

    summary = {
        "run_date": pd.Timestamp.now().isoformat(timespec="seconds"),
        "rule": "threshold_on_market_value_gap",
        "value_score": "pca_pc1" if pca is not None else "sum",
        "threshold": tau,
        "saved_columns": SAVED_COLUMNS,
        "result_coding": {"HOME_WIN": 1, "DRAW": 0, "AWAY_WIN": -1},
        "tuning": {
            "val_fraction": config.val_fraction,
            "threshold_max": config.threshold_max,
            "threshold_steps": config.threshold_steps,
            "tuning_balanced_accuracy": float(tune_bal),
            "balanced_accuracy_at_tau_zero": float(bal_at_zero),
        },
        "config": config.__dict__,
        "value_columns": {"home": VALUE_COLS_HOME, "away": VALUE_COLS_AWAY},
        "source": source_name,
        "split": {"test_season": int(config.test_season)},
        "train": {
            "n_rows": int(len(train_full)),
            "metrics": train_metrics,
            "prediction_distribution": train_pred_dist,
            "actual_distribution": train_actual_dist,
        },
        "test": {
            "n_rows": int(len(test_df)),
            "metrics": test_metrics,
            "prediction_distribution": test_pred_dist,
            "actual_distribution": test_actual_dist,
        },
    }
    NAIVE_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"\n  Saved train predictions to {NAIVE_TRAIN_PRED_PATH}")
    print(f"  Saved test  predictions to {NAIVE_TEST_PRED_PATH}")
    print(f"  Saved summary to          {NAIVE_SUMMARY_PATH}")

    print("\n" + "=" * 74)
    print("Done.")
    print("=" * 74)


if __name__ == "__main__":
    main()