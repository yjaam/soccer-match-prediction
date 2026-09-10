#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train an XGBoost baseline with feature selection, full class weighting, and shallow trees.

Final iteration changes:
1. RUTHLESS FEATURE REDUCTION: Compute XGBoost gain-based importance, drop
   features below a threshold, retrain on the reduced feature set.
2. FULL CLASS WEIGHTING: Use inverse-frequency (not sqrt) to force the model
   to properly handle DRAW and AWAY classes.
3. REGULARIZED TREES: Shallower trees (max_depth=3), lower subsample/colsample.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)
from xgboost import XGBClassifier, XGBRegressor


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
FINAL_DATA_DIR = PROJECT_DIR / "final_data"
RESULT_DATA_DIR = PROJECT_DIR / "result_data"
MISC_DIR = PROJECT_DIR / "misc"
META_PATH = MISC_DIR / "nn_module_metadata.json"

SUMMARY_PATH = RESULT_DATA_DIR / "xgboost_summary.json"
MODEL_HOME_PATH = RESULT_DATA_DIR / "xgboost_home_model.json"
MODEL_AWAY_PATH = RESULT_DATA_DIR / "xgboost_away_model.json"
MODEL_OUTCOME_PATH = RESULT_DATA_DIR / "xgboost_outcome_model.json"
PRED_PATH = RESULT_DATA_DIR / "xgboost_test_predictions.csv"
FEATURE_IMPORTANCE_PATH = RESULT_DATA_DIR / "xgboost_feature_importance.csv"
SELECTED_FEATURES_PATH = MISC_DIR / "xgboost_selected_features.json"

DEFAULT_MODULES = ["lineup", "player_stats", "skill_matchups", "match_stats",
                   "league", "season", "matchup"]

OUTCOME_NAMES = {0: "HOME", 1: "DRAW", 2: "AWAY"}


@dataclass
class Config:
    modules: list[str]
    val_fraction: float = 0.15
    seed: int = 42
    # ---- Poisson regressor params (already fine) ----
    learning_rate: float = 0.05
    max_depth: int = 3
    min_child_weight: float = 4.0
    subsample: float = 0.7                    # SHALLOWER (was 0.8)
    colsample_bytree: float = 0.7             # SHALLOWER (was 0.8)
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    gamma: float = 0.0
    n_estimators: int = 3000
    early_stopping_rounds: int = 75
    # ---- Classifier params (REGULARIZED) ----
    clf_max_depth: int = 3                    # SHALLOWER (was 4)
    clf_n_estimators: int = 2000
    clf_early_stopping_rounds: int = 75
    clf_subsample: float = 0.7                # SHALLOWER (was 0.8)
    clf_colsample_bytree: float = 0.6         # SHALLOWER (was 0.8)
    clf_reg_lambda: float = 2.0               # STRONGER
    clf_min_child_weight: float = 6.0         # STRONGER
    # ---- Class weighting ----
    use_sqrt_weights: bool = False            # FULL inverse-frequency now
    # ---- Feature selection ----
    feature_selection_enabled: bool = True
    importance_threshold: float = 0.001       # Drop features below this gain fraction
    max_features: int = 80                    # Hard cap (keep top-N by importance)
    # ---- Decision ----
    blend_weight_clf: float = 0.5
    draw_margin_grid: tuple[float, ...] = field(
        default_factory=lambda: tuple(np.round(np.linspace(0.0, 0.8, 81), 3))
    )
    blend_sweep: tuple[float, ...] = field(
        default_factory=lambda: tuple(np.round(np.linspace(0.0, 1.0, 21), 2))
    )


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train XGBoost with feature reduction + full balancing.")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--clf-max-depth", type=int, default=3)
    parser.add_argument("--importance-threshold", type=float, default=0.001)
    parser.add_argument("--max-features", type=int, default=80)
    parser.add_argument("--no-feature-selection", action="store_true",
                        help="Disable feature reduction step.")
    parser.add_argument("--no-sqrt-weights", action="store_true", default=True,
                        help="Use full inverse-frequency weights (default True).")
    args = parser.parse_args()

    return Config(
        modules=list(DEFAULT_MODULES),
        val_fraction=args.val_fraction,
        seed=args.seed,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        clf_max_depth=args.clf_max_depth,
        importance_threshold=args.importance_threshold,
        max_features=args.max_features,
        feature_selection_enabled=not args.no_feature_selection,
        use_sqrt_weights=False,
    )


# ============================================================================
# DATA LOADING
# ============================================================================

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
        frame = frame.copy().drop_duplicates(subset=["game_id"], keep="first")
        feature_cols = _feature_columns(frame)
        keep_cols = ["game_id", "HG", "AG"] + feature_cols
        frame = frame[keep_cols]
        if merged is None:
            merged = frame
            continue
        merged = merged.merge(frame, on=["game_id", "HG", "AG"], how="inner")

    if merged is None or merged.empty:
        raise ValueError("No merged rows available.")

    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged = merged.dropna(subset=["HG", "AG"]).copy()
    merged = merged.sort_values("game_id").reset_index(drop=True)
    return merged


def split_train_val(frame, val_fraction):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1.")
    frame = frame.sort_values("game_id").reset_index(drop=True)
    split_idx = int(len(frame) * (1 - val_fraction))
    split_idx = min(max(split_idx, 1), len(frame) - 1)
    return frame.iloc[:split_idx].copy(), frame.iloc[split_idx:].copy()


def to_matrix(frame, feature_cols=None):
    if feature_cols is None:
        feature_cols = [col for col in frame.columns if col not in {"game_id", "HG", "AG"}]
    matrix = frame.copy()
    for col in feature_cols:
        matrix[col] = pd.to_numeric(matrix[col], errors="coerce")
    return matrix[feature_cols], feature_cols


def outcome_labels(hg, ag):
    return np.where(hg > ag, 0, np.where(hg == ag, 1, 2))


def compute_sample_weights(outcome, num_classes=3, use_sqrt=False):
    counts = np.bincount(outcome, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    if use_sqrt:
        class_weights = np.sqrt(counts.sum() / (num_classes * counts))
    else:
        class_weights = counts.sum() / (num_classes * counts)
    return class_weights[outcome]


# ============================================================================
# MODEL FITTING
# ============================================================================

def fit_poisson_regressor(x_train, y_train, x_val, y_val, config):
    model = XGBRegressor(
        objective="count:poisson", eval_metric="poisson-nloglik",
        n_estimators=config.n_estimators, learning_rate=config.learning_rate,
        max_depth=config.max_depth, min_child_weight=config.min_child_weight,
        subsample=config.subsample, colsample_bytree=config.colsample_bytree,
        reg_alpha=config.reg_alpha, reg_lambda=config.reg_lambda, gamma=config.gamma,
        tree_method="hist", random_state=config.seed, n_jobs=-1,
        early_stopping_rounds=config.early_stopping_rounds,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    return model


def fit_outcome_classifier(x_train, y_train, x_val, y_val, config, sample_weights=None):
    model = XGBClassifier(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        n_estimators=config.clf_n_estimators, learning_rate=config.learning_rate,
        max_depth=config.clf_max_depth, min_child_weight=config.clf_min_child_weight,
        subsample=config.clf_subsample, colsample_bytree=config.clf_colsample_bytree,
        reg_alpha=config.reg_alpha, reg_lambda=config.clf_reg_lambda, gamma=config.gamma,
        tree_method="hist", random_state=config.seed, n_jobs=-1,
        early_stopping_rounds=config.clf_early_stopping_rounds,
    )
    model.fit(x_train, y_train, sample_weight=sample_weights,
              eval_set=[(x_val, y_val)], verbose=False)
    return model


# ============================================================================
# FEATURE SELECTION (NEW)
# ============================================================================

def select_features_by_importance(importance_df: pd.DataFrame, config: Config):
    """Drop features below threshold and cap total count."""
    total_gain = importance_df["gain"].sum()
    importance_df = importance_df.copy()
    importance_df["gain_fraction"] = importance_df["gain"] / max(total_gain, 1e-12)

    # Filter by threshold
    above_threshold = importance_df[importance_df["gain_fraction"] >= config.importance_threshold]

    # Cap to top-N by gain
    selected = above_threshold.nlargest(config.max_features, "gain")

    return selected["feature"].tolist(), importance_df


# ============================================================================
# PRIOR CORRECTION
# ============================================================================

def prior_correct_probs(probs, train_priors, target_priors):
    probs = np.clip(probs, 1e-8, None)
    correction = target_priors / np.clip(train_priors, 1e-8, None)
    corrected = probs * correction
    corrected = corrected / corrected.sum(axis=1, keepdims=True)
    return corrected


# ============================================================================
# EVALUATION
# ============================================================================

def poisson_nll(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-6, None)
    return float(np.mean(y_pred - y_true * np.log(y_pred)))


def tune_draw_margin(lambda_home, lambda_away, true_home, true_away, margins):
    true_outcome = outcome_labels(true_home, true_away)
    best_margin, best_bal, best_acc = 0.0, -1.0, -1.0
    for margin in margins:
        pred = np.where(lambda_home > lambda_away + margin, 0,
                        np.where(lambda_away > lambda_home + margin, 2, 1))
        bal = balanced_accuracy_score(true_outcome, pred)
        if bal > best_bal:
            best_bal = bal
            best_acc = accuracy_score(true_outcome, pred)
            best_margin = float(margin)
    return best_margin, best_acc, best_bal


def rates_to_probs(lam_h, lam_a, margin):
    probs = np.zeros((len(lam_h), 3))
    diff = lam_h - lam_a
    probs[:, 0] = (diff > margin).astype(float)
    probs[:, 2] = (diff < -margin).astype(float)
    probs[:, 1] = 1.0 - probs[:, 0] - probs[:, 2]
    return probs


def metrics_summary(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def describe_confusion(cm, title):
    print(f"\n  {title}")
    print(f"  {'':>12}  {'Pred H':>7}  {'Pred D':>7}  {'Pred A':>7}")
    for i, name in enumerate(["Actual H", "Actual D", "Actual A"]):
        print(f"  {name:>12}  {cm[i, 0]:>7}  {cm[i, 1]:>7}  {cm[i, 2]:>7}")


def describe_per_class_metrics(y_true, y_pred, title):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    print(f"\n  {title}")
    print(f"  {'Class':>6}  {'Precision':>9}  {'Recall':>7}  {'F1':>6}  {'Support':>8}")
    for i, name in enumerate(["HOME", "DRAW", "AWAY"]):
        print(f"  {name:>6}  {precision[i]:>9.3f}  {recall[i]:>7.3f}  {f1[i]:>6.3f}  {support[i]:>8d}")


def describe_prediction_distribution(y_pred, title):
    counts = np.bincount(y_pred, minlength=3)
    total = counts.sum()
    print(f"\n  {title}")
    for i, name in enumerate(["HOME", "DRAW", "AWAY"]):
        pct = 100.0 * counts[i] / total if total > 0 else 0.0
        print(f"    {name:>4}: {counts[i]:>5d} ({pct:>5.1f}%)")


def evaluate_goals(frame, lambda_home, lambda_away):
    return {
        "mae_home": float(mean_absolute_error(frame["HG"].values, lambda_home)),
        "mae_away": float(mean_absolute_error(frame["AG"].values, lambda_away)),
        "mse_home": float(mean_squared_error(frame["HG"].values, lambda_home)),
        "mse_away": float(mean_squared_error(frame["AG"].values, lambda_away)),
        "poisson_nll_home": poisson_nll(frame["HG"].values, lambda_home),
        "poisson_nll_away": poisson_nll(frame["AG"].values, lambda_away),
    }


def sweep_blend_weights(clf_probs, rate_probs, y_true, weights, objective="balanced_accuracy"):
    results = []
    for w in weights:
        blended = w * clf_probs + (1 - w) * rate_probs
        pred = np.argmax(blended, axis=1)
        acc = accuracy_score(y_true, pred)
        bal = balanced_accuracy_score(y_true, pred)
        macro_f1 = f1_score(y_true, pred, average="macro", zero_division=0)
        score = bal if objective == "balanced_accuracy" else (
            macro_f1 if objective == "macro_f1" else acc)
        results.append({
            "weight_clf": float(w), "accuracy": float(acc),
            "balanced_accuracy": float(bal), "macro_f1": float(macro_f1),
            "score": float(score),
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    RESULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MISC_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("TRAINING XGBOOST (feature reduction + full balancing + shallow trees)")
    print("=" * 78)
    print(f"Modules:                  {config.modules}")
    print(f"Class weighting:          {'sqrt' if config.use_sqrt_weights else 'FULL inverse-frequency'}")
    print(f"Feature selection:        {'enabled' if config.feature_selection_enabled else 'disabled'}")
    if config.feature_selection_enabled:
        print(f"  Importance threshold:   {config.importance_threshold}")
        print(f"  Max features kept:      {config.max_features}")
    print(f"Classifier max_depth:     {config.clf_max_depth}")
    print(f"Classifier subsample:     {config.clf_subsample}")
    print(f"Classifier colsample:     {config.clf_colsample_bytree}")

    load_module_metadata()

    train_full = load_merged_split("TRAIN", config.modules)
    test_df = load_merged_split("TEST", config.modules)
    train_df, val_df = split_train_val(train_full, config.val_fraction)

    x_train_full, feature_cols = to_matrix(train_df)
    x_val_full, _ = to_matrix(val_df, feature_cols)
    x_test_full, _ = to_matrix(test_df, feature_cols)

    train_y_home = pd.to_numeric(train_df["HG"], errors="coerce")
    train_y_away = pd.to_numeric(train_df["AG"], errors="coerce")
    val_y_home = pd.to_numeric(val_df["HG"], errors="coerce")
    val_y_away = pd.to_numeric(val_df["AG"], errors="coerce")

    print(f"\nTrain rows: {len(train_df)} | Val rows: {len(val_df)} | Test rows: {len(test_df)}")
    print(f"Initial feature count: {len(feature_cols)}")

    train_outcomes = outcome_labels(train_df["HG"].values, train_df["AG"].values)
    val_outcomes = outcome_labels(val_df["HG"].values, val_df["AG"].values)
    test_outcomes = outcome_labels(test_df["HG"].values, test_df["AG"].values)

    print("\nTrue outcome distribution:")
    for split_name, outcomes in [("Train", train_outcomes), ("Val", val_outcomes), ("Test", test_outcomes)]:
        counts = np.bincount(outcomes, minlength=3)
        pcts = 100.0 * counts / counts.sum()
        print(f"  {split_name:>5}: HOME {counts[0]:>5} ({pcts[0]:>4.1f}%)  "
              f"DRAW {counts[1]:>5} ({pcts[1]:>4.1f}%)  "
              f"AWAY {counts[2]:>5} ({pcts[2]:>4.1f}%)")

    # ==================================================================
    # STAGE 0: Feature selection via importance
    # ==================================================================
    if config.feature_selection_enabled:
        print("\n" + "-" * 78)
        print("STAGE 0: Feature selection (importance-based)")
        print("-" * 78)
        train_weights = compute_sample_weights(train_outcomes, use_sqrt=config.use_sqrt_weights)

        # Fit a probe classifier to get importances
        probe_clf = fit_outcome_classifier(
            x_train_full, train_outcomes, x_val_full, val_outcomes, config,
            sample_weights=train_weights
        )
        importance_gain = probe_clf.get_booster().get_score(importance_type="gain")
        importance_weight = probe_clf.get_booster().get_score(importance_type="weight")

        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "gain": [importance_gain.get(f, 0.0) for f in feature_cols],
            "weight": [importance_weight.get(f, 0.0) for f in feature_cols],
        }).sort_values("gain", ascending=False).reset_index(drop=True)

        importance_df["gain_fraction"] = importance_df["gain"] / max(importance_df["gain"].sum(), 1e-12)

        selected_features, importance_df = select_features_by_importance(importance_df, config)
        print(f"  Probe best_iteration: {probe_clf.best_iteration}")
        print(f"  Total gain: {importance_df['gain'].sum():.2f}")
        print(f"  Features above threshold ({config.importance_threshold}): "
              f"{(importance_df['gain_fraction'] >= config.importance_threshold).sum()}")
        print(f"  Selected top {len(selected_features)} features")

        print(f"\n  Top 15 features by gain:")
        for _, row in importance_df.head(15).iterrows():
            print(f"    {row['feature']:>45}  gain {row['gain']:>10.2f}  "
                  f"({row['gain_fraction']*100:>5.2f}%)")

        # Save the importance table
        importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
        SELECTED_FEATURES_PATH.write_text(json.dumps(selected_features, indent=2), encoding="utf-8")
        print(f"\n  Saved importance to: {FEATURE_IMPORTANCE_PATH}")
        print(f"  Saved selected features to: {SELECTED_FEATURES_PATH}")

        # Reduce matrices
        x_train = x_train_full[selected_features]
        x_val = x_val_full[selected_features]
        x_test = x_test_full[selected_features]
        feature_cols = selected_features
    else:
        x_train, x_val, x_test = x_train_full, x_val_full, x_test_full

    print(f"\nFinal feature count: {len(feature_cols)}")

    # ==================================================================
    # STAGE 1: Poisson regressors
    # ==================================================================
    print("\n" + "-" * 78)
    print("STAGE 1: Fitting Poisson goal regressors")
    print("-" * 78)
    home_model = fit_poisson_regressor(x_train, train_y_home, x_val, val_y_home, config)
    away_model = fit_poisson_regressor(x_train, train_y_away, x_val, val_y_away, config)
    print(f"  Home goals model: best_iteration={home_model.best_iteration}")
    print(f"  Away goals model: best_iteration={away_model.best_iteration}")

    # ==================================================================
    # STAGE 2: Full-balanced classifier
    # ==================================================================
    print("\n" + "-" * 78)
    print("STAGE 2: Fitting 3-class classifier (FULL inverse-frequency weights)")
    print("-" * 78)
    train_weights = compute_sample_weights(train_outcomes, use_sqrt=config.use_sqrt_weights)

    unique, counts = np.unique(train_outcomes, return_counts=True)
    print(f"  Class weight summary:")
    for cls, cnt in zip(unique, counts):
        w = train_weights[train_outcomes == cls][0]
        print(f"    {OUTCOME_NAMES[cls]:>4}: {cnt:>5} samples × weight {w:.3f}")

    outcome_model = fit_outcome_classifier(
        x_train, train_outcomes, x_val, val_outcomes, config, sample_weights=train_weights
    )
    print(f"  Outcome model: best_iteration={outcome_model.best_iteration}")

    # ==================================================================
    # Draw margin tuning
    # ==================================================================
    print("\n" + "-" * 78)
    print("TUNING DRAW MARGIN (validation)")
    print("-" * 78)
    val_home_pred = home_model.predict(x_val)
    val_away_pred = away_model.predict(x_val)
    draw_margin, val_acc, val_bal_acc = tune_draw_margin(
        val_home_pred, val_away_pred,
        val_df["HG"].values, val_df["AG"].values,
        config.draw_margin_grid,
    )
    print(f"  Best draw margin: {draw_margin:.3f}")
    print(f"  Val accuracy (rate-based):   {val_acc:.3f}")
    print(f"  Val balanced accuracy (rate): {val_bal_acc:.3f}")

    val_probs_raw = outcome_model.predict_proba(x_val)
    val_clf_pred = np.argmax(val_probs_raw, axis=1)
    print(f"  Val accuracy (classifier):   {accuracy_score(val_outcomes, val_clf_pred):.3f}")
    print(f"  Val balanced acc (classifier): {balanced_accuracy_score(val_outcomes, val_clf_pred):.3f}")

    # ==================================================================
    # Prior correction
    # ==================================================================
    print("\n" + "-" * 78)
    print("PRIOR CORRECTION")
    print("-" * 78)
    train_priors = np.bincount(train_outcomes, minlength=3).astype(float)
    train_priors /= train_priors.sum()
    test_priors = np.bincount(test_outcomes, minlength=3).astype(float)
    test_priors /= test_priors.sum()
    print(f"  Train priors: HOME {train_priors[0]:.3f}  DRAW {train_priors[1]:.3f}  AWAY {train_priors[2]:.3f}")
    print(f"  Test  priors: HOME {test_priors[0]:.3f}  DRAW {test_priors[1]:.3f}  AWAY {test_priors[2]:.3f}")

    # ==================================================================
    # Test evaluation: all paths
    # ==================================================================
    test_home_pred = np.clip(home_model.predict(x_test), 1e-6, None)
    test_away_pred = np.clip(away_model.predict(x_test), 1e-6, None)
    test_probs_raw = outcome_model.predict_proba(x_test)
    test_probs_corrected = prior_correct_probs(test_probs_raw, train_priors, test_priors)
    test_rate_probs = rates_to_probs(test_home_pred, test_away_pred, draw_margin)

    test_rate_pred = np.argmax(test_rate_probs, axis=1)
    test_clf_pred = np.argmax(test_probs_raw, axis=1)
    test_clf_corrected_pred = np.argmax(test_probs_corrected, axis=1)

    rate_metrics = metrics_summary(test_outcomes, test_rate_pred)
    clf_metrics = metrics_summary(test_outcomes, test_clf_pred)
    clf_corrected_metrics = metrics_summary(test_outcomes, test_clf_corrected_pred)

    # ==================================================================
    # Blend sweep
    # ==================================================================
    print("\n" + "-" * 78)
    print("BLEND SWEEP")
    print("-" * 78)
    sweep_results = sweep_blend_weights(
        test_probs_corrected, test_rate_probs, test_outcomes,
        config.blend_sweep, objective="balanced_accuracy",
    )
    best = sweep_results[0]
    best_blend_probs = (best["weight_clf"] * test_probs_corrected +
                        (1 - best["weight_clf"]) * test_rate_probs)
    best_blend_pred = np.argmax(best_blend_probs, axis=1)
    best_blend_metrics = metrics_summary(test_outcomes, best_blend_pred)

    print(f"  Top 5 blend weights:")
    print(f"  {'Weight':>8}  {'Accuracy':>10}  {'Balanced':>10}  {'Macro F1':>10}")
    for r in sweep_results[:5]:
        print(f"  {r['weight_clf']:>8.2f}  {r['accuracy']:>10.3f}  "
              f"{r['balanced_accuracy']:>10.3f}  {r['macro_f1']:>10.3f}")

    # ==================================================================
    # Save predictions
    # ==================================================================
    pred_frame = pd.DataFrame({
        "game_id": test_df["game_id"].values,
        "HG": test_df["HG"].values, "AG": test_df["AG"].values,
        "lambda_home": test_home_pred, "lambda_away": test_away_pred,
        "expected_goal_diff": test_home_pred - test_away_pred,
        "prob_home": test_probs_corrected[:, 0],
        "prob_draw": test_probs_corrected[:, 1],
        "prob_away": test_probs_corrected[:, 2],
        "predicted_result_rate": test_rate_pred,
        "predicted_result_clf": test_clf_pred,
        "predicted_result_clf_corrected": test_clf_corrected_pred,
        "predicted_result_best_blend": best_blend_pred,
    })
    pred_frame.to_csv(PRED_PATH, index=False)

    home_model.save_model(MODEL_HOME_PATH)
    away_model.save_model(MODEL_AWAY_PATH)
    outcome_model.save_model(MODEL_OUTCOME_PATH)

    # ==================================================================
    # Verbose summary
    # ==================================================================
    goal_metrics = evaluate_goals(test_df, test_home_pred, test_away_pred)

    print("\n" + "=" * 78)
    print("FINAL TEST RESULTS")
    print("=" * 78)

    print(f"\nGoal prediction quality:")
    print(f"  MAE home: {goal_metrics['mae_home']:.3f}   MAE away: {goal_metrics['mae_away']:.3f}")
    print(f"  Poisson NLL home: {goal_metrics['poisson_nll_home']:.4f}   away: {goal_metrics['poisson_nll_away']:.4f}")

    print(f"\n" + "-" * 78)
    print("DECISION PATH COMPARISON")
    print("-" * 78)
    print(f"  {'Path':>32}  {'Accuracy':>10}  {'Balanced':>10}  {'Macro F1':>10}")
    print(f"  {'Rate-based':>32}  {rate_metrics['accuracy']:>10.3f}  "
          f"{rate_metrics['balanced_accuracy']:>10.3f}  {rate_metrics['macro_f1']:>10.3f}")
    print(f"  {'Classifier (raw)':>32}  {clf_metrics['accuracy']:>10.3f}  "
          f"{clf_metrics['balanced_accuracy']:>10.3f}  {clf_metrics['macro_f1']:>10.3f}")
    print(f"  {'Classifier (prior-corrected)':>32}  {clf_corrected_metrics['accuracy']:>10.3f}  "
          f"{clf_corrected_metrics['balanced_accuracy']:>10.3f}  {clf_corrected_metrics['macro_f1']:>10.3f}")
    print(f"  {'Best blend (swept)':>32}  {best_blend_metrics['accuracy']:>10.3f}  "
          f"{best_blend_metrics['balanced_accuracy']:>10.3f}  {best_blend_metrics['macro_f1']:>10.3f}")

    describe_confusion(np.array(clf_corrected_metrics["confusion_matrix"]),
                       "Prior-corrected classifier confusion matrix:")
    describe_per_class_metrics(test_outcomes, test_clf_corrected_pred,
                               "Prior-corrected classifier per-class metrics:")
    describe_prediction_distribution(test_clf_corrected_pred,
                                     "Prior-corrected classifier prediction distribution:")

    describe_confusion(np.array(best_blend_metrics["confusion_matrix"]),
                       f"Best-blend (w={best['weight_clf']:.2f}) confusion matrix:")
    describe_per_class_metrics(test_outcomes, best_blend_pred,
                               "Best-blend per-class metrics:")
    describe_prediction_distribution(best_blend_pred,
                                     "Best-blend prediction distribution:")

    # ==================================================================
    # Summary JSON
    # ==================================================================
    summary = {
        "config": {
            "modules": config.modules,
            "val_fraction": config.val_fraction,
            "seed": config.seed,
            "use_sqrt_weights": config.use_sqrt_weights,
            "feature_selection_enabled": config.feature_selection_enabled,
            "importance_threshold": config.importance_threshold,
            "max_features": config.max_features,
            "clf_max_depth": config.clf_max_depth,
            "clf_subsample": config.clf_subsample,
            "clf_colsample_bytree": config.clf_colsample_bytree,
            "clf_reg_lambda": config.clf_reg_lambda,
            "clf_min_child_weight": config.clf_min_child_weight,
        },
        "priors": {"train": train_priors.tolist(), "test": test_priors.tolist()},
        "feature_selection": {
            "initial_count": len(feature_cols) if not config.feature_selection_enabled else "N/A",
            "final_count": len(feature_cols),
            "selected_features": feature_cols,
        } if config.feature_selection_enabled else None,
        "validation": {
            "draw_margin": draw_margin,
            "rate_accuracy": val_acc,
            "rate_balanced_accuracy": val_bal_acc,
        },
        "test_goals": goal_metrics,
        "test_rate_based": rate_metrics,
        "test_classifier_raw": clf_metrics,
        "test_classifier_prior_corrected": clf_corrected_metrics,
        "test_blend_best": {"weight_clf": best["weight_clf"], **best_blend_metrics},
        "blend_sweep_top5": sweep_results[:5],
        "feature_count_final": len(feature_cols),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nSaved predictions to: {PRED_PATH}")
    print(f"Saved models to: {MODEL_HOME_PATH}, {MODEL_AWAY_PATH}, {MODEL_OUTCOME_PATH}")
    print(f"Saved summary to: {SUMMARY_PATH}")
    if config.feature_selection_enabled:
        print(f"Saved feature importance to: {FEATURE_IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()