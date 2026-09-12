#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Prepare all data the Shiny dashboard reads.

Reads:
    ./weekly_results/lstm_predictions.csv     (raw predictions from k_train_lstm.py)
    ./upcoming_lineups/upcoming_lineups_latest.csv
    ../data/big5_matches_fixed.csv            (historical results + league fallback)

Writes (weekly_results/ — append-only history, one per matchday week):
    weekly_predictions_latest.csv             (always the newest clean summary)
    weekly_predictions_YYYY-MM-DD.csv         (one per matchday week; the date
                                               is the Monday of that week, NOT
                                               the run date, so running twice
                                               in the same week overwrites the
                                               same file)
    lstm_summary.json                         (metadata shown in the "Model
                                               Summary" panel of the app)

Every weekly_predictions_*.csv gets three ground-truth columns filled from
the historical results table:
    actual_home_goals, actual_away_goals, actual_outcome
On a fresh run they are empty; every time this script runs, it refreshes them
from ../data/big5_matches_fixed.csv for ALL dated files (not just the current
matchday), so past weeks get populated automatically once the matches have
been played.

If lstm_predictions.csv is missing, this script automatically invokes
k_train_lstm.py to produce it first.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make our own directory importable so `import k_train_lstm` works regardless
# of the current working directory or how the script is launched.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
for _p in (str(SCRIPT_DIR), str(PROJECT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json                    # noqa: E402
import subprocess              # noqa: E402
from datetime import datetime  # noqa: E402

import numpy as np             # noqa: E402
import pandas as pd            # noqa: E402

try:
    from k_train_lstm import (  # noqa: E402
        ACTUAL_COLS,
        MODEL_PATH,
        OUTCOME_LABEL,
        PRED_PATH,
        SUMMARY_PATH,
        WEEKLY_RESULTS_DIR,
        build_team_to_league_from_history,
        dated_output_paths,
        extract_team_codes_from_game_id,
        load_actual_results_from_history,
        load_upcoming_context,
        resolve_league_for_game,
        update_actuals_from_results,
        LEAGUE_DISPLAY,
    )
except ModuleNotFoundError as e:
    raise SystemExit(
        f"Could not import k_train_lstm.\n"
        f"Expected file: {SCRIPT_DIR / 'k_train_lstm.py'}\n"
        f"Make sure it exists at exactly that path.\n"
        f"Original error: {e}"
    )


# If True, every run overwrites any manual corrections you've made with the
# values from big5_matches_fixed.csv. If False (default), only empty cells
# are filled.
OVERWRITE_ACTUALS = False


def ensure_raw_predictions_exist():
    """If lstm_predictions.csv is missing, run k_train_lstm.py first."""
    if PRED_PATH.exists() and MODEL_PATH.exists():
        return
    print(f"⚠ {PRED_PATH.name} not found — running k_train_lstm.py first...")
    subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "k_train_lstm.py")],
        check=True,
    )


def build_weekly_table(pred_frame: pd.DataFrame) -> tuple[pd.DataFrame, int, dict]:
    """Resolve league + team names and produce the dashboard-facing table.

    The three ACTUAL_COLS are added as empty placeholders with the correct
    nullable dtypes. They get filled by populate_actuals_in_all_weekly_files()
    from ../data/big5_matches_fixed.csv.
    """
    team_lookup, league_lookup = load_upcoming_context()
    team_to_league = build_team_to_league_from_history()

    print(f"\nLeague resolution sources:")
    print(f"  Scraper Competition tags:   {len(league_lookup)} game_ids")
    print(f"  Team→league from history:   {len(team_to_league)} team codes")

    rows = []
    unresolved = 0

    for _, row in pred_frame.iterrows():
        gid = row["game_id"]
        home_code, away_code = extract_team_codes_from_game_id(gid)
        home_name = team_lookup.get(home_code, home_code) if home_code else home_code
        away_name = team_lookup.get(away_code, away_code) if away_code else away_code

        league_raw = resolve_league_for_game(gid, league_lookup, team_to_league)
        league_display = LEAGUE_DISPLAY.get(league_raw, league_raw)
        if league_raw == "unknown":
            unresolved += 1

        prob_h = float(row["prob_home"])
        prob_d = float(row["prob_draw"])
        prob_a = float(row["prob_away"])
        outcome_code = int(row["predicted_result"])
        confidence = max(prob_h, prob_d, prob_a)

        rows.append({
            "game_id": gid,
            "league": league_display,
            "home_team": home_name,
            "away_team": away_name,
            "predicted_home_goals": round(float(row["lambda_home"]), 2),
            "predicted_away_goals": round(float(row["lambda_away"]), 2),
            "predicted_goal_diff": round(float(row["expected_goal_diff"]), 2),
            "prob_home_win": round(prob_h, 3),
            "prob_draw": round(prob_d, 3),
            "prob_away_win": round(prob_a, 3),
            "predicted_outcome": OUTCOME_LABEL[outcome_code],
            "confidence": round(confidence, 3),
            # Ground-truth placeholders — filled later.
            "actual_home_goals": pd.NA,
            "actual_away_goals": pd.NA,
            "actual_outcome": pd.NA,
        })

    diag = {
        "from_scraper": len(league_lookup),
        "from_history": len(team_to_league),
    }

    df = pd.DataFrame(rows)
    # Force correct nullable dtypes so pandas doesn't downgrade them to
    # float64 / object (which would then reject string outcome values).
    df["actual_home_goals"] = df["actual_home_goals"].astype("Int64")
    df["actual_away_goals"] = df["actual_away_goals"].astype("Int64")
    df["actual_outcome"] = df["actual_outcome"].astype("string")

    return df, unresolved, diag


def merge_existing_actuals(new_df: pd.DataFrame, dated_path: Path) -> pd.DataFrame:
    """
    Carry over any non-empty actual_* values from an existing dated file
    into new_df (matched on game_id). This makes re-runs safe.
    """
    if not dated_path.exists():
        return new_df

    try:
        old = pd.read_csv(dated_path)
    except Exception:
        return new_df

    if old.empty or "game_id" not in old.columns:
        return new_df

    present = [c for c in ACTUAL_COLS if c in old.columns]
    if not present:
        return new_df

    mask = old[present].notna().any(axis=1)
    carried = old.loc[mask, ["game_id"] + present].copy()
    if carried.empty:
        return new_df

    preserved = new_df.drop(columns=ACTUAL_COLS, errors="ignore")
    merged = preserved.merge(carried, on="game_id", how="left")

    for c in ACTUAL_COLS:
        if c not in merged.columns:
            if c == "actual_outcome":
                merged[c] = pd.Series([pd.NA] * len(merged), dtype="string")
            else:
                merged[c] = pd.Series([pd.NA] * len(merged), dtype="Int64")

    # Re-assert dtypes after merge (merge can loosen them back to object/float).
    merged["actual_home_goals"] = pd.to_numeric(
        merged["actual_home_goals"], errors="coerce"
    ).astype("Int64")
    merged["actual_away_goals"] = pd.to_numeric(
        merged["actual_away_goals"], errors="coerce"
    ).astype("Int64")
    merged["actual_outcome"] = merged["actual_outcome"].astype("string")

    pred_cols = list(preserved.columns)
    merged = merged[pred_cols + ACTUAL_COLS]

    n_carried = int(merged[ACTUAL_COLS].notna().any(axis=1).sum())
    if n_carried > 0:
        print(f"  ✓ Preserved actuals for {n_carried} fixture(s) from "
              f"existing {dated_path.name}")

    return merged


def populate_actuals_in_all_weekly_files() -> None:
    """
    For EVERY dated weekly_predictions_*.csv in weekly_results/, fill in the
    actual_* columns from big5_matches_fixed.csv.

    - Fresh cells are filled.
    - Already-filled cells are left alone unless OVERWRITE_ACTUALS is True.
    - The 'latest' copy is skipped (it's refreshed from the current matchday
      at the end of main()).
    """
    print("\n" + "-" * 78)
    print("POPULATING ACTUAL RESULTS IN PAST WEEKLY FILES")
    print("-" * 78)

    results = load_actual_results_from_history()
    if not results:
        print("  (nothing to do — no historical results loaded)")
        return

    files = sorted(WEEKLY_RESULTS_DIR.glob("weekly_predictions_*.csv"))
    if not files:
        print("  (no weekly_predictions_*.csv files found)")
        return

    total_updated = 0
    files_touched = 0
    for path in files:
        # Skip the "latest" copy — handled separately in main().
        if path.name == "weekly_predictions_latest.csv":
            continue

        before = 0
        df_before = None
        try:
            df_before = pd.read_csv(path)
            if "actual_home_goals" in df_before.columns:
                before = int(df_before["actual_home_goals"].notna().sum())
        except Exception:
            df_before = None

        n = update_actuals_from_results(path, results, overwrite=OVERWRITE_ACTUALS)
        total_updated += n
        if n > 0:
            files_touched += 1
            total_rows = len(df_before) if df_before is not None else "?"
            print(f"  ✓ {path.name}: +{n} row(s) populated "
                  f"({before} → {before + n} of {total_rows})")

    print(f"\n  Total: {total_updated} row(s) populated across "
          f"{files_touched} file(s)")


def refresh_latest_from_current_matchday(dated_path: Path, latest_path: Path) -> None:
    """
    After the current matchday's dated file has been (re)written and
    populated, copy it verbatim to weekly_predictions_latest.csv.
    """
    if not dated_path.exists():
        return
    try:
        df = pd.read_csv(dated_path)
    except Exception as e:
        print(f"⚠ Could not refresh latest copy: {e}")
        return
    df.to_csv(latest_path, index=False, na_rep="")


def save_summary(weekly_df: pd.DataFrame,
                 dated_path: Path,
                 latest_path: Path,
                 unresolved: int,
                 diag: dict):
    """Write lstm_summary.json for the dashboard (merging with existing)."""
    base = {}
    if SUMMARY_PATH.exists():
        try:
            base = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        except Exception:
            base = {}

    if all(c in weekly_df.columns for c in ACTUAL_COLS):
        played = int(weekly_df[ACTUAL_COLS].notna().any(axis=1).sum())
    else:
        played = 0

    base.update({
        "run_date": datetime.now().isoformat(timespec="seconds"),
        "matchday_anchor": dated_path.stem.replace("weekly_predictions_", ""),
        "rows": {**base.get("rows", {}), "upcoming": len(weekly_df)},
        "played": played,
        "league_resolution": {
            "from_scraper": diag["from_scraper"],
            "from_history": diag["from_history"],
            "unresolved": unresolved,
        },
        "output_files": {
            **(base.get("output_files", {})),
            "dated_weekly": str(dated_path),
            "latest_weekly": str(latest_path),
            "raw_predictions": str(PRED_PATH),
            "model": str(MODEL_PATH),
        },
    })
    SUMMARY_PATH.write_text(json.dumps(base, indent=2), encoding="utf-8")


def main():
    WEEKLY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_raw_predictions_exist()

    pred_frame = pd.read_csv(PRED_PATH)
    if pred_frame.empty:
        print("⚠ lstm_predictions.csv is empty — nothing to prepare.")
        return

    # ---- 1. Build this matchday's table (empty actual columns) ----
    weekly_df, unresolved, diag = build_weekly_table(pred_frame)

    dated_path, latest_path = dated_output_paths(
        WEEKLY_RESULTS_DIR,
        game_ids=pred_frame["game_id"].tolist(),
    )

    dated_existed_before = dated_path.exists()

    # ---- 2. Carry over any actuals from a previous run of this matchday ----
    weekly_df = merge_existing_actuals(weekly_df, dated_path)

    weekly_df.to_csv(dated_path, index=False, na_rep="")

    if dated_existed_before:
        print(f"\n  ⚠ Overwrote existing matchday file: {dated_path.name}")
    else:
        print(f"\n  ✓ Created new matchday file:        {dated_path.name}")

    if unresolved > 0:
        print(f"\n⚠ {unresolved} fixtures could not be assigned a league.")

    # ---- 3. Auto-populate actuals for ALL dated files (past + current) ----
    populate_actuals_in_all_weekly_files()

    # ---- 4. Refresh the latest copy from the current matchday file,
    #         which now has actuals filled in where available ----
    refresh_latest_from_current_matchday(dated_path, latest_path)
    print(f"  Refreshed latest copy:              {latest_path.name}")

    # ---- 5. Re-read the dated file so the summary reflects the populated
    #         actuals, then write lstm_summary.json ----
    try:
        weekly_df_final = pd.read_csv(dated_path)
    except Exception:
        weekly_df_final = weekly_df
    save_summary(weekly_df_final, dated_path, latest_path, unresolved, diag)

    # ---- 6. Report ----
    print("\n" + "=" * 78)
    print("WEEKLY PREDICTIONS")
    print("=" * 78)
    print(weekly_df_final.to_string(index=False))

    print(f"\n  Saved {len(weekly_df_final)} upcoming fixtures.")
    print(f"  Predicted outcome distribution: "
          f"HOME={(weekly_df_final['predicted_outcome'] == 'HOME_WIN').sum()}  "
          f"DRAW={(weekly_df_final['predicted_outcome'] == 'DRAW').sum()}  "
          f"AWAY={(weekly_df_final['predicted_outcome'] == 'AWAY_WIN').sum()}")

    if "actual_outcome" in weekly_df_final.columns:
        played = weekly_df_final["actual_outcome"].notna().sum()
        if played > 0:
            hits = (weekly_df_final["predicted_outcome"]
                    == weekly_df_final["actual_outcome"]).sum()
            print(f"  Played: {played}  Correct: {hits}  "
                  f"Accuracy: {hits/played:.1%}")

    print("  League distribution:")
    for lg, count in weekly_df_final["league"].value_counts().items():
        print(f"    {lg}: {count}")

    print(f"\n  Saved matchday file to:         {dated_path}")
    print(f"  Saved latest weekly table to:   {latest_path}")
    print(f"  Saved summary to:               {SUMMARY_PATH}")


if __name__ == "__main__":
    main()