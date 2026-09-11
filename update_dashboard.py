#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run the entire dashboard pipeline in order.

This script lives at the project root and orchestrates the 11 pipeline scripts
inside 5_new_prediction_dashboard/, ensuring:

- The correct Python interpreter is used (the one running this script)
- The working directory is the project root for every subprocess
- PYTHONPATH is set so `from src.…` imports work
- The pipeline fails fast on the first error
- A clean, timestamped log is printed to the console
- Optional: `--log-file path` to also write the log to disk
- Optional: `--from <script>` to resume from a specific stage
- Optional: `--only <script>` to run a single stage

Usage examples:
    python update_dashboard.py
    python update_dashboard.py --log-file weekly_pipeline.log
    python update_dashboard.py --from h_create_lineup_embedding.py
    python update_dashboard.py --only k_make_predictions.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ============================================================================
# PATHS
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR / "5_new_prediction_dashboard"
WEEKLY_RESULTS_DIR = PIPELINE_DIR / "weekly_results"
FINAL_OUTPUT = WEEKLY_RESULTS_DIR / "weekly_predictions.csv"


# ============================================================================
# PIPELINE DEFINITION
# ============================================================================

# Ordered list of (script_filename, human_readable_label)
PIPELINE = [
    ("a_scrape_upcoming_lineup.py",     "Scraping upcoming fixtures"),
    ("b_get_upcoming_market_values.py", "Computing market values"),
    ("c_get_upcoming_fifa_ratings.py",  "Computing FIFA ratings"),
    ("d_get_upcoming_xg_values.py",     "Computing xG form"),
    ("e_apply_pca.py",                  "Applying player-stat PCA"),
    ("f_create_skill_matchups.py",      "Building skill matchups"),
    ("g_load_match_stats.py",           "Building match statistics"),
    ("h_create_lineup_embedding.py",    "Encoding lineups"),
    ("i_season_league_embedding.py",    "Encoding league / season context"),
    ("j_data_for_model.py",             "Assembling model inputs"),
    ("k_make_predictions.py",           "Training model & predicting"),
]


# ============================================================================
# LOGGING
# ============================================================================

class Tee:
    """Write to both stdout and an optional file."""
    def __init__(self, file_path: Path | None = None):
        self.file = None
        if file_path is not None:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            self.file = open(file_path, "a", encoding="utf-8")

    def write(self, text: str):
        sys.stdout.write(text)
        sys.stdout.flush()
        if self.file is not None:
            self.file.write(text)
            self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()


def log(tee: Tee, msg: str = ""):
    tee.write(msg + "\n")


# ============================================================================
# RUNNER
# ============================================================================

def run_stage(script_name: str, label: str, tee: Tee, index: int, total: int) -> bool:
    """Run a single pipeline stage. Returns True on success."""
    script_path = PIPELINE_DIR / script_name

    header = f"[{index}/{total}] {label}"
    log(tee, "")
    log(tee, "=" * 78)
    log(tee, header)
    log(tee, "=" * 78)

    if not script_path.exists():
        log(tee, f"  ✗ Script not found: {script_path}")
        return False

    # Environment: ensure src/ imports resolve
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SCRIPT_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"

    start = time.time()
    try:
        # cwd = project root so relative paths inside scripts behave predictably
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(SCRIPT_DIR),
            env=env,
            check=False,
        )
    except Exception as e:
        log(tee, f"  ✗ Failed to launch: {e}")
        return False

    elapsed = time.time() - start

    if result.returncode == 0:
        log(tee, f"  ✓ Completed in {elapsed:.1f}s")
        return True
    else:
        log(tee, f"  ✗ Failed with exit code {result.returncode} after {elapsed:.1f}s")
        return False


# ============================================================================
# MAIN
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the full dashboard pipeline.")
    p.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional path to also write the pipeline log to disk (append mode).",
    )
    p.add_argument(
        "--from",
        dest="from_script",
        type=str,
        default=None,
        help="Start from this script (inclusive). Must match a filename in the pipeline.",
    )
    p.add_argument(
        "--only",
        type=str,
        default=None,
        help="Run only this single script. Must match a filename in the pipeline.",
    )
    p.add_argument(
        "--no-banner",
        action="store_true",
        help="Suppress the startup banner.",
    )
    return p.parse_args()


def select_stages(args: argparse.Namespace):
    """Return the list of (script, label) tuples to run based on CLI args."""
    if args.only:
        matches = [p for p in PIPELINE if p[0] == args.only]
        if not matches:
            raise SystemExit(f"Unknown script for --only: {args.only}")
        return matches

    if args.from_script:
        idx = next((i for i, p in enumerate(PIPELINE) if p[0] == args.from_script), None)
        if idx is None:
            raise SystemExit(f"Unknown script for --from: {args.from_script}")
        return PIPELINE[idx:]

    return PIPELINE


def main() -> int:
    args = parse_args()

    log_file = Path(args.log_file).resolve() if args.log_file else None
    tee = Tee(log_file)

    try:
        stages = select_stages(args)

        if not args.no_banner:
            log(tee, "")
            log(tee, "=" * 78)
            log(tee, "  UPDATE DASHBOARD — full pipeline runner")
            log(tee, "=" * 78)
            log(tee, f"  Started:       {datetime.now():%Y-%m-%d %H:%M:%S}")
            log(tee, f"  Project root:  {SCRIPT_DIR}")
            log(tee, f"  Python:        {sys.executable}")
            log(tee, f"  Pipeline dir:  {PIPELINE_DIR}")
            log(tee, f"  Stages:        {len(stages)} of {len(PIPELINE)}")
            if log_file:
                log(tee, f"  Log file:      {log_file}")
            log(tee, "")

        overall_start = time.time()
        failed_stage = None

        for i, (script, label) in enumerate(stages, start=1):
            ok = run_stage(script, label, tee, i, len(stages))
            if not ok:
                failed_stage = (script, label)
                break

        overall_elapsed = time.time() - overall_start

        log(tee, "")
        log(tee, "=" * 78)
        if failed_stage is None:
            log(tee, f"  ✓ PIPELINE COMPLETE — {len(stages)} stages in {overall_elapsed:.1f}s")
            if FINAL_OUTPUT.exists():
                mtime = datetime.fromtimestamp(FINAL_OUTPUT.stat().st_mtime)
                log(tee, f"  Predictions updated: {FINAL_OUTPUT}")
                log(tee, f"  File mtime:          {mtime:%Y-%m-%d %H:%M:%S}")
            log(tee, "=" * 78)
            return 0
        else:
            log(tee, f"  ✗ PIPELINE FAILED at stage: {failed_stage[1]}")
            log(tee, f"    Script: {failed_stage[0]}")
            log(tee, f"    Elapsed: {overall_elapsed:.1f}s")
            log(tee, "=" * 78)
            return 1
    finally:
        tee.close()


if __name__ == "__main__":
    sys.exit(main())