#!/usr/bin/env python3
"""Run the module-specific data pipeline for the neural-network project.

Usage examples:
  python run_data_pipeline.py module1
  python run_data_pipeline.py module2
  python run_data_pipeline.py module3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent

MODULE_PIPELINES: dict[str, list[Path]] = {
    "module1": [
        PROJECT_DIR / "1_player_stats" / "01_load_market_values.py",
        PROJECT_DIR / "1_player_stats" / "02_load_fifa_ratings.py",
        PROJECT_DIR / "1_player_stats" / "a1_scrape_new_lineups.py",
        PROJECT_DIR / "1_player_stats" / "a2_extract_historic_lineups.py",
        PROJECT_DIR / "1_player_stats" / "a3_filter_lineups_by_competition.py",
        PROJECT_DIR / "1_player_stats" / "b_get_market_values.py",
        PROJECT_DIR / "1_player_stats" / "c_get_fifa_ratings.py",
        PROJECT_DIR / "1_player_stats" / "d_get_attendance_and_position.py",
        PROJECT_DIR / "1_player_stats" / "e_scrape_xg.py",
        PROJECT_DIR / "1_player_stats" / "f_get_form.py",
        PROJECT_DIR / "1_player_stats" / "h_player_group_pca.py",
        PROJECT_DIR / "1_player_stats" / "h2_create_skill_matchups.py",
    ],
    "module2": [
        PROJECT_DIR / "2_match_statistics" / "a_load_data.py",
        PROJECT_DIR / "2_match_statistics" / "b_id_matches.py",
        PROJECT_DIR / "2_match_statistics" / "c_create_match_statistic_pca.py",
    ],
    "module3": [
        PROJECT_DIR / "3_lineup_embeddings" / "a_create_lineup_embedding.py",
        PROJECT_DIR / "3_lineup_embeddings" / "b_create_season_and_league_embedding.py",
    ],
}


def run_script(script_path: Path) -> None:
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    print(f"\n{'=' * 80}")
    print(f"Running: {script_path.relative_to(PROJECT_DIR)}")
    print(f"{'=' * 80}")
    subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_DIR, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a module-specific data pipeline.")
    parser.add_argument(
        "module",
        choices=sorted(MODULE_PIPELINES),
        help="Which pipeline to run: module1, module2, or module3.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = MODULE_PIPELINES[args.module]

    print(f"Selected pipeline: {args.module}")
    print(f"Scripts to run: {len(pipeline)}")

    for script_path in pipeline:
        run_script(script_path)

    print(f"\nCompleted pipeline: {args.module}")


if __name__ == "__main__":
    main()