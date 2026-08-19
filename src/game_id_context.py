#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Helpers to infer league and season information from game_id.

game_id format is expected to be: YYYYMMDD_HOME_AWAY.

Notes:
- Season can be inferred from the date token directly.
- League cannot be inferred from the token alone, so this module resolves it
  via a lookup table built from data/big5_matches_fixed.csv.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MATCHES_PATH = PROJECT_DIR / "data" / "big5_matches_fixed.csv"


def _season_from_date(match_date: pd.Timestamp) -> str | None:
    if pd.isna(match_date):
        return None
    start_year = int(match_date.year) if int(match_date.month) >= 7 else int(match_date.year) - 1
    return f"{start_year}-{start_year + 1}"


def _parse_game_id_date(game_id: str) -> pd.Timestamp:
    parts = str(game_id).split("_", 2)
    if len(parts) < 3:
        return pd.NaT
    return pd.to_datetime(parts[0], format="%Y%m%d", errors="coerce")


@lru_cache(maxsize=4)
def _build_game_lookup(matches_path: str) -> dict[str, dict[str, Any]]:
    path = Path(matches_path)
    if not path.exists():
        return {}

    df = pd.read_csv(path, low_memory=False)
    required = {"game_id", "league", "season"}
    if not required.issubset(df.columns):
        return {}

    out: dict[str, dict[str, Any]] = {}
    subset = df[["game_id", "league", "season"]].dropna(subset=["game_id"]).copy()

    for row in subset.itertuples(index=False):
        gid = str(row.game_id)
        if gid not in out:
            out[gid] = {"league": row.league, "season": row.season}

    return out


def infer_league_and_season(game_id: str, matches_path: Path | str = DEFAULT_MATCHES_PATH) -> dict[str, Any]:
    """Infer league and season for a given game_id.

    Args:
        game_id: Match identifier formatted as YYYYMMDD_HOME_AWAY.
        matches_path: Optional CSV path with columns game_id, league, season.

    Returns:
        A dict with keys: game_id, match_date, league, season.
        - league is resolved from lookup when available, otherwise None.
        - season is taken from lookup first, then inferred from date fallback.
    """

    match_date = _parse_game_id_date(game_id)
    lookup = _build_game_lookup(str(matches_path))
    found = lookup.get(str(game_id), {})

    season_from_lookup = found.get("season")
    season_fallback = _season_from_date(match_date)

    return {
        "game_id": game_id,
        "match_date": None if pd.isna(match_date) else match_date.date().isoformat(),
        "league": found.get("league"),
        "season": season_from_lookup if pd.notna(season_from_lookup) else season_fallback,
    }
