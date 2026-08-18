#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Add unified game_id to scraped match-statistics data.

Input is the raw scrape CSV produced by a_load_data.py. This script does not
modify that file; it writes a new fixed file with a `game_id` column.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
sys.path.append(str(SRC_DIR))

from team_name_mapping_FINAL import resolve_team_name_fuzzy, soccer_teams  # noqa: E402


INPUT_PATH = PROJECT_DIR / "data" / "big5_matches.csv"
OUTPUT_PATH = PROJECT_DIR / "data" / "big5_matches_fixed.csv"
UNMATCHED_PATH = PROJECT_DIR / "data" / "unmatched_match_stats_teams.csv"

MONTH_TOKEN_PATTERN = re.compile(
	r"(January|February|March|April|May|June|July|August|September|October|November|December)-\d{1,2}-\d{4}",
	flags=re.IGNORECASE,
)

NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")

LOCAL_ALIASES = {
	"Manchester Utd": "MUN",
	"PSG": "PSG",
	"Frankfurt": "SGE",
	"Gladbach": "BMG",
	"Hellas Verona": "VER",
	"Hertha BSC": "BSC",
	"Nottingham": "NFO",
	"Atlético Madrid": "ATM",
	"Köln": "KOE",
	"Alavés": "ALA",
	"Saint-Étienne": "STE",
	"Rayo Vallecano": "RAY",
	"Dep. La Coruña": "DEP",
	"Málaga": "MAL",
	"Cádiz": "CAD",
	"Almería": "ALM",
	"Nîmes": "NIM",
	"Leganés": "LEG",
	"Darmstadt 98": "D98",
}


def normalize_name(name: str) -> str:
	text = unicodedata.normalize("NFKD", name)
	text = "".join(ch for ch in text if not unicodedata.combining(ch))
	text = text.lower().strip()
	text = NON_ALNUM_PATTERN.sub(" ", text)
	return " ".join(text.split())


NORMALIZED_LOOKUP = {}
for team_name, code in soccer_teams.items():
	NORMALIZED_LOOKUP.setdefault(normalize_name(str(team_name)), code)


def extract_date_from_url(url_value: str) -> pd.Timestamp:
	if pd.isna(url_value):
		return pd.NaT
	url_text = str(url_value)
	match = MONTH_TOKEN_PATTERN.search(url_text)
	if not match:
		return pd.NaT
	return pd.to_datetime(match.group(0), format="%B-%d-%Y", errors="coerce")


def resolve_team_code(name_value: str) -> str | None:
	if pd.isna(name_value):
		return None
	name_text = str(name_value).strip()

	code = resolve_team_name_fuzzy(name_text)
	if code:
		return code

	alias_code = LOCAL_ALIASES.get(name_text)
	if alias_code:
		return alias_code

	return NORMALIZED_LOOKUP.get(normalize_name(name_text))


def build_game_id(row: pd.Series) -> str | None:
	if pd.isna(row["match_date"]) or pd.isna(row["home_code"]) or pd.isna(row["away_code"]):
		return None
	date_token = row["match_date"].strftime("%Y%m%d")
	return f"{date_token}_{row['home_code']}_{row['away_code']}"


def main() -> None:
	if not INPUT_PATH.exists():
		raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

	df = pd.read_csv(INPUT_PATH, low_memory=False)
	required = {"match_id", "url", "home_team", "away_team"}
	missing = sorted(required.difference(df.columns))
	if missing:
		raise KeyError(f"Missing required columns in {INPUT_PATH.name}: {missing}")

	out = df.copy()
	out["home_code"] = out["home_team"].map(resolve_team_code)
	out["away_code"] = out["away_team"].map(resolve_team_code)
	out["match_date"] = out["url"].apply(extract_date_from_url)
	out["game_id"] = out.apply(build_game_id, axis=1)

	unmatched = out.loc[
		out["game_id"].isna(),
		["match_id", "url", "home_team", "away_team", "home_code", "away_code", "match_date"],
	].copy()
	unmatched.to_csv(UNMATCHED_PATH, index=False)

	out = out.drop(columns=["home_code", "away_code", "match_date"])
	out.to_csv(OUTPUT_PATH, index=False)

	total_rows = len(out)
	matched_rows = int(out["game_id"].notna().sum())
	unmatched_rows = total_rows - matched_rows

	print(f"Loaded: {total_rows} rows from {INPUT_PATH}")
	print(f"Matched game_id: {matched_rows}")
	print(f"Unmatched rows: {unmatched_rows}")
	print(f"Saved fixed file to: {OUTPUT_PATH}")
	print(f"Saved unmatched report to: {UNMATCHED_PATH}")


if __name__ == "__main__":
	main()
