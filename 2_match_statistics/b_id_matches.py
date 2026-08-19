#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Add unified game_id and xG columns to scraped match-statistics data.

Input is the raw scrape CSV produced by a_load_data.py. This script does not
modify that file; it writes a new fixed file with a `game_id` column and
home/away xG values merged from data/big5_xg_clean.csv.
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
XG_INPUT_PATH = PROJECT_DIR / "data" / "big5_xg_clean.csv"

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


def prepare_xg_frame(path: Path) -> pd.DataFrame:
	if not path.exists():
		raise FileNotFoundError(f"xG file not found: {path}")

	xg = pd.read_csv(path, low_memory=False)
	if "home_xg" not in xg.columns or "away_xg" not in xg.columns:
		raise KeyError("xG file must contain home_xg and away_xg columns.")

	if "game_id" not in xg.columns:
		xg = xg.copy()
		xg["home_code"] = xg.get("home_team_mapped")
		xg["away_code"] = xg.get("away_team_mapped")

		if "home_team" in xg.columns:
			xg["home_code"] = xg["home_code"].where(xg["home_code"].notna(), xg["home_team"].map(resolve_team_code))
		if "away_team" in xg.columns:
			xg["away_code"] = xg["away_code"].where(xg["away_code"].notna(), xg["away_team"].map(resolve_team_code))

		xg["match_date"] = pd.to_datetime(xg.get("date"), errors="coerce")
		xg["game_id"] = xg.apply(
			lambda r: f"{r['match_date'].strftime('%Y%m%d')}_{r['home_code']}_{r['away_code']}"
			if pd.notna(r["match_date"]) and pd.notna(r["home_code"]) and pd.notna(r["away_code"])
			else None,
			axis=1,
		)

	xg = xg[["game_id", "home_xg", "away_xg"]].copy()
	xg["home_xg"] = pd.to_numeric(xg["home_xg"], errors="coerce")
	xg["away_xg"] = pd.to_numeric(xg["away_xg"], errors="coerce")
	xg = xg.dropna(subset=["game_id"]).drop_duplicates(subset=["game_id"], keep="first")
	return xg


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

	xg = prepare_xg_frame(XG_INPUT_PATH)
	out = out.merge(xg, on="game_id", how="left")

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
	xg_matched_rows = int(out["home_xg"].notna().sum() + out["away_xg"].notna().sum())
	xg_cells_total = int(total_rows * 2)

	print(f"Loaded: {total_rows} rows from {INPUT_PATH}")
	print(f"Matched game_id: {matched_rows}")
	print(f"Unmatched rows: {unmatched_rows}")
	print(f"xG coverage: {xg_matched_rows}/{xg_cells_total} home/away xG cells matched")
	print(f"Saved fixed file to: {OUTPUT_PATH}")
	print(f"Saved unmatched report to: {UNMATCHED_PATH}")


if __name__ == "__main__":
	main()
