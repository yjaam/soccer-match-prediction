"""Scrape match-level statistics from FBref via soccerdata.

This script keeps the first version intentionally small:
- it uses soccerdata's FBref reader to discover the matches for a league/season
- it fetches the protected FBref match report through soccerdata's own session
- it returns one row per match with home/away statistics

FBref exposes possession and a compact team stats block directly in the match report.
Player-level summary tables provide shots, goals, and card totals.
By default, the script covers the Big 5 leagues from 2014 onward.
Use test mode to limit the scrape to a small handful of matches.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import re
from pathlib import Path
from typing import Any

import pandas as pd
import soccerdata as sd
from bs4 import BeautifulSoup

DEFAULT_LEAGUES = ["Big 5 European Leagues Combined"]
DEFAULT_SEASONS = list(range(2014, datetime.now().year))
DEFAULT_OUTPUT = Path("data/match_statistics_big5_since_2014.csv")
TEST_OUTPUT = Path("data/match_statistics_test.csv")
TEST_LIMIT = 5


def _slugify(label: str) -> str:
	return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _parse_percent(text: str) -> float | None:
	match = re.search(r"([0-9]+(?:\.[0-9]+)?)%", text)
	if match is None:
		return None
	return float(match.group(1))


def _parse_number(text: str) -> float | int | None:
	cleaned = text.replace("\xa0", " ").strip()
	if not cleaned:
		return None
	if re.fullmatch(r"[0-9]+", cleaned):
		return int(cleaned)
	if re.fullmatch(r"[0-9]+\.[0-9]+", cleaned):
		return float(cleaned)
	return None


def _parse_fraction(text: str) -> dict[str, float | int | None]:
	"""Parse values such as ``1 of 6 — 17%``.

	The function is tolerant of missing values and returns ``None`` for
	fields that are not present in the display string.
	"""

	normalized = text.replace("\xa0", " ").replace("—", "-").strip()
	if not normalized:
		return {"value": None, "total": None, "percent": None}

	if " of " not in normalized:
		return {"value": _parse_number(normalized), "total": None, "percent": _parse_percent(normalized)}

	left, right = normalized.split(" of ", 1)
	total_part = right
	percent = _parse_percent(normalized)
	if "-" in total_part:
		total_part = total_part.split("-", 1)[0].strip()

	return {
		"value": _parse_number(left),
		"total": _parse_number(total_part),
		"percent": percent,
	}


def _flatten_player_stats(df: pd.DataFrame) -> pd.DataFrame:
	df = df.reset_index()
	flattened_columns: list[str] = []
	for column in df.columns:
		if isinstance(column, tuple):
			parts = [str(part) for part in column if part not in ("", None)]
			flattened_columns.append("_".join(parts))
		else:
			flattened_columns.append(str(column))
	df.columns = flattened_columns
	return df


def _parse_team_stats(html: str) -> tuple[str, str, dict[str, dict[str, Any]]]:
	soup = BeautifulSoup(html, "html.parser")
	team_stats = soup.find("div", id="team_stats")
	if team_stats is None:
		raise ValueError("Could not find the team stats block in the FBref match report.")

	rows = team_stats.find_all("tr")
	if not rows:
		raise ValueError("Team stats block is empty.")

	team_names = [cell.get_text(" ", strip=True) for cell in rows[0].find_all("th")]
	if len(team_names) != 2:
		raise ValueError("Could not determine home and away team names from the match report.")

	stats: dict[str, dict[str, Any]] = {team_names[0]: {}, team_names[1]: {}}

	for label_row, value_row in zip(rows[1::2], rows[2::2], strict=False):
		label = label_row.get_text(" ", strip=True)
		cells = value_row.find_all("td")
		if len(cells) != 2:
			continue
		left = cells[0].get_text(" ", strip=True)
		right = cells[1].get_text(" ", strip=True)

		key = _slugify(label)
		if key == "possession":
			stats[team_names[0]]["possession_pct"] = _parse_percent(left)
			stats[team_names[1]]["possession_pct"] = _parse_percent(right)
			continue

		if key == "shots_on_target":
			left_parsed = _parse_fraction(left)
			right_parsed = _parse_fraction(right)
			stats[team_names[0]]["shots_on_target"] = left_parsed["value"]
			stats[team_names[0]]["shots_total_from_team_stats"] = left_parsed["total"]
			stats[team_names[1]]["shots_on_target"] = right_parsed["value"]
			stats[team_names[1]]["shots_total_from_team_stats"] = right_parsed["total"]
			stats[team_names[0]]["shots_on_target_pct"] = left_parsed["percent"]
			stats[team_names[1]]["shots_on_target_pct"] = right_parsed["percent"]
			continue

		if key == "saves":
			left_parsed = _parse_fraction(left)
			right_parsed = _parse_fraction(right)
			stats[team_names[0]]["saves"] = left_parsed["value"]
			stats[team_names[1]]["saves"] = right_parsed["value"]
			stats[team_names[0]]["saves_from_team_stats"] = left_parsed["total"]
			stats[team_names[1]]["saves_from_team_stats"] = right_parsed["total"]
			stats[team_names[0]]["save_pct"] = left_parsed["percent"]
			stats[team_names[1]]["save_pct"] = right_parsed["percent"]
			continue

		stats[team_names[0]][key] = left or None
		stats[team_names[1]][key] = right or None

	return team_names[0], team_names[1], stats


def _summarize_player_stats(player_stats: pd.DataFrame) -> pd.DataFrame:
	flattened = _flatten_player_stats(player_stats)
	for column in ["Performance_Gls", "Performance_Sh", "Performance_SoT", "Performance_CrdY", "Performance_CrdR"]:
		if column not in flattened.columns:
			flattened[column] = 0

	team_totals = (
		flattened.groupby("team", dropna=False)[
			["Performance_Gls", "Performance_Sh", "Performance_SoT", "Performance_CrdY", "Performance_CrdR"]
		]
		.sum(min_count=1)
		.rename(
			columns={
				"Performance_Gls": "goals",
				"Performance_Sh": "shots",
				"Performance_SoT": "shots_on_target_from_players",
				"Performance_CrdY": "yellow_cards",
				"Performance_CrdR": "red_cards",
			}
		)
		.reset_index()
	)
	return team_totals


def scrape_match_statistics(
	leagues: str | list[str],
	seasons: str | int | list[str | int],
	*,
	match_ids: list[str] | str | None = None,
	test_mode: bool = False,
	test_limit: int = TEST_LIMIT,
	force_cache: bool = False,
) -> pd.DataFrame:
	"""Scrape detailed match statistics for one or more leagues/seasons.

	Returns one row per match with home/away-prefixed columns.
	"""

	fbref = sd.FBref(leagues, seasons)
	schedule = fbref.read_schedule(force_cache=force_cache).reset_index()
	schedule = schedule[schedule["match_report"].notna()]

	if match_ids is not None:
		requested_ids = [match_ids] if isinstance(match_ids, str) else list(match_ids)
		schedule = schedule[schedule["game_id"].isin(requested_ids)]

	if test_mode:
		schedule = schedule.head(test_limit)

	records: list[dict[str, Any]] = []

	for _, match in schedule.iterrows():
		match_url = "https://fbref.com" + match["match_report"]
		html = fbref.get(match_url).read().decode("utf-8", errors="ignore")
		home_team, away_team, team_stats = _parse_team_stats(html)

		player_stats = fbref.read_player_match_stats(
			stat_type="summary",
			match_id=match["game_id"],
			force_cache=force_cache,
		)
		player_totals = _summarize_player_stats(player_stats)

		row: dict[str, Any] = {
			"league": match["league"],
			"season": match["season"],
			"game": match["game"],
			"game_id": match["game_id"],
			"date": match["date"],
			"home_team": match["home_team"],
			"away_team": match["away_team"],
			"score": match["score"],
			"match_report": match["match_report"],
		}

		team_to_prefix = {
			home_team: "home",
			away_team: "away",
		}

		for team_name, prefix in team_to_prefix.items():
			stats = team_stats.get(team_name, {})
			totals = player_totals[player_totals["team"] == team_name]
			totals_record = totals.iloc[0].to_dict() if not totals.empty else {}

			row[f"{prefix}_team"] = team_name
			row[f"{prefix}_possession_pct"] = stats.get("possession_pct")
			row[f"{prefix}_shots_on_target"] = stats.get("shots_on_target")
			row[f"{prefix}_shots_total_from_team_stats"] = stats.get("shots_total_from_team_stats")
			row[f"{prefix}_shots"] = totals_record.get("shots")
			row[f"{prefix}_goals"] = totals_record.get("goals")
			row[f"{prefix}_yellow_cards"] = totals_record.get("yellow_cards")
			row[f"{prefix}_red_cards"] = totals_record.get("red_cards")
			row[f"{prefix}_saves"] = stats.get("saves")
			row[f"{prefix}_save_pct"] = stats.get("save_pct")

		records.append(row)

	output = pd.DataFrame.from_records(records)
	if not output.empty:
		output = output.sort_values(["date", "game_id"]).reset_index(drop=True)
	return output


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Scrape detailed match statistics from FBref.")
	parser.add_argument(
		"--leagues",
		nargs="+",
		help="FBref league ids. Defaults to the Big 5 combined reader.",
	)
	parser.add_argument(
		"--seasons",
		nargs="+",
		type=int,
		help="Season start years. Defaults to 2014 through the latest completed season.",
	)
	parser.add_argument(
		"--match-id",
		dest="match_ids",
		action="append",
		help="Optional game_id filter. Repeat the flag to fetch multiple matches.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		help="Optional CSV path. If omitted, the result is only printed.",
	)
	parser.add_argument(
		"--test-mode",
		action="store_true",
		help="Only scrape a small handful of matches.",
	)
	parser.add_argument(
		"--test-limit",
		type=int,
		default=TEST_LIMIT,
		help="Maximum number of matches to scrape in test mode.",
	)
	parser.add_argument(
		"--force-cache",
		action="store_true",
		help="Force cached FBref pages when available.",
	)
	parser.add_argument(
		"league",
		nargs="?",
		help="Optional legacy positional FBref league id override.",
	)
	parser.add_argument(
		"season",
		nargs="?",
		help="Optional legacy positional season override.",
	)
	return parser


def main() -> None:
	parser = _build_parser()
	args = parser.parse_args()

	if args.league and args.season:
		leagues: str | list[str] = args.league
		seasons: str | int | list[str | int] = args.season
	else:
		leagues = args.leagues if args.leagues is not None else DEFAULT_LEAGUES
		seasons = args.seasons if args.seasons is not None else DEFAULT_SEASONS

	df = scrape_match_statistics(
		leagues,
		seasons,
		match_ids=args.match_ids,
		test_mode=args.test_mode,
		test_limit=args.test_limit,
		force_cache=args.force_cache,
	)

	output_path = args.output
	if output_path is None:
		output_path = TEST_OUTPUT if args.test_mode else DEFAULT_OUTPUT

	output_path.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(output_path, index=False)
	print(f"Saved {len(df)} matches to {output_path}")


if __name__ == "__main__":
	main()
