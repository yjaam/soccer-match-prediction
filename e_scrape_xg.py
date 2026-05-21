#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
from datetime import datetime
from understatapi import UnderstatClient

# Helper in same folder
from team_name_resolver import TeamNameResolver


def parse_matchday(m):
    # Different wrappers/versions may expose different keys
    for key in ("round", "gameweek", "week"):
        if key in m and m[key] is not None:
            return pd.to_numeric(m[key], errors="coerce")
    return pd.NA


def load_main_team_names() -> list[str]:
    """Load the canonical team vocabulary from the main game CSV."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_path = os.path.join(script_dir, "data", "lineups_values_ratings_games.csv")
    main_df = pd.read_csv(main_path)
    home_names = main_df["Home Team"].dropna().astype(str)
    away_names = main_df["Away Team"].dropna().astype(str)
    return sorted(set(home_names).union(set(away_names)))


def fetch_league_season_matches(league_name: str, seasons):
    """
    Fetch match data for explicit season start years.
    Returns list of rows in unified format.
    """
    understat = UnderstatClient()
    # Candidates for common display names mapped to potential Understat identifiers.
    LEAGUE_CANDIDATES = {
        "Premier League": ["Premier League", "EPL", "premier-league", "england"],
        "La Liga": ["La Liga", "LaLiga", "La_Liga", "laliga", "spain"],
        "Serie A": [
            "Serie A",
            "SerieA",
            "Serie_A",
            "serie-a",
            "serie_a",
            "italy",
            "italia",
            "Serie A TIM",
            "Serie A TIM",
            "SERIE A",
        ],
        "Ligue 1": [
            "Ligue 1",
            "Ligue1",
            "Ligue_1",
            "ligue-1",
            "ligue_1",
            "france",
            "Ligue 1 Uber Eats",
            "Ligue1UberEats",
            "Ligue1_UberEats",
        ],
        "Bundesliga": ["Bundesliga", "bundesliga"],
    }

    candidates = LEAGUE_CANDIDATES.get(league_name, [league_name])

    rows = []
    for s in seasons:
        season_str = str(s)
        matches = None
        used_candidate = None

        for cand in candidates:
            try:
                league = understat.league(league=cand)
                matches = league.get_match_data(season=season_str)
                used_candidate = cand
            except Exception:
                matches = None

            # if we got results (even empty list), stop trying other candidates
            if matches is not None:
                break

        if matches is None:
            print(f"[WARN] {league_name} season {season_str} skipped: no valid league identifier found among candidates {candidates}")
            continue

        if not matches:
            print(f"[INFO] {league_name} season {season_str}: no matches returned (tried '{used_candidate}').")
            continue

        for m in matches:
            date_raw = m.get("datetime")
            date = pd.to_datetime(date_raw, errors="coerce")

            home_team = (m.get("h") or {}).get("title")
            away_team = (m.get("a") or {}).get("title")

            xg = m.get("xG") or {}
            home_xg = pd.to_numeric(xg.get("h"), errors="coerce")
            away_xg = pd.to_numeric(xg.get("a"), errors="coerce")

            rows.append({
                "league": league_name,
                "season": s,  # season start year
                "matchday": parse_matchday(m),
                "date": date.date() if pd.notna(date) else pd.NaT,
                "home_team": home_team,
                "away_team": away_team,
                "home_xg": home_xg,
                "away_xg": away_xg,
            })

    return rows


def fetch_bundesliga_xg_current_and_previous():
    """Fetch xG for the Big-5 European leagues (previous + current season).

    This generalises the previous Bundesliga-only helper to collect data
    for the five major leagues: Premier League, La Liga, Serie A, Ligue 1
    and Bundesliga.
    """
    current_year = datetime.today().year
    # fetch all seasons from 2012 up to the current season start year
    seasons = list(range(2012, current_year + 1))

    # Understat league names used by the API client. These have worked with
    # the prior code; adjust if your Understat client expects different strings.
    leagues = [
        "Premier League",
        "La Liga",
        "Serie A",
        "Ligue 1",
        "Bundesliga",
    ]

    all_rows = []
    for lg in leagues:
        print(f"[INFO] Fetching {lg} for seasons {seasons} ...")
        all_rows.extend(fetch_league_season_matches(lg, seasons))

    if not all_rows:
        raise RuntimeError("No match data found from Understat for requested leagues/seasons.")

    df = pd.DataFrame(all_rows)

    # Clean table
    df = df.dropna(subset=["date", "home_team", "away_team"]).copy()
    df["matchday"] = pd.to_numeric(df["matchday"], errors="coerce")

    # Remove exact duplicates if any
    df = df.drop_duplicates(
        subset=["league", "season", "date", "home_team", "away_team"],
        keep="first",
    )

    # Sort for stable downstream form-calculation
    df = df.sort_values(
        by=["date", "league", "season", "matchday", "home_team"],
        na_position="last",
    ).reset_index(drop=True)

    return df


def add_canonical_team_names(df: pd.DataFrame, target_names) -> pd.DataFrame:
    """
    Uses TeamNameResolver to produce stable mapped team names.
    For Understat-internal consistency this still helps normalize aliases.
    """
    out = df.copy()

    resolver = TeamNameResolver(target_names=target_names, min_fuzzy_score=82)

    out["home_team_mapped"] = out["home_team"].apply(lambda x: resolver.resolve(x) if pd.notna(x) else x)
    out["away_team_mapped"] = out["away_team"].apply(lambda x: resolver.resolve(x) if pd.notna(x) else x)

    # Final fallback: if resolver returns None, keep original
    out["home_team_mapped"] = out["home_team_mapped"].fillna(out["home_team"])
    out["away_team_mapped"] = out["away_team_mapped"].fillna(out["away_team"])

    return out


def main():
    df = fetch_bundesliga_xg_current_and_previous()
    df = add_canonical_team_names(df, load_main_team_names())

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # write Big-5 xG output
    out_path = os.path.join(script_dir, "data", "big5_xg_clean.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved {len(df)} rows to: {out_path}")

    # Summary
    if not df.empty:
        summary = (
            df.groupby(["league", "season"], dropna=False)
              .size()
              .reset_index(name="matches")
              .sort_values(["league", "season"])
        )
        print("\n[INFO] Rows per league/season:")
        print(summary.to_string(index=False))

    print("\nPreview:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()