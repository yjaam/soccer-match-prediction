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


def fetch_league_season_matches(league_name: str, seasons):
    """
    Fetch match data for explicit season start years.
    Returns list of rows in unified format.
    """
    understat = UnderstatClient()
    league = understat.league(league=league_name)

    rows = []
    for s in seasons:
        season_str = str(s)
        try:
            matches = league.get_match_data(season=season_str)
        except Exception as e:
            print(f"[WARN] {league_name} season {season_str} skipped: {e}")
            continue

        if not matches:
            print(f"[INFO] {league_name} season {season_str}: no matches returned.")
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
    """
    Default:
      - current season start year = current calendar year
      - previous season start year = current calendar year - 1
      - leagues = Bundesliga + 2. Bundesliga
    """
    current_year = datetime.today().year
    seasons = [current_year - 1, current_year]  # previous + current
    leagues = ["Bundesliga", "2. Bundesliga"]

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
        keep="first"
    )

    # Sort for stable downstream form-calculation
    df = df.sort_values(
        by=["date", "league", "season", "matchday", "home_team"],
        na_position="last"
    ).reset_index(drop=True)

    return df


def add_canonical_team_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses TeamNameResolver to produce stable mapped team names.
    For Understat-internal consistency this still helps normalize aliases.
    """
    out = df.copy()

    # Build resolver from all observed names in this dataset
    target_names = sorted(set(out["home_team"].dropna().astype(str)).union(
                          set(out["away_team"].dropna().astype(str))))
    resolver = TeamNameResolver(target_names=target_names, min_fuzzy_score=82)

    out["home_team_mapped"] = out["home_team"].apply(lambda x: resolver.resolve(x) if pd.notna(x) else x)
    out["away_team_mapped"] = out["away_team"].apply(lambda x: resolver.resolve(x) if pd.notna(x) else x)

    # Final fallback: if resolver returns None, keep original
    out["home_team_mapped"] = out["home_team_mapped"].fillna(out["home_team"])
    out["away_team_mapped"] = out["away_team_mapped"].fillna(out["away_team"])

    return out


def main():
    df = fetch_bundesliga_xg_current_and_previous()
    df = add_canonical_team_names(df)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, "data", "bundesliga_xg_clean.csv")
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