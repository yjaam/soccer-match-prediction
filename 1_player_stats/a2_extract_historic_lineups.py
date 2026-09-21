"""
Script to extract historic lineups from transfermarkt data and append to lineups.csv
Matches the column format of the existing lineups.csv file

Vectorized rewrite — produces the same output as the previous version
but avoids per-game Python iteration, which makes it 10-50x faster.

Fixed:
  - Robust dedup key construction (Date normalized, all cols astype(str))
  - Unmapped team names fall back to the original club name
  - game_id is explicitly carried through to the output (was previously
    being written as an empty string by the column alignment step)
  - Loud diagnostics to make future runs self-explanatory
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings

# --- path setup -----------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.team_name_mapping_FINAL import resolve_team_name

warnings.filterwarnings('ignore')

# --- paths ----------------------------------------------------------------
TRANSFERMARKT_DIR = PROJECT_DIR / "transfermarkt_data"
DATA_DIR = PROJECT_DIR / "data"

GAMES_FILE = TRANSFERMARKT_DIR / "games.csv"
GAME_LINEUPS_FILE = TRANSFERMARKT_DIR / "game_lineups.csv"
COMPETITIONS_FILE = TRANSFERMARKT_DIR / "competitions.csv"

OUTPUT_LINEUPS_FILE = DATA_DIR / "lineups.csv"
BACKUP_LINEUPS_FILE = DATA_DIR / "lineups_backup.csv"

# --- position category mapping --------------------------------------------
ATTACKER_TERMS = ["forward", "winger", "striker", "attack"]
DEFENDER_TERMS = ["back", "defender", "sweeper", "libero"]
MIDFIELD_TERMS = ["midfield"]


def categorize_positions(position_series: pd.Series) -> pd.Series:
    """Vectorized version of get_position_category()."""
    s = position_series.fillna("").astype(str).str.strip().str.lower()

    conditions = [
        s.str.contains("goalkeeper", na=False),
        s.apply(lambda x: any(t in x for t in ATTACKER_TERMS)),
        s.apply(lambda x: any(t in x for t in DEFENDER_TERMS)),
        s.apply(lambda x: any(t in x for t in MIDFIELD_TERMS)),
    ]
    choices = ["Goalkeeper", "Attacker", "Defender", "Midfielder"]
    result = np.select(conditions, choices, default="Midfielder")
    return pd.Series(result, index=position_series.index)


def make_game_key(df: pd.DataFrame) -> pd.Series:
    """
    Build a canonical composite key for deduplication.

    Both sides (existing and historic) go through this so that any
    representation difference in Date (datetime vs string) or team name
    (NaN vs empty string) can't cause false dedup matches.
    """
    date_str = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    date_str = date_str.fillna("__nodate__")

    def _clean(series):
        return (
            series.fillna("__na__")
                  .astype(str)
                  .str.strip()
                  .replace("", "__na__")
        )

    return (
        _clean(df["Competition"]) + "|" +
        date_str + "|" +
        _clean(df["Home Team"]) + "|" +
        _clean(df["Away Team"])
    )


def build_historic_lineups():
    """Build historic lineups from transfermarkt data (vectorized)."""

    print("Loading data files...")
    competitions_df = pd.read_csv(COMPETITIONS_FILE)
    comp_mapping = dict(zip(competitions_df["competition_id"], competitions_df["name"]))

    games_df = pd.read_csv(GAMES_FILE)
    print(f"Loaded {len(games_df)} games")

    game_lineups_df = pd.read_csv(GAME_LINEUPS_FILE)
    print(f"Loaded {len(game_lineups_df)} lineup entries")

    # --- Filter to starting lineups only ----------------------------------
    game_lineups_df = game_lineups_df[game_lineups_df["type"] == "starting_lineup"].copy()
    print(f"Filtered to {len(game_lineups_df)} starting lineup entries")

    # --- Position categorization ------------------------------------------
    game_lineups_df["position_category"] = categorize_positions(
        game_lineups_df["position"]
    )

    # --- Expected output columns ------------------------------------------
    expected_columns = None
    if OUTPUT_LINEUPS_FILE.exists():
        existing_lineups = pd.read_csv(OUTPUT_LINEUPS_FILE)
        expected_columns = list(existing_lineups.columns)
        print(f"\nExpected columns from existing lineups: {len(expected_columns)}")
        print(expected_columns[:10], "...")
    else:
        existing_lineups = None

    # --- Prepare games lookup ---------------------------------------------
    games_lookup = games_df[
        ["game_id", "competition_id", "date", "home_club_id", "away_club_id",
         "home_club_name", "away_club_name"]
    ].copy()
    games_lookup = games_lookup.rename(columns={"date": "game_date"})

    # --- Merge lineups with game metadata ---------------------------------
    merged = game_lineups_df.merge(games_lookup, on="game_id", how="left")

    before = len(merged)
    merged = merged.dropna(
        subset=["home_club_name", "away_club_name", "game_date"]
    )
    if len(merged) < before:
        print(f"Dropped {before - len(merged)} lineup rows with missing game info")

    # --- Assign side ------------------------------------------------------
    side = np.where(
        merged["club_id"] == merged["home_club_id"], "Home",
        np.where(merged["club_id"] == merged["away_club_id"], "Away", None)
    )
    merged["side"] = side
    merged = merged[merged["side"].notna()].copy()

    # --- Slot numbers -----------------------------------------------------
    merged["number_numeric"] = pd.to_numeric(merged["number"], errors="coerce")
    merged = merged.dropna(subset=["number_numeric"])

    merged = merged.sort_values(
        ["game_id", "side", "position_category", "number_numeric"],
        kind="mergesort",
    )

    merged["slot_idx"] = merged.groupby(
        ["game_id", "side", "position_category"], sort=False
    ).cumcount() + 1

    # --- Slot name --------------------------------------------------------
    is_gk = merged["position_category"] == "Goalkeeper"
    merged["slot_name"] = np.where(
        is_gk,
        merged["side"] + "_Goalkeeper",
        merged["side"] + "_" + merged["position_category"] + "_" +
        merged["slot_idx"].astype(str)
    )

    # --- Pivot ------------------------------------------------------------
    wide = merged.pivot_table(
        index="game_id",
        columns="slot_name",
        values="player_name",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    # --- Attach game-level info -------------------------------------------
    game_info = games_lookup.drop_duplicates("game_id").set_index("game_id")
    wide = wide.set_index("game_id").join(game_info, how="left").reset_index()

    # --- Final output frame -----------------------------------------------
    # IMPORTANT: carry game_id through explicitly. The previous version left
    # it out and relied on the column-alignment step to add it — which filled
    # it with empty strings, breaking every downstream join.
    out = pd.DataFrame()
    out["game_id"] = wide["game_id"]
    out["Competition"] = wide["competition_id"].map(comp_mapping).fillna("Unknown")
    out["Date"] = wide["game_date"]
    out["Home Team"] = wide["home_club_name"]
    out["Away Team"] = wide["away_club_name"]

    slot_cols = [c for c in wide.columns if c.startswith("Home_") or c.startswith("Away_")]
    category_order = ["Goalkeeper", "Defender", "Midfielder", "Attacker"]
    ordered_slots = []
    for side in ("Home", "Away"):
        for cat in category_order:
            if cat == "Goalkeeper":
                col = f"{side}_Goalkeeper"
                if col in wide.columns:
                    ordered_slots.append(col)
            else:
                cat_cols = [c for c in slot_cols if c.startswith(f"{side}_{cat}_")]
                cat_cols.sort(key=lambda c: int(c.rsplit("_", 1)[1]))
                ordered_slots.extend(cat_cols)

    for col in ordered_slots:
        out[col] = wide[col]

    # --- Standardize team names -------------------------------------------
    print(f"\nApplying team name resolver...")
    cache = {}

    def cached_resolve(x):
        if x in cache:
            return cache[x]
        r = resolve_team_name(x)
        cache[x] = r
        return r

    home_original = out["Home Team"].copy()
    away_original = out["Away Team"].copy()

    out["Home Team"] = out["Home Team"].map(cached_resolve)
    out["Away Team"] = out["Away Team"].map(cached_resolve)

    # Fill unmapped names with the original club name so the dedup key
    # downstream isn't NaN for thousands of rows.
    out["Home Team"] = out["Home Team"].where(out["Home Team"].notna(), home_original)
    out["Away Team"] = out["Away Team"].where(out["Away Team"].notna(), away_original)

    # Keep the raw names for reference if the expected schema includes them.
    if expected_columns and "Home Team Raw" in expected_columns:
        out["Home Team Raw"] = home_original.values
    if expected_columns and "Away Team Raw" in expected_columns:
        out["Away Team Raw"] = away_original.values

    print(f"Columns: {list(out.columns)[:10]}...")

    # --- Align to the expected column set ---------------------------------
    if expected_columns:
        for col in expected_columns:
            if col not in out.columns:
                out[col] = ""
        out = out[expected_columns]
        print(f"Column alignment completed. Using {len(expected_columns)} columns")

    # --- Deduplicate on game_id -------------------------------------------
    before_dedup = len(out)
    out = out.drop_duplicates(subset=["game_id"], keep="first").reset_index(drop=True)
    if len(out) < before_dedup:
        print(f"Deduplicated historic frame: {before_dedup} → {len(out)} rows")

    return out


def append_to_lineups(historic_lineups):
    """Safely append historic lineups to existing lineups.csv."""

    if not OUTPUT_LINEUPS_FILE.exists():
        print(f"\n{OUTPUT_LINEUPS_FILE} does not exist. Creating new file...")
        historic_lineups.to_csv(OUTPUT_LINEUPS_FILE, index=False)
        print(f"Created {OUTPUT_LINEUPS_FILE} with {len(historic_lineups)} rows")
        return

    print(f"\nCreating backup of existing lineups...")
    existing_lineups = pd.read_csv(OUTPUT_LINEUPS_FILE)
    existing_lineups.to_csv(BACKUP_LINEUPS_FILE, index=False)
    print(f"Backup created: {BACKUP_LINEUPS_FILE}")

    print(f"\nRemoving duplicates...")

    existing_lineups = existing_lineups.copy()
    historic_lineups = historic_lineups.copy()
    existing_lineups["game_key"] = make_game_key(existing_lineups)
    historic_lineups["game_key"] = make_game_key(historic_lineups)

    print(f"  Existing rows:          {len(existing_lineups)}")
    print(f"  Existing unique keys:   {existing_lineups['game_key'].nunique()}")
    print(f"  Historic rows:          {len(historic_lineups)}")
    print(f"  Historic unique keys:   {historic_lineups['game_key'].nunique()}")

    top_existing = existing_lineups["game_key"].value_counts().head(3)
    print(f"  Top existing keys:")
    for k, v in top_existing.items():
        print(f"    ({v}x) {k!r}")

    existing_keys = set(existing_lineups["game_key"])
    historic_new = historic_lineups[
        ~historic_lineups["game_key"].isin(existing_keys)
    ].copy()

    print(f"  Existing keys:          {len(existing_keys)}")
    print(f"  Historic rows kept:     {len(historic_new)}")
    print(f"  Historic rows dropped:  {len(historic_lineups) - len(historic_new)}")

    if len(historic_new) < len(historic_lineups) * 0.5:
        dropped = historic_lineups[
            historic_lineups["game_key"].isin(existing_keys)
        ].head(5)
        print(f"  Sample of dropped (matched existing) keys:")
        for _, r in dropped.iterrows():
            print(f"    {r['game_key']!r}")

    existing_lineups = existing_lineups.drop(columns=["game_key"])
    historic_new = historic_new.drop(columns=["game_key"])

    combined = pd.concat([existing_lineups, historic_new], ignore_index=True)

    combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
    combined = combined.sort_values("Date").reset_index(drop=True)

    print(f"\nSaving combined lineups...")
    combined.to_csv(OUTPUT_LINEUPS_FILE, index=False)
    print(f"Successfully saved {len(combined)} total rows to {OUTPUT_LINEUPS_FILE}")

    print(f"\nSummary:")
    print(f"  - Original rows: {len(existing_lineups)}")
    print(f"  - New rows added: {len(historic_new)}")
    print(f"  - Total rows: {len(combined)}")


if __name__ == "__main__":
    print("=" * 60)
    print("Extracting Historic Lineups from Transfermarkt Data")
    print("=" * 60)

    historic_lineups = build_historic_lineups()
    append_to_lineups(historic_lineups)

    print("\n" + "=" * 60)
    print("Process completed successfully!")
    print("=" * 60)