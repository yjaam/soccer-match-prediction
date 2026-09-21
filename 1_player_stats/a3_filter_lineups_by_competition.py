"""
Filter lineups.csv to keep only the top 5 European competitions.

Ensures Home Team and Away Team are three-letter codes before building
the game_id, so downstream joins are reliable.
"""

import pandas as pd
from pathlib import Path
import sys
import unicodedata
import warnings

warnings.filterwarnings("ignore")

# --- paths ----------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.team_name_mapping_FINAL import resolve_team_name, resolve_team_name_fuzzy

DATA_DIR = PROJECT_DIR / "data"
LINEUPS_FILE = DATA_DIR / "lineups.csv"
LINEUPS_BACKUP_FILE = DATA_DIR / "lineups_backup_before_filter.csv"

# --- target competitions --------------------------------------------------
TARGET_COMPETITIONS = [
    "serie-a",
    "premier-league",
    "bundesliga",
    "laliga",
    "ligue-1",
]

CODE_REGEX = r"^[A-Z]{3}$"


def is_code(x) -> bool:
    """True if x is a valid-looking 3-letter uppercase code."""
    if pd.isna(x):
        return False
    s = str(x).strip()
    return len(s) == 3 and s.isalpha() and s.isupper()


def try_resolve_to_code(x, resolver_cache: dict):
    """
    Return a 3-letter code for x, or None if we can't determine one.

    Strategy:
      1. If already a code, keep it.
      2. Otherwise try resolve_team_name (exact dict lookup).
      3. Then resolve_team_name_fuzzy (handles suffix/prefix variants).
      4. Otherwise return None.
    """
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    if is_code(s):
        return s

    if s in resolver_cache:
        return resolver_cache[s]

    code = resolve_team_name(s) or resolve_team_name_fuzzy(s)
    resolver_cache[s] = code
    return code


def main():
    print("=" * 60)
    print("Filtering Lineups to Top 5 European Competitions")
    print("=" * 60)

    if not LINEUPS_FILE.exists():
        print(f"Error: {LINEUPS_FILE} not found.")
        return

    print(f"\nLoading {LINEUPS_FILE}...")
    lineups = pd.read_csv(LINEUPS_FILE, low_memory=False)
    original_count = len(lineups)
    print(f"Original rows: {original_count}")

    print(f"\nCompetitions before filtering:")
    print(lineups["Competition"].value_counts())

    # Backup
    print(f"\nCreating backup...")
    lineups.to_csv(LINEUPS_BACKUP_FILE, index=False)
    print(f"Backup saved to {LINEUPS_BACKUP_FILE}")

    # --- Step 1: filter to target competitions ----------------------------
    print(f"\nFiltering to target competitions: {', '.join(TARGET_COMPETITIONS)}")
    df = lineups[lineups["Competition"].isin(TARGET_COMPETITIONS)].copy()
    print(f"Rows after competition filter: {len(df)}")

    if df.empty:
        print("No rows left after competition filter. Aborting.")
        return

    # --- Step 2: normalize team columns to codes --------------------------
    print(f"\nResolving team names to three-letter codes...")
    cache = {}

    def _resolve(series):
        return series.map(lambda x: try_resolve_to_code(x, cache))

    df["Home Team Code"] = _resolve(df["Home Team"])
    df["Away Team Code"] = _resolve(df["Away Team"])

    # Diagnostics: what did we fail to resolve?
    unresolved_home = df.loc[df["Home Team Code"].isna(), "Home Team"].dropna().unique()
    unresolved_away = df.loc[df["Away Team Code"].isna(), "Away Team"].dropna().unique()
    all_unresolved = sorted(set(map(str, unresolved_home)) | set(map(str, unresolved_away)))

    if all_unresolved:
        print(f"\n  ⚠ Could not resolve {len(all_unresolved)} unique team names:")
        for name in all_unresolved[:25]:
            print(f"      {name!r}")
        if len(all_unresolved) > 25:
            print(f"      ... and {len(all_unresolved) - 25} more")
        print(f"\n  These rows will be dropped. To keep them, add the missing")
        print(f"  variants to src/team_name_mapping_FINAL.py and re-run this")
        print(f"  script plus the upstream a2_extract_historic_lineups.py.")

    # --- Step 3: keep only rows where BOTH sides resolved to a code -------
    before = len(df)
    df = df[df["Home Team Code"].notna() & df["Away Team Code"].notna()].copy()
    dropped = before - len(df)
    print(f"\nRows dropped (unresolvable team names): {dropped}")
    print(f"Rows kept: {len(df)}")

    # Promote the resolved codes back into the canonical columns
    df["Home Team"] = df["Home Team Code"]
    df["Away Team"] = df["Away Team Code"]
    df = df.drop(columns=["Home Team Code", "Away Team Code"])

    # --- Step 4: build game_id --------------------------------------------
    print(f"\nAdding game_id column...")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Drop rows with unparseable dates — they'd produce nan-prefixed IDs
    bad_dates = df["Date"].isna().sum()
    if bad_dates > 0:
        print(f"  Dropping {bad_dates} rows with unparseable Date")
        df = df[df["Date"].notna()].copy()

    df["game_id"] = (
        df["Date"].dt.strftime("%Y%m%d")
        + "_" + df["Home Team"]
        + "_" + df["Away Team"]
    )

    # Move game_id to the first column
    cols = ["game_id"] + [c for c in df.columns if c != "game_id"]
    df = df[cols]

    # --- Step 5: dedup on game_id -----------------------------------------
    before = len(df)
    df = df.drop_duplicates(subset=["game_id"], keep="first").reset_index(drop=True)
    print(f"\nRemoved {before - len(df)} duplicate game_id rows")
    print(f"Rows after dedup: {len(df)}")

    # --- Step 6: sort and save --------------------------------------------
    df = df.sort_values("Date").reset_index(drop=True)

    print(f"\nFinal competitions:")
    print(df["Competition"].value_counts())

    print(f"\nSaving filtered lineups to {LINEUPS_FILE}...")
    df.to_csv(LINEUPS_FILE, index=False)
    print(f"Successfully saved {len(df)} rows")

    print("\n" + "=" * 60)
    print("Filtering completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()