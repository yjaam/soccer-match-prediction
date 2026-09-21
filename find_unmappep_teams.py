#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Console diagnostic: find every team name in the lineup files that the
current src/team_name_mapping_FINAL.py cannot resolve to a 3-letter code."""

import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from src.team_name_mapping_FINAL import (
    soccer_teams,
    resolve_team_name,
    resolve_team_name_fuzzy,
)

DATA_DIR = SCRIPT_DIR / "data"

FILES = [
    DATA_DIR / "lineups.csv",
    DATA_DIR / "lineups_backup_before_filter.csv",
]


def is_code(x):
    if pd.isna(x):
        return False
    s = str(x).strip()
    return len(s) == 3 and s.isalpha() and s.isupper()


def normalize(s):
    if pd.isna(s):
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    s = " ".join(s.split())
    return s


# Build normalized reverse lookup for suggestions
normalized_lookup = {}
for name, code in soccer_teams.items():
    key = normalize(name)
    if key and key not in normalized_lookup:
        normalized_lookup[key] = (name, code)


def suggest(raw):
    key = normalize(raw)
    if not key:
        return None, None

    if key in normalized_lookup:
        basis, code = normalized_lookup[key]
        return code, f"normalized match: {basis!r}"

    candidates = []
    for nk, (basis, code) in normalized_lookup.items():
        if not nk:
            continue
        if key in nk or nk in key:
            candidates.append((len(min(key, nk, key=len)), basis, code))
    if candidates:
        candidates.sort(reverse=True)
        _, basis, code = candidates[0]
        return code, f"substring match: {basis!r}"

    first_word = key.split()[0] if key.split() else ""
    if len(first_word) >= 4:
        for nk, (basis, code) in normalized_lookup.items():
            if nk.startswith(first_word):
                return code, f"first-word match: {basis!r}"

    return None, None


def scan_file(path: Path):
    if not path.exists():
        return Counter(), {}
    df = pd.read_csv(path, low_memory=False)
    if "Home Team" not in df.columns or "Away Team" not in df.columns:
        return Counter(), {}

    unmapped = Counter()
    comp_for_name = {}

    def check(team, competition):
        if pd.isna(team):
            return
        s = str(team).strip()
        if not s or is_code(s):
            return
        code = resolve_team_name(s) or resolve_team_name_fuzzy(s)
        if code:
            return
        unmapped[s] += 1
        comp_for_name.setdefault(s, set()).add(str(competition))

    for _, row in df.iterrows():
        comp = row.get("Competition", "unknown")
        check(row["Home Team"], comp)
        check(row["Away Team"], comp)

    return unmapped, comp_for_name


def main():
    print("=" * 74)
    print("SCANNING FOR UNMAPPED TEAM NAMES")
    print("=" * 74)

    grand_total = Counter()
    grand_comp = {}

    for path in FILES:
        print(f"\n--- {path.name} ---")
        if not path.exists():
            print("  (file not found)")
            continue
        unmapped, comp_for_name = scan_file(path)
        print(f"  Unique unmapped names: {len(unmapped)}")
        print(f"  Total unmapped rows:   {sum(unmapped.values())}")
        grand_total.update(unmapped)
        for name, comps in comp_for_name.items():
            grand_comp.setdefault(name, set()).update(comps)

    if not grand_total:
        print("\nAll team names are already mapped.")
        return

    print("\n" + "=" * 74)
    print(f"CONSOLIDATED: {len(grand_total)} unique unmapped names, "
          f"{sum(grand_total.values())} total occurrences")
    print("=" * 74)

    rows = []
    for name, count in sorted(grand_total.items(), key=lambda kv: -kv[1]):
        suggestion, basis = suggest(name)
        comps = ", ".join(sorted(grand_comp.get(name, set())))
        rows.append((count, name, suggestion, basis, comps))

    print(f"\n{'Count':>6}  {'Name':<42}  {'Sugg':<6}  {'Basis'}")
    print("-" * 110)
    for count, name, sugg, basis, comps in rows:
        print(f"{count:>6}  {name[:42]:<42}  {sugg or '?':<6}  {(basis or '')[:40]}")

    print("\n" + "=" * 74)
    print("READY-TO-PASTE ENTRIES (fill in codes marked '???' yourself)")
    print("=" * 74)
    for _, name, sugg, _, _ in rows:
        if sugg:
            print(f"    {name!r}: {sugg!r},")
        else:
            print(f"    {name!r}: '???',   # no suggestion — pick a code")


if __name__ == "__main__":
    main()