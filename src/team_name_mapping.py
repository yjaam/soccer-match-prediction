#!/usr/bin/env python3
"""
Practical team-name harmonizer.

Goal
----
Given messy team names coming from multiple sources (some long legal names, some short),
map them to a canonical *short* name vocabulary (your `football_teams.py` list).

Approach (practical + robust)
-----------------------------
1) Normalize strings (casefold, strip accents, remove punctuation, collapse whitespace).
2) Apply curated alias rules for common tricky cases (e.g., "Paris Saint-Germain..." -> "Paris SG").
3) Fuzzy match remaining names against the canonical short list using RapidFuzz if installed;
   otherwise fall back to Python's difflib.
4) Provide a reusable `map_series()` for pandas columns and a CLI to process a CSV.

What you need to do
-------------------
- Paste your two lists into `football_clubs.py` and `football_teams.py` as `FOOTBALL_CLUBS`
  and `FOOTBALL_TEAMS` respectively (examples below).
- Then run this script to generate a mapping and apply it to your dataset columns.

Example `football_teams.py`:
  FOOTBALL_TEAMS = [ ... ]   # list of canonical short names

Example `football_clubs.py`:
  FOOTBALL_CLUBS = [ ... ]   # list of long names (optional; can be used to pre-build mapping)

CLI examples
------------
# Just build + print the mapping (long->short) using both vocabularies:
python team_name_mapper.py --print-mapping

# Apply mapping to a CSV columns (e.g., HomeTeam, AwayTeam):
python team_name_mapper.py --in matchhistory_data/match_statistics.csv \
  --out matchhistory_data/match_statistics_mapped.csv \
  --cols HomeTeam AwayTeam

Notes
-----
- This will never be perfect without some manual review. Use --review to emit a CSV
  of low-confidence matches for human correction.
"""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

# ----------------------------
# Load your vocab lists
# ----------------------------
try:
    from football_teams import FOOTBALL_TEAMS  # canonical short names
except Exception as e:
    raise RuntimeError(
        "Could not import FOOTBALL_TEAMS from football_teams.py. "
        "Create football_teams.py containing a list named FOOTBALL_TEAMS."
    ) from e

# football_clubs is optional (used to prebuild mapping candidates)
try:
    from football_clubs import FOOTBALL_CLUBS
except Exception:
    FOOTBALL_CLUBS = []


# ----------------------------
# Normalization
# ----------------------------

_PUNCT_RE = re.compile(r"[^a-z0-9\s]+")
_WS_RE = re.compile(r"\s+")


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def norm(s: str) -> str:
    """Aggressive normalization for matching."""
    if s is None:
        return ""
    s = str(s).strip()
    if not s:
        return ""
    s = strip_accents(s)
    s = s.casefold()
    s = s.replace("&", " and ")
    s = s.replace("’", "'")
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    # drop very common noise words in long legal club names
    noise = {
        "football", "club", "fc", "cf", "afc", "sc", "sv", "ss",
        "association", "sporting", "athletic", "societe", "societa",
        "de", "la", "le", "el", "der", "des", "du", "s", "a", "d", "sad",
        "spa", "s p a", "s p a", "s p a", "sp a", "hsc",
        "olympique", "stade", "verein", "fur", "fussball", "und",
        "sportverein", "futbol", "futball", "calcio",
    }
    parts = [p for p in s.split(" ") if p and p not in noise]
    return " ".join(parts)


# ----------------------------
# Curated aliases (high precision)
# ----------------------------
# Keys and values are normalized via norm() at build time.
RAW_ALIASES: Dict[str, str] = {
    # France
    "paris saint germain": "Paris SG",
    "paris saint germain fc": "Paris SG",
    "olympique lyonnais": "Lyon",
    "olympique de marseille": "Marseille",
    "as saint etienne": "St Etienne",
    "association sportive de monaco": "Monaco",
    "stade rennais": "Rennes",
    "stade de reims": "Reims",
    "racing club de lens": "Lens",
    "montpellier hsc": "Montpellier",
    "football club de nantes": "Nantes",
    "football club de metz": "Metz",
    "toulouse football club": "Toulouse",
    "stade brestois 29": "Brest",
    "association de la jeunesse auxerroise": "Auxerre",
    "le havre athletic club": "Le Havre",
    "ac ajaccio": "Ajaccio",
    "ajaccio gfco": "Ajaccio GFCO",
    # Spain
    "futbol club barcelona": "Barcelona",
    "real madrid club de futbol": "Real Madrid",
    "club atletico de madrid": "Ath Madrid",
    "athletic club bilbao": "Ath Bilbao",
    "real sociedad de futbol": "Sociedad",
    "real club celta de vigo": "Celta",
    "reial club deportiu espanyol": "Espanol",
    "valencia club de futbol": "Valencia",
    "villarreal club de futbol": "Villarreal",
    "real betis balompie": "Betis",
    "ud las palmas": "Las Palmas",
    "club atletico osasuna": "Osasuna",
    "getafe club de futbol": "Getafe",
    "rayo vallecano": "Vallecano",
    "deportivo alaves": "Alaves",
    "girona futbol club": "Girona",
    "cadiz cf": "Cadiz",
    "real valladolid cf": "Valladolid",
    "sd eibar": "Eibar",
    "sd huesca": "Huesca",
    "cd leganes": "Leganes",
    "elche club de futbol": "Elche",
    "deportivo de la coruna": "La Coruna",
    "cordoba cf": "Cordoba",
    # Germany
    "fc bayern munchen": "Bayern Munich",
    "borussia dortmund": "Dortmund",
    "bayer 04 leverkusen": "Leverkusen",
    "eintracht frankfurt": "Ein Frankfurt",
    "hamburger sport verein": "Hamburg",
    "1 fussball club koln": "FC Koln",
    "verein fur leibesubungen wolfsburg": "Wolfsburg",
    "sport club freiburg": "Freiburg",
    "1 fussball und sportverein mainz 05": "Mainz",
    "turn und sportgemeinschaft 1899 hoffenheim": "Hoffenheim",
    "borussia monchengladbach": "M'gladbach",
    "fussball club augsburg 1907": "Augsburg",
    "spvgg greuther furth": "Greuther Furth",
    "vfl bochum": "Bochum",
    "1 fussballclub union berlin": "Union Berlin",
    "sv darmstadt 98": "Darmstadt",
    "fc ingolstadt 04": "Ingolstadt",
    "rasenballsport leipzig": "RB Leipzig",
    "fortuna dusseldorf": "Fortuna Dusseldorf",
    "1 fussballclub heidenheim 1846": "Heidenheim",
    "fussball club st pauli von 1910": "St Pauli",
    "holstein kiel": "Holstein Kiel",
    # Italy
    "associazione calcio milan": "Milan",
    "football club internazionale milano": "Inter",
    "associazione sportiva roma": "Roma",
    "societa sportiva lazio": "Lazio",
    "societa sportiva calcio napoli": "Napoli",
    "juventus football club": "Juventus",
    "associazione calcio fiorentina": "Fiorentina",
    "unione sportiva sassuolo calcio": "Sassuolo",
    "atalanta bergamasca calcio": "Atalanta",
    "bologna football club 1909": "Bologna",
    "parma calcio 1913": "Parma",
    "torino calcio": "Torino",
    "cagliari calcio": "Cagliari",
    "udinese calcio": "Udinese",
    "unione sportiva lecce": "Lecce",
    "brescia calcio": "Brescia",
    "spezia calcio": "Spezia",
    "us salernitana 1919": "Salernitana",
    "venezia fc": "Venezia",
    "ac monza": "Monza",
    "unione sportiva cremonese": "Cremonese",
    "palermo fc": "Palermo",
    "hellas verona": "Verona",
    "chievo verona": "Chievo",
    "delfino pescara 1936": "Pescara",
    "fc crotone": "Crotone",
    "benevento calcio": "Benevento",
    "spal": "Spal",
    "ac carpi": "Carpi",
    "frosinone calcio": "Frosinone",
    "cesena fc": "Cesena",
    "genoa cricket and football club": "Genoa",
    # England
    "manchester united": "Man United",
    "manchester city": "Man City",
    "tottenham hotspur": "Tottenham",
    "west bromwich albion": "West Brom",
    "queens park rangers": "QPR",
    "nottingham forest": "Nott'm Forest",
    "wolverhampton wanderers": "Wolves",
    "brighton and hove albion": "Brighton",
}

def build_alias_map() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in RAW_ALIASES.items():
        out[norm(k)] = v
    return out


# ----------------------------
# Fuzzy matcher
# ----------------------------

def _build_canonical_index(canon: Sequence[str]) -> Tuple[List[str], Dict[str, str]]:
    canon = [c for c in canon if isinstance(c, str) and c.strip()]
    norm_to_canon: Dict[str, str] = {}
    for c in canon:
        norm_to_canon[norm(c)] = c
    return canon, norm_to_canon


def _get_fuzzy_impl():
    """
    Prefer RapidFuzz (much better). Fall back to difflib.
    Returns: (match_func, score_name)
    """
    try:
        from rapidfuzz import process, fuzz  # type: ignore

        def match_one(query: str, choices: Sequence[str]) -> Tuple[Optional[str], float]:
            if not query:
                return None, 0.0
            # token_sort_ratio works well for swapped word order
            res = process.extractOne(query, choices, scorer=fuzz.token_sort_ratio)
            if res is None:
                return None, 0.0
            choice, score, _idx = res
            return choice, float(score)

        return match_one, "rapidfuzz.token_sort_ratio"
    except Exception:
        import difflib

        def match_one(query: str, choices: Sequence[str]) -> Tuple[Optional[str], float]:
            if not query:
                return None, 0.0
            best = difflib.get_close_matches(query, choices, n=1, cutoff=0.0)
            if not best:
                return None, 0.0
            # difflib doesn't give a great comparable score; approximate via SequenceMatcher ratio
            score = difflib.SequenceMatcher(a=query, b=best[0]).ratio() * 100.0
            return best[0], float(score)

        return match_one, "difflib.SequenceMatcher"


@dataclass
class MatchResult:
    original: str
    mapped: Optional[str]
    score: float
    method: str


def build_mapper(
    canonical_short: Sequence[str],
    min_score: float = 86.0,
) -> Tuple[Callable[[object], Optional[str]], Dict[str, MatchResult]]:
    """
    Returns:
      - mapper function: maps a raw team name -> canonical short name (or None)
      - debug dict: normalized_original -> MatchResult (for review/export)
    """
    canon, norm_to_canon = _build_canonical_index(canonical_short)
    alias_map = build_alias_map()

    # Fuzzy operates on normalized canonical keys (not raw), but we return raw canonical short.
    canonical_norm_keys = list(norm_to_canon.keys())
    fuzzy_match_one, fuzzy_name = _get_fuzzy_impl()

    debug: Dict[str, MatchResult] = {}

    def map_one(x: object) -> Optional[str]:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        raw = str(x).strip()
        if not raw:
            return None

        n = norm(raw)

        # exact match to canonical
        if n in norm_to_canon:
            debug[n] = MatchResult(original=raw, mapped=norm_to_canon[n], score=100.0, method="exact")
            return norm_to_canon[n]

        # curated alias
        if n in alias_map:
            debug[n] = MatchResult(original=raw, mapped=alias_map[n], score=100.0, method="alias")
            return alias_map[n]

        # fuzzy against canonical normalized keys
        best_norm, score = fuzzy_match_one(n, canonical_norm_keys)
        if best_norm is None:
            debug[n] = MatchResult(original=raw, mapped=None, score=0.0, method=f"fuzzy:{fuzzy_name}")
            return None

        mapped = norm_to_canon.get(best_norm)
        if mapped is not None and score >= min_score:
            debug[n] = MatchResult(original=raw, mapped=mapped, score=score, method=f"fuzzy:{fuzzy_name}")
            return mapped

        debug[n] = MatchResult(original=raw, mapped=None, score=score, method=f"fuzzy:{fuzzy_name}")
        return None

    return map_one, debug


def map_series(
    s: pd.Series,
    canonical_short: Sequence[str],
    min_score: float = 86.0,
    unknown_value: Optional[str] = None,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Map a pandas Series of team names to canonical short names.

    Returns:
      - mapped Series
      - review DataFrame with match diagnostics (good for manual fixes)
    """
    mapper, debug = build_mapper(canonical_short=canonical_short, min_score=min_score)

    mapped = s.map(mapper)

    review = (
        pd.DataFrame(
            [
                {
                    "normalized": k,
                    "original": v.original,
                    "mapped": v.mapped,
                    "score": v.score,
                    "method": v.method,
                }
                for k, v in debug.items()
            ]
        )
        .sort_values(["mapped", "score"], ascending=[True, False], na_position="first")
        .reset_index(drop=True)
    )

    if unknown_value is not None:
        mapped = mapped.fillna(unknown_value)

    return mapped, review


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=None, help="Input CSV to transform")
    ap.add_argument("--out", dest="out", default=None, help="Output CSV (required if --in is set)")
    ap.add_argument("--cols", nargs="*", default=None, help="Columns to map (e.g., HomeTeam AwayTeam)")
    ap.add_argument("--min-score", type=float, default=86.0, help="Minimum fuzzy score to accept")
    ap.add_argument("--review", default=None, help="Write review CSV of matches (recommended)")
    ap.add_argument("--print-mapping", action="store_true", help="Print mapping candidates for FOOTBALL_CLUBS")
    args = ap.parse_args()

    canonical = [x for x in FOOTBALL_TEAMS if isinstance(x, str) and x.strip()]

    if args.print_mapping:
        if not FOOTBALL_CLUBS:
            raise RuntimeError(
                "FOOTBALL_CLUBS is empty or football_clubs.py is missing. "
                "Add FOOTBALL_CLUBS to football_clubs.py if you want --print-mapping."
            )
        mapper, debug = build_mapper(canonical_short=canonical, min_score=args.min_score)
        rows = []
        for club in FOOTBALL_CLUBS:
            mapped = mapper(club)
            n = norm(club)
            mr = debug.get(n)
            rows.append(
                {
                    "club_long": club,
                    "club_long_normalized": n,
                    "mapped_short": mapped,
                    "score": None if mr is None else mr.score,
                    "method": None if mr is None else mr.method,
                }
            )
        df_map = pd.DataFrame(rows).sort_values(["mapped_short", "score"], ascending=[True, False], na_position="first")
        print(df_map.to_string(index=False))
        return

    if args.inp:
        if not args.out:
            raise ValueError("--out is required when using --in")
        if not args.cols:
            raise ValueError("--cols is required (team name columns to map)")

        df = pd.read_csv(args.inp, low_memory=False)

        all_reviews = []
        for c in args.cols:
            if c not in df.columns:
                raise ValueError(f"Column '{c}' not found. Available columns: {list(df.columns)}")
            mapped, review = map_series(df[c], canonical_short=canonical, min_score=args.min_score)
            df[c] = mapped
            review.insert(0, "column", c)
            all_reviews.append(review)

        df.to_csv(args.out, index=False)

        if args.review:
            pd.concat(all_reviews, ignore_index=True).to_csv(args.review, index=False)

        print(f"Wrote: {args.out}")
        if args.review:
            print(f"Wrote review: {args.review}")
        return

    raise ValueError("Nothing to do. Use --in/--out/--cols to transform a CSV, or --print-mapping.")


if __name__ == "__main__":
    main()