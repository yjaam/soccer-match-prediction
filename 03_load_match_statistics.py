#!/usr/bin/env python3
"""
Scrape match CSVs from football-data.co.uk (raw CSV endpoints) for the Big Five leagues
and seasons 2014–2026 (inclusive), concatenate, and save into ./matchhistory_data/.

URL pattern:
  https://www.football-data.co.uk/mmz4281/{SEASON_CODE}/{DIV}.csv

Defaults:
  - leagues: E0, D1, I1, SP1, F1
  - seasons (start years): 2014..2026 inclusive => 1415..2627

Output:
  - Writes combined CSV to: matchhistory_data/match_statistics.csv (by default)
  - Adds: source_url, div, season_code, season_start_year
  - Preserves all original CSV columns.
"""

from __future__ import annotations

import argparse
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import requests


BASE = "https://www.football-data.co.uk/mmz4281"

DEFAULT_LEAGUES = ["E0", "D1", "I1", "SP1", "F1"]
DEFAULT_START_YEAR = 2014
DEFAULT_END_YEAR = 2026

DEFAULT_OUT_DIR = Path("matchhistory_data")
DEFAULT_OUT_FILE = "match_statistics.csv"


@dataclass(frozen=True)
class FetchTarget:
    div: str
    season_start_year: int

    @property
    def season_code(self) -> str:
        yy = self.season_start_year % 100
        yy2 = (self.season_start_year + 1) % 100
        return f"{yy:02d}{yy2:02d}"

    @property
    def url(self) -> str:
        return f"{BASE}/{self.season_code}/{self.div}.csv"


def iter_targets(divs: Iterable[str], start_year: int, end_year: int) -> List[FetchTarget]:
    if end_year < start_year:
        raise ValueError("end_year must be >= start_year")
    return [FetchTarget(div=d, season_start_year=y) for d in divs for y in range(start_year, end_year + 1)]


def fetch_csv(url: str, timeout: int = 60) -> bytes:
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "match-history-scraper/1.0"})
    if resp.status_code != 200:
        raise requests.HTTPError(f"HTTP {resp.status_code} for {url}", response=resp)
    return resp.content


def parse_csv_bytes(content: bytes) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(content), encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(content))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(DEFAULT_OUT_DIR / DEFAULT_OUT_FILE),
        help="Output CSV path (default: matchhistory_data/match_statistics.csv)",
    )
    ap.add_argument("--leagues", nargs="*", default=None, help="Division codes, e.g. E0 D1 I1 SP1 F1")
    ap.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR, help="Season start year (inclusive)")
    ap.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR, help="Season start year (inclusive)")
    ap.add_argument(
        "--on-missing",
        choices=["skip", "fail"],
        default="skip",
        help="What to do if a URL is missing (HTTP != 200). Default: skip",
    )
    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds")
    args = ap.parse_args()

    out_path = Path(args.out)
    # Ensure it goes into matchhistory_data/ by default; if user passes a bare filename,
    # we still place it into matchhistory_data/.
    if out_path.parent == Path("."):
        out_path = DEFAULT_OUT_DIR / out_path

    # Your note: folder already exists. We won't create it silently; but we can fail with a clear error.
    if not DEFAULT_OUT_DIR.exists():
        raise RuntimeError(
            f"Expected output directory '{DEFAULT_OUT_DIR}' to exist. "
            "Create it first (you said you already did)."
        )

    divs = args.leagues if args.leagues else DEFAULT_LEAGUES
    targets = iter_targets(divs, args.start_year, args.end_year)

    frames: List[pd.DataFrame] = []
    failures: List[Tuple[str, str]] = []

    for t in targets:
        url = t.url
        try:
            content = fetch_csv(url, timeout=args.timeout)
            df = parse_csv_bytes(content)

            df.insert(0, "source_url", url)
            df.insert(1, "div", t.div)
            df.insert(2, "season_code", t.season_code)
            df.insert(3, "season_start_year", t.season_start_year)

            frames.append(df)
            print(f"OK   {t.div} {t.season_code}  rows={len(df):,}  cols={df.shape[1]:,}")
        except Exception as e:
            msg = str(e)
            failures.append((url, msg))
            print(f"MISS {t.div} {t.season_code}  {msg}")
            if args.on_missing == "fail":
                raise

    if not frames:
        raise RuntimeError("No CSVs were downloaded successfully; nothing to write.")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.to_csv(out_path, index=False)

    print(f"\nWrote {out_path}")
    print(f"Total rows: {len(combined):,}  Total cols: {combined.shape[1]:,}")

    if failures:
        print(f"\nMissing/failed downloads: {len(failures)}")
        for url, reason in failures[:20]:
            print(f" - {url}  ({reason})")
        if len(failures) > 20:
            print(f" ... and {len(failures) - 20} more")


if __name__ == "__main__":
    main()