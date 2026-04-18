"""
scripts/fetch_gdelt.py

Downloads sampled GDELT 1.0 daily event files for 2022-2024 and computes a
Crisis News Intensity (CNI) metric per country per CERF round period.

CNI = sum of NumArticles for all events where GoldsteinScale < 0 in the country
      during that round period (represents volume of destabilising event coverage).

Sampling strategy: 1st and 15th of every month → 72 days per three-year window,
giving a representative but manageable ~720 MB of raw data.

Outputs:
    data/gdelt_coverage.csv   — CNI per country-year-round
    data/cache/                — per-day parquet cache (gitignored)

Usage:
    source .venv/bin/activate
    python scripts/fetch_gdelt.py
"""

import datetime
import io
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# ── ISO3 (CERF) → FIPS 10-4 (GDELT ActionGeo_CountryCode) ──────────────────
ISO3_TO_FIPS: dict[str, str] = {
    "AFG": "AF",  # Afghanistan
    "DZA": "AG",  # Algeria
    "AGO": "AO",  # Angola
    "BGD": "BG",  # Bangladesh
    "BFA": "UV",  # Burkina Faso
    "BDI": "BY",  # Burundi
    "CMR": "CM",  # Cameroon
    "CAF": "CT",  # Central African Republic
    "TCD": "CD",  # Chad
    "COL": "CO",  # Colombia
    "COD": "CG",  # DR Congo
    "ERI": "ER",  # Eritrea
    "ETH": "ET",  # Ethiopia
    "HTI": "HA",  # Haiti
    "HND": "HO",  # Honduras
    "KEN": "KE",  # Kenya
    "LBN": "LE",  # Lebanon
    "MDG": "MA",  # Madagascar
    "MWI": "MI",  # Malawi
    "MLI": "ML",  # Mali
    "MOZ": "MZ",  # Mozambique
    "MMR": "BM",  # Myanmar
    "NER": "NG",  # Niger
    "NGA": "NI",  # Nigeria
    "PAK": "PK",  # Pakistan
    "SDN": "SU",  # Sudan
    "SSD": "OD",  # South Sudan
    "SYR": "SY",  # Syria
    "UGA": "UG",  # Uganda
    "VEN": "VE",  # Venezuela
    "YEM": "YM",  # Yemen
    "PSE": "GZ",  # Palestinian territories (Gaza); also WE for West Bank
}

# Reverse map; Palestinian territories may appear under either FIPS code
FIPS_TO_ISO3: dict[str, str] = {v: k for k, v in ISO3_TO_FIPS.items()}
FIPS_TO_ISO3["WE"] = "PSE"  # West Bank → same ISO3 as Gaza

TARGET_FIPS: set[str] = set(ISO3_TO_FIPS.values()) | {"WE"}

# ── GDELT 1.0 column positions (0-indexed, tab-separated, no header) ─────────
# Full schema: https://www.gdeltproject.org/data/documentation/GDELT-Event_Codebook-V2.0.pdf
GDELT_USECOLS = [1, 30, 33, 34, 51]
GDELT_COL_NAMES = ["SQLDATE", "GoldsteinScale", "NumArticles", "AvgTone", "ActionGeo_CountryCode"]

GDELT_URL = "http://data.gdeltproject.org/events/{date}.export.CSV.zip"
CACHE_DIR = Path("data/cache")
OUT_PATH = Path("data/gdelt_coverage.csv")


def sample_dates(start_year: int, end_year: int) -> list[str]:
    """Return YYYYMMDD strings for the 1st and 15th of every month."""
    today = datetime.date.today()
    dates: list[str] = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            for day in (1, 15):
                try:
                    dt = datetime.date(year, month, day)
                    if dt <= today:
                        dates.append(dt.strftime("%Y%m%d"))
                except ValueError:
                    pass
    return dates


def download_and_parse(date_str: str) -> Optional[pd.DataFrame]:
    """
    Fetch one GDELT daily export, keep only crisis events in target countries.
    Results are cached as parquet so re-runs are instant.
    """
    cache_path = CACHE_DIR / f"{date_str}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    url = GDELT_URL.format(date=date_str)
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  [skip] {date_str}: {exc}")
        return None

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as fh:
            df = pd.read_csv(
                fh,
                sep="\t",
                header=None,
                usecols=GDELT_USECOLS,
                names=GDELT_COL_NAMES,  # pandas assigns these in usecols order
                dtype=str,
                on_bad_lines="skip",
            )

    # Filter to target countries immediately to keep memory low
    df = df[df["ActionGeo_CountryCode"].isin(TARGET_FIPS)].copy()
    if df.empty:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        return None

    df["GoldsteinScale"] = pd.to_numeric(df["GoldsteinScale"], errors="coerce")
    df["NumArticles"] = pd.to_numeric(df["NumArticles"], errors="coerce").fillna(0).astype(int)
    df["AvgTone"] = pd.to_numeric(df["AvgTone"], errors="coerce")

    # Keep only crisis/destabilising events (negative Goldstein scale)
    crisis_df = df[df["GoldsteinScale"] < 0].copy()

    crisis_df["ISO3"] = crisis_df["ActionGeo_CountryCode"].map(FIPS_TO_ISO3)
    crisis_df["date_parsed"] = pd.to_datetime(crisis_df["SQLDATE"], format="%Y%m%d", errors="coerce")
    crisis_df["Year"] = crisis_df["date_parsed"].dt.year
    # Mirror CERF round assignment: Jan-Jun = I, Jul-Dec = II
    crisis_df["Round"] = crisis_df["date_parsed"].dt.month.apply(lambda m: "I" if m <= 6 else "II")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    crisis_df.to_parquet(cache_path, index=False)
    return crisis_df


def main() -> None:
    dates = sample_dates(2022, 2024)
    print(f"Fetching {len(dates)} daily GDELT files (1st + 15th of each month, 2022-2024)...")

    chunks: list[pd.DataFrame] = []
    for i, date_str in enumerate(dates, 1):
        print(f"  [{i:3d}/{len(dates)}] {date_str}", end="  ")
        chunk = download_and_parse(date_str)
        if chunk is not None and not chunk.empty:
            chunks.append(chunk)
            print(f"→ {len(chunk):,} crisis rows")
        else:
            print("→ (no crisis rows)")

    if not chunks:
        print("\nNo data collected. Check network or GDELT URL.")
        return

    combined = pd.concat(chunks, ignore_index=True)

    # Aggregate to country-year-round
    cni = (
        combined[combined["ISO3"].notna()]
        .groupby(["ISO3", "Year", "Round"], as_index=False)
        .agg(
            # Primary metric: total articles covering negative events
            crisis_article_volume=("NumArticles", "sum"),
            # Supplementary
            neg_event_count=("NumArticles", "count"),
            avg_tone=("AvgTone", "mean"),
            avg_goldstein=("GoldsteinScale", "mean"),
        )
        .sort_values(["ISO3", "Year", "Round"])
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cni.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(cni)} country-round records → {OUT_PATH}")
    print(cni.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
