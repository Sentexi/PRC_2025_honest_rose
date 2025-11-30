#!/usr/bin/env python3
# 09_map_regions.py
"""
Map ICAO airport codes to continents/regions.

Reads:
  data/processed/features_intervals_v2.csv
    - must contain columns: origin_icao, dest_icao

Adds:
  - origin_region
  - dest_region

Writes:
  data/processed/features_intervals_v3.csv

Notes:
- Mapping is based on ICAO location indicator prefixes.
- Two-letter overrides are applied first (e.g., PA=Alaska=North America, PH=Hawaii=Oceania),
  then single-letter fallbacks.
- Unknown or malformed codes map to "Unknown".
"""

from pathlib import Path
import sys
import csv
import pandas as pd

IN_PATH  = Path("data/processed/features_intervals_v2.csv")
OUT_PATH = Path("data/processed/features_intervals_v3.csv")


def detect_delimiter(path: Path) -> str:
    """Detect delimiter (comma/semicolon/tab/pipe). Defaults to comma."""
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        sample = f.read(65536)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        return ","


# --- ICAO -> Region mapping helpers -------------------------------------------------------------

# Two-letter overrides (checked first)
TWO_LETTER = {
    # North America special cases
    "PA": "North America",  # Alaska (USA)
    "BG": "North America",  # Greenland

    # Oceania special cases (US & Pacific territories)
    "PH": "Oceania",  # Hawaii
    "PJ": "Oceania",
    "PK": "Oceania",
    "PL": "Oceania",
    "PM": "Oceania",
    "PN": "Oceania",
    "PO": "Oceania",
    "PP": "Oceania",
    "PT": "Oceania",
    "PW": "Oceania",
    "PG": "Oceania",

    # Europe specifics
    "BI": "Europe",        # Iceland
    "UK": "Europe",        # Ukraine
    "UM": "Europe",        # Belarus
    "LC": "Europe",        # Cyprus (treated as Europe here)

    # Western Asia / Middle East specifics
    "LL": "Asia",          # Israel
    "LT": "Asia",          # Turkey (treated as Asia here)
}

# Single-letter fallbacks (applied if no two-letter override matched)
ONE_LETTER = {
    "E": "Europe",         # UK/IE/Scandinavia/Benelux/Poland/Baltics etc.
    "L": "Europe",         # Most of continental Europe (FR/ES/IT/GR/AT/CH/...),
                           # except overrides above (LL, LT already handled).
    "B": "Europe",         # Iceland/Greenland space; Greenland handled by 'BG' override.

    "C": "North America",  # Canada
    "K": "North America",  # USA (CONUS)
    "M": "North America",  # Mexico, Central America
    "T": "North America",  # Caribbean

    "P": "Oceania",        # Pacific (except 'PA' Alaska handled above)
    "N": "Oceania",        # South Pacific (NZ, Fiji, etc.)
    "Y": "Oceania",        # Australia

    "D": "Africa",         # North/West Africa
    "F": "Africa",         # Southern/Central Africa
    "G": "Africa",         # West Africa & Canary Islands
    "H": "Africa",         # Northeast/East Africa

    "O": "Asia",           # Middle East
    "R": "Asia",           # Japan/Korea/Philippines/Taiwan
    "U": "Asia",           # Russia/Central Asia (EU part partly misclassified)
    "V": "Asia",           # South/Southeast Asia
    "W": "Asia",           # Indonesia/Malaysia/Singapore/Brunei
    "Z": "Asia",           # China

    "S": "South America",  # South America block
    "A": "Antarctica",     # Rare/placeholder
}


def icao_to_region(code: str) -> str:
    """Return continent/region for an ICAO code using prefix heuristics."""
    if not isinstance(code, str):
        return "Unknown"
    code = code.strip().upper()
    if len(code) < 2:
        return "Unknown"

    two = code[:2]
    if two in TWO_LETTER:
        return TWO_LETTER[two]

    one = code[0]
    return ONE_LETTER.get(one, "Unknown")


# --- Main ---------------------------------------------------------------------------------------

def main():
    if not IN_PATH.is_file():
        print(f"ERROR: Input file not found: {IN_PATH}", file=sys.stderr)
        sys.exit(1)

    # Read with delimiter auto-detection; keep all columns
    sep = detect_delimiter(IN_PATH)
    try:
        df = pd.read_csv(IN_PATH, sep=sep, dtype=str, low_memory=False)
    except Exception as e:
        print(f"ERROR reading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    # Normalize headers
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

    # Check required cols
    required = {"origin_icao", "dest_icao"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Required columns missing: {missing}", file=sys.stderr)
        sys.exit(1)

    # Map to regions (vectorized apply)
    df["origin_region"] = df["origin_icao"].fillna("").map(icao_to_region)
    df["dest_region"]   = df["dest_icao"].fillna("").map(icao_to_region)

    # Write output (CSV, comma-separated)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Done. Wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
