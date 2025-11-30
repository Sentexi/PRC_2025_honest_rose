#!/usr/bin/env python3
# 09A_map_regions_submission.py
"""
Submission variant of 09_map_regions.py with only the file paths changed.

- Input : data/processed/submission_intervals_v2.csv
  (must contain columns: origin_icao, dest_icao)
- Output: data/processed/submission_intervals_v3.csv

All mapping logic (delimiter detection, ICAO→region tables and function) is
reused directly from the original 09_map_regions.py to ensure full consistency.
"""

from pathlib import Path
import sys
import importlib.util
import pandas as pd

# --- Paths (changed) ---------------------------------------------------------
IN_PATH  = Path("data/processed/submission_intervals_v2.csv")
OUT_PATH = Path("data/processed/submission_intervals_v3.csv")

# --- Load original 09_map_regions.py for consistent logic --------------------
ORIGINAL_SCRIPT = Path(__file__).parent / "09_map_regions.py"

if not ORIGINAL_SCRIPT.is_file():
    print(f"ERROR: Could not find original mapping script at: {ORIGINAL_SCRIPT}", file=sys.stderr)
    sys.exit(1)

spec = importlib.util.spec_from_file_location("regions_orig", str(ORIGINAL_SCRIPT))
regions_orig = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(regions_orig)

# We expect these to exist in the original:
# - regions_orig.detect_delimiter(Path) -> str
# - regions_orig.icao_to_region(code: str) -> str

def main():
    if not IN_PATH.is_file():
        print(f"ERROR: Input file not found: {IN_PATH}", file=sys.stderr)
        sys.exit(1)

    # Read with the same delimiter detection as original
    try:
        sep = regions_orig.detect_delimiter(IN_PATH)
    except Exception as e:
        print(f"ERROR detecting delimiter: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(IN_PATH, sep=sep, dtype=str, low_memory=False)
    except Exception as e:
        print(f"ERROR reading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    # Minimal column checks to match original expectations
    missing = [c for c in ["origin_icao", "dest_icao"] if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)

    # Apply the exact same mapping function from the original script
    df["origin_region"] = df["origin_icao"].fillna("").map(regions_orig.icao_to_region)
    df["dest_region"]   = df["dest_icao"].fillna("").map(regions_orig.icao_to_region)

    # Write out (comma-separated)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Done. Wrote: {OUT_PATH}")

if __name__ == "__main__":
    main()
