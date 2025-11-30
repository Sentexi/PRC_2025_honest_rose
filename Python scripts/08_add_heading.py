#!/usr/bin/env python3
# 08_add_heading.py
"""
Append median course/heading ("track") per idx to features_intervals_v1_UPDATE.csv
and write features_intervals_v2.csv.

- Reads:  data/processed/features_intervals_v1_UPDATE.csv   (must contain column: idx)
- For each row, opens: data/processed/Intervals/{idx}_flight_data.csv
  and computes the median of the 'track' column (numeric, NaNs ignored).
- Appends the result as a new column 'track' and saves:
  data/processed/features_intervals_v2.csv

Notes:
- Missing files or missing/non-numeric 'track' values yield NaN in the output.
- Median is the standard numeric median (not circular); adjust if you later
  need circular statistics for headings.
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np

# optional progress bar (won't fail if not installed)
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(it, **kwargs):  # fallback
        return it


FEATURES_IN = Path("data/processed/features_intervals_v1_UPDATE.csv")
INTERVALS_DIR = Path("data/processed/Intervals")
FEATURES_OUT = Path("data/processed/features_intervals_v2.csv")


def median_track_for_idx(idx_val) -> float:
    """
    Return numeric median of 'track' for the given idx from its intervals CSV.
    If file/column is missing or no valid numeric values exist, returns np.nan.
    """
    file_path = INTERVALS_DIR / f"{idx_val}_flight_data.csv"
    if not file_path.is_file():
        # file missing -> NaN
        return np.nan

    try:
        # Load only the 'track' column; coerce to numeric and drop NaNs
        s = pd.read_csv(file_path, usecols=["track"], low_memory=False)["track"]
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            return np.nan
        return float(s.median())
    except ValueError:
        # 'track' not present or other read issue
        return np.nan
    except Exception:
        # Any unexpected read/parse error -> NaN but keep pipeline moving
        return np.nan


def main():
    if not FEATURES_IN.is_file():
        print(f"ERROR: Input file not found: {FEATURES_IN}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(FEATURES_IN, low_memory=False)

    if "idx" not in df.columns:
        print("ERROR: Column 'idx' not found in input CSV.", file=sys.stderr)
        sys.exit(1)

    # Compute median track per row (aligns 1:1 with rows; robust if idx repeats)
    medians = []
    for idx_val in tqdm(df["idx"].tolist(), desc="Computing track medians"):
        medians.append(median_track_for_idx(idx_val))

    # Append new column
    df["track"] = medians

    # Save combined output
    FEATURES_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_OUT, index=False)

    print(f"Done. Wrote: {FEATURES_OUT}")


if __name__ == "__main__":
    main()
