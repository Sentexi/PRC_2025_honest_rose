#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
08A_add_heading_submission.py

Submission variant of 08_add_heading.py:
- Reads  data/processed/submission_intervals_v1.csv
- For each idx, opens data/processed/Intervals_submission/{idx}_flight_data.csv
- Extracts heading from 'track' (preferred) or 'heading' column, numeric median
- Writes  data/processed/submission_intervals_v2.csv
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm


def detect_heading_column(csv_path: str) -> Optional[str]:
    """
    Read only the header to decide which column to use for heading.
    Preference: 'track' > 'heading'.
    """
    try:
        hdr = pd.read_csv(csv_path, nrows=0)
    except Exception:
        return None
    cols = set(hdr.columns)
    if "track" in cols:
        return "track"
    if "heading" in cols:
        return "heading"
    # Sometimes weird capitalization occurs
    for c in cols:
        cl = c.strip().lower()
        if cl == "track":
            return c
        if cl == "heading":
            return c
    return None


def median_heading_from_file(csv_path: str) -> float:
    """
    Return numeric median of heading from a single points CSV.
    Uses 'track' if present, else 'heading'. Returns np.nan if unavailable.
    """
    if not os.path.exists(csv_path):
        return np.nan

    col = detect_heading_column(csv_path)
    if col is None:
        return np.nan

    try:
        s = pd.read_csv(
            csv_path,
            usecols=[col],
            dtype={col: "float64"},
            low_memory=False
        )[col]
    except ValueError:
        # If usecols failed (column disappeared?), fallback to full read
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            if col not in df.columns:
                return np.nan
            s = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            return np.nan
    except Exception:
        return np.nan

    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return np.nan
    # Plain numeric median (NOT circular), per user's request for parity
    return float(np.median(s.values))


def compute_track_medians(df: pd.DataFrame, points_dir: str) -> pd.Series:
    """
    For each idx in df, compute (and cache) the numeric median heading from the points file.
    """
    med_cache: Dict[int, float] = {}
    out = np.full(len(df), np.nan, dtype=float)

    for i, idx in enumerate(tqdm(df["idx"].tolist(), desc="heading (median)", unit="interval")):
        if pd.isna(idx):
            out[i] = np.nan
            continue
        idx_int = int(idx)
        if idx_int in med_cache:
            out[i] = med_cache[idx_int]
            continue

        csv_path = os.path.join(points_dir, f"{idx_int}_flight_data.csv")
        med_val = median_heading_from_file(csv_path)
        med_cache[idx_int] = med_val
        out[i] = med_val

    return pd.Series(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", default="data/processed/submission_intervals_v1.csv",
                    help="Input features CSV (submission_intervals_v1.csv)")
    ap.add_argument("--points-dir", default="data/processed/Intervals_submission",
                    help="Folder containing {idx}_flight_data.csv (submission points)")
    ap.add_argument("--out", dest="out_csv", default="data/processed/submission_intervals_v2.csv",
                    help="Output features CSV (submission_intervals_v2.csv)")
    args = ap.parse_args()

    in_path = Path(args.in_csv)
    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    df = pd.read_csv(in_path, low_memory=False)

    if "idx" not in df.columns:
        raise ValueError("Column 'idx' is required in the input CSV.")

    # Compute numeric median heading per idx
    track_median = compute_track_medians(df, args.points_dir)

    # Insert/overwrite `track` column to mirror the training feature name
    df["track"] = track_median

    # Save
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved → {args.out_csv}")


if __name__ == "__main__":
    main()
