#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process *_intervals_v4.csv files to:
- map aircraft_type -> aircraft_size (narrow/wide/other)
- collapse rare aircraft_type codes (< MIN_COUNT occurrences) per file:
    * known narrow -> "other narrow body"
    * known wide   -> "other wide body"
    * unknown code -> "other"
- derive flight_phase from alt_diff with a small deadband
- write to *_intervals_v5.csv

Runs on:
  data/processed/features_intervals_v4.csv     -> ..._v5.csv
  data/processed/submission_intervals_v4.csv   -> ..._v5.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

MIN_COUNT = 130          # collapse categories with fewer than this many rows
ALT_DEADBAND_FT = 50.0   # |alt_diff| <= 50 ft -> "cruise"

# Mapping provided by you (kept exactly as specified)
SIZE_MAP = {
    "A20N": "narrow body",
    "A21N": "narrow body",
    "A320": "narrow body",
    "A321": "narrow body",
    "B38M": "narrow body",
    "B737": "narrow body",
    "B738": "narrow body",
    "B739": "narrow body",
    "A306": "wide body",
    "A319": "wide body",
    "A332": "wide body",
    "A333": "wide body",
    "A359": "wide body",
    "B744": "wide body",
    "B748": "wide body",
    "B772": "wide body",
    "B77W": "wide body",
    "B788": "wide body",
    "B789": "wide body",
}

def phase_from_alt_diff(x: float) -> str:
    if pd.isna(x):
        return "cruise"  # fallback
    if x > ALT_DEADBAND_FT:
        return "climb"
    if x < -ALT_DEADBAND_FT:
        return "descent"
    return "cruise"

def process_file(input_path: str, output_path: str) -> None:
    input_p = Path(input_path)
    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_p, low_memory=False)

    # Ensure required columns exist
    required_cols = {"aircraft_type", "alt_diff"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{input_p} is missing required columns: {sorted(missing)}")

    # Ensure types
    df["aircraft_type"] = df["aircraft_type"].astype(str)

    # 1) aircraft_size from mapping; unknown -> "other"
    df["aircraft_size"] = df["aircraft_type"].map(SIZE_MAP).fillna("other")

    # 2) collapse rare aircraft_type by counts BEFORE any replacement (per file)
    counts = df["aircraft_type"].value_counts(dropna=False)
    rare_codes = set(counts[counts < MIN_COUNT].index)

    rare_mask = df["aircraft_type"].isin(rare_codes)
    narrow_mask = rare_mask & (df["aircraft_size"] == "narrow body")
    wide_mask   = rare_mask & (df["aircraft_size"] == "wide body")
    other_mask  = rare_mask & (df["aircraft_size"] == "other")  # unknown code & rare

    df.loc[narrow_mask, "aircraft_type"] = "other narrow body"
    df.loc[wide_mask,   "aircraft_type"] = "other wide body"
    df.loc[other_mask,  "aircraft_type"] = "other"

    # 3) flight_phase from alt_diff (numeric + deadband)
    df["alt_diff"] = pd.to_numeric(df["alt_diff"], errors="coerce")
    df["flight_phase"] = df["alt_diff"].apply(phase_from_alt_diff)

    # Write
    df.to_csv(output_p, index=False)
    print(f"✓ {input_p.name} -> {output_p.name} ({len(df):,} rows)")

def main():
    process_file(
        "data/processed/features_intervals_v4.csv",
        "data/processed/features_intervals_v5.csv",
    )
    process_file(
        "data/processed/submission_intervals_v4.csv",
        "data/processed/submission_intervals_v5.csv",
    )
    process_file(
        "data/processed/final_submission_intervals_v3.csv",
        "data/processed/final_submission_intervals_v5.csv",
    )

if __name__ == "__main__":
    main()
