#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Merge submission_intervals_v3.csv with weather_data_submission.csv by idx and compute wind_angle.

Definition:
- features.track: aircraft track/heading in degrees [0, 360)
- weather.wind_direction: meteorological direction the wind COMES FROM (°)
- wind_towards = (wind_direction + 180) % 360
- wind_angle = absolute smallest angle between track and wind_towards, in [0, 180]

Input  (default paths):
  data/processed/submission_intervals_v3.csv
  data/processed/weather_data_submission.csv
Output:
  data/processed/submission_intervals_v4.csv
"""

from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR   = Path("data/processed")
FEATURES_IN = DATA_DIR / "submission_intervals_v3.csv"
WEATHER_IN  = DATA_DIR / "weather_data_submission.csv"
OUT_PATH    = DATA_DIR / "submission_intervals_v4.csv"

REQ_FEATURE_COLS = {"idx", "track"}
REQ_WEATHER_COLS = {"idx", "wind_direction"}

def _wrap_to_minus180_180(delta_deg: pd.Series) -> pd.Series:
    """Wrap degrees to (-180, 180]; result is in [-180, 180)."""
    return (delta_deg + 180.0) % 360.0 - 180.0

def compute_wind_angle(track_deg: pd.Series, wind_from_deg: pd.Series) -> pd.Series:
    """Angle between aircraft track and wind TOWARDS direction (deg in [0, 180])."""
    # Normalize inputs to [0, 360)
    track = track_deg.astype(float) % 360.0
    wind_from = wind_from_deg.astype(float) % 360.0
    wind_towards = (wind_from + 180.0) % 360.0
    # Smallest signed difference, then absolute value
    diff = _wrap_to_minus180_180(track - wind_towards)
    return diff.abs()

def main() -> int:
    # Sanity: files exist
    if not FEATURES_IN.exists():
        print(f"ERROR: Missing features file: {FEATURES_IN}", file=sys.stderr)
        return 1
    if not WEATHER_IN.exists():
        print(f"ERROR: Missing weather file: {WEATHER_IN}", file=sys.stderr)
        return 1

    # Load
    try:
        feats = pd.read_csv(FEATURES_IN, low_memory=False)
    except Exception as e:
        print(f"ERROR reading {FEATURES_IN}: {e}", file=sys.stderr)
        return 1

    try:
        wx = pd.read_csv(WEATHER_IN, low_memory=False)
    except Exception as e:
        print(f"ERROR reading {WEATHER_IN}: {e}", file=sys.stderr)
        return 1

    # Check required columns
    missing_f = REQ_FEATURE_COLS - set(feats.columns)
    missing_w = REQ_WEATHER_COLS - set(wx.columns)
    if missing_f:
        print(f"ERROR: features file missing columns: {sorted(missing_f)}", file=sys.stderr)
        return 1
    if missing_w:
        print(f"ERROR: weather file missing columns: {sorted(missing_w)}", file=sys.stderr)
        return 1

    # Ensure numeric types for the calculation (coerce to NaN if bad)
    feats["track"] = pd.to_numeric(feats["track"], errors="coerce")
    wx["wind_direction"] = pd.to_numeric(wx["wind_direction"], errors="coerce")

    # Merge by idx (inner: keep rows that exist in both)
    merged = pd.merge(feats, wx, on="idx", how="inner", suffixes=("", "_wx"))

    # Compute wind_angle
    merged["wind_angle"] = compute_wind_angle(merged["track"], merged["wind_direction"])

    # Save
    try:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(OUT_PATH, index=False)
    except Exception as e:
        print(f"ERROR writing {OUT_PATH}: {e}", file=sys.stderr)
        return 1

    # Quick summary
    total_in = len(feats)
    total_wx = len(wx)
    total_out = len(merged)
    print(
        f"Done. features rows: {total_in}, weather rows: {total_wx}, merged rows: {total_out}\n"
        f"Wrote: {OUT_PATH}"
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
