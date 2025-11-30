#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
15A_merge_weather_data_submission.py

Merge submission_intervals_v3.csv with weather_data_submission.csv by idx and compute:
- track in radians (overwrite 'track' column)
- delta_t_isa (°C) from 'altitude_m' and 'temperature'
- headwind, crosswind ∈ [-1, 1] from relative wind/track angle (wind is FROM-direction)

Defaults (override via CLI flags):
  in:  data/processed/submission_intervals_v3.csv
       data/processed/weather_data_submission.csv
  out: data/processed/submission_intervals_v4.csv
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

TAU = 2.0 * np.pi
PI  = np.pi

def deg2rad(x: pd.Series) -> pd.Series:
    return np.deg2rad(pd.to_numeric(x, errors="coerce"))

def wrap_rad_minuspi_pi(x: pd.Series | np.ndarray) -> pd.Series:
    """Wrap angle to (-π, π]."""
    return (x + PI) % TAU - PI

def isa_temperature_c_from_alt_m(alt_m: pd.Series) -> pd.Series:
    """
    ISA temperature (°C) from geometric altitude (m).
    - Troposphere lapse: -6.5 K/km up to 11,000 m.
    - Isothermal above: -56.5 °C.
    """
    h = pd.to_numeric(alt_m, errors="coerce").clip(lower=0)
    t_tropo = 15.0 - 0.0065 * np.minimum(h, 11000.0)  # °C
    return np.where(h <= 11000.0, t_tropo, -56.5)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Merge submission intervals with weather and engineer features.")
    p.add_argument("--features-in",
                   default="data/processed/submission_intervals_v3.csv",
                   help="Path to submission_intervals_v3.csv")
    p.add_argument("--weather-in",
                   default="data/processed/weather_data_submission.csv",
                   help="Path to weather_data_submission.csv")
    p.add_argument("--out",
                   default="data/processed/submission_intervals_v4.csv",
                   help="Output CSV path")
    return p

def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    features_path = Path(args.features_in)
    weather_path  = Path(args.weather_in)
    out_path      = Path(args.out)

    # --- load ---
    if not features_path.exists():
        print(f"ERROR: Missing features file: {features_path}", file=sys.stderr); return 1
    if not weather_path.exists():
        print(f"ERROR: Missing weather file: {weather_path}", file=sys.stderr); return 1

    try:
        feats = pd.read_csv(features_path, low_memory=False)
    except Exception as e:
        print(f"ERROR reading {features_path}: {e}", file=sys.stderr); return 1

    try:
        wx = pd.read_csv(weather_path, low_memory=False)
    except Exception as e:
        print(f"ERROR reading {weather_path}: {e}", file=sys.stderr); return 1

    # --- schema checks ---
    req_feature_cols = {"idx", "track"}
    req_weather_cols = {"idx", "wind_direction", "temperature", "altitude_m"}

    missing_f = req_feature_cols - set(feats.columns)
    missing_w = req_weather_cols - set(wx.columns)
    if missing_f:
        print(f"ERROR: features file missing columns: {sorted(missing_f)}", file=sys.stderr); return 1
    if missing_w:
        print(f"ERROR: weather file missing columns: {sorted(missing_w)}", file=sys.stderr); return 1

    # --- type coercion ---
    feats["track"]          = pd.to_numeric(feats["track"], errors="coerce")
    wx["wind_direction"]    = pd.to_numeric(wx["wind_direction"], errors="coerce")
    wx["temperature"]       = pd.to_numeric(wx["temperature"], errors="coerce")
    wx["altitude_m"]        = pd.to_numeric(wx["altitude_m"], errors="coerce")

    # --- merge ---
    merged = pd.merge(feats, wx, on="idx", how="inner", suffixes=("", "_wx"))

    # --- ΔT ISA ---
    t_isa_c = isa_temperature_c_from_alt_m(merged["altitude_m"])
    merged["delta_t_isa"] = merged["temperature"] - t_isa_c  # °C

    # --- angles in radians ---
    # Track: degrees → radians in [0, 2π), overwrite same column name
    track_deg = (merged["track"].astype(float)) % 360.0
    track_rad = deg2rad(track_deg) % TAU
    merged["track"] = track_rad

    # Wind: FROM-direction degrees → TOWARDS (add π), then radians
    wind_from_deg    = (merged["wind_direction"].astype(float)) % 360.0
    wind_from_rad    = deg2rad(wind_from_deg)
    wind_towards_rad = (wind_from_rad + PI) % TAU

    # Relative angle (track vs wind_towards), wrapped to (-π, π]
    delta_rad = wrap_rad_minuspi_pi(track_rad - wind_towards_rad)

    # XGBoost-friendly encodings (components, no circular seam):
    merged["headwind"]  = np.cos(delta_rad)  # +1 tail, -1 head, 0 pure cross
    merged["crosswind"] = np.sin(delta_rad)  # signed left/right

    # Remove categorical wind label if present
    if "wind_cardinal" in merged.columns:
        merged = merged.drop(columns=["wind_cardinal"])

    # --- write ---
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_path, index=False)
    except Exception as e:
        print(f"ERROR writing {out_path}: {e}", file=sys.stderr); return 1

    print(
        f"Done.\n"
        f"  submission rows: {len(feats)}\n"
        f"  weather    rows: {len(wx)}\n"
        f"  merged     rows: {len(merged)}\n"
        f"Wrote: {out_path}"
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
