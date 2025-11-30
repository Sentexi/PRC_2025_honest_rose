#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Merge features_intervals_v3.csv with weather_data.csv by idx and compute:
- track in radians (overwrite 'track' column)
- delta_t_isa (°C) from 'altitude_m' and 'temperature'
- headwind, crosswind ∈ [-1, 1] as cosine/sine of the relative wind angle in radians
  (wind directions are meteorological: FROM; we convert to TOWARDS by adding π)

Why these encodings?
- Radians + sin/cos avoid circular seams and are easiest for tree/linear models.
  headwind = cos(delta_rad) gives +1 tailwind, -1 headwind, 0 pure crosswind.
  crosswind = sin(delta_rad) is signed left/right component.

I/O (defaults)
  in:  data/processed/features_intervals_v3.csv
       data/processed/weather_data.csv
  out: data/processed/features_intervals_v4.csv
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data/processed")
FEATURES_IN = DATA_DIR / "features_intervals_v3.csv"
WEATHER_IN  = DATA_DIR / "weather_data.csv"
OUT_PATH    = DATA_DIR / "features_intervals_v4.csv"

REQ_FEATURE_COLS = {"idx", "track"}
REQ_WEATHER_COLS = {"idx", "wind_direction", "temperature", "altitude_m"}

TAU = 2.0 * np.pi
PI  = np.pi

def deg2rad(x: pd.Series) -> pd.Series:
    return np.deg2rad(x.astype(float))

def wrap_rad_minuspi_pi(x: pd.Series | np.ndarray) -> pd.Series:
    """Wrap angle to (-π, π]."""
    return (x + PI) % TAU - PI

def isa_temperature_c_from_alt_m(alt_m: pd.Series) -> pd.Series:
    """
    ISA temperature (°C) from geometric altitude (m).
    Troposphere lapse  -6.5 K/km to 11,000 m, then isothermal at -56.5 °C.
    """
    h = pd.to_numeric(alt_m, errors="coerce").clip(lower=0)
    t_tropo = 15.0 - 0.0065 * np.minimum(h, 11000.0)  # °C
    # Above 11 km, clamp to -56.5 °C
    return np.where(h <= 11000.0, t_tropo, -56.5)

def main() -> int:
    # --- load ---
    if not FEATURES_IN.exists():
        print(f"ERROR: Missing features file: {FEATURES_IN}", file=sys.stderr); return 1
    if not WEATHER_IN.exists():
        print(f"ERROR: Missing weather file: {WEATHER_IN}", file=sys.stderr); return 1

    try:
        feats = pd.read_csv(FEATURES_IN, low_memory=False)
    except Exception as e:
        print(f"ERROR reading {FEATURES_IN}: {e}", file=sys.stderr); return 1

    try:
        wx = pd.read_csv(WEATHER_IN, low_memory=False)
    except Exception as e:
        print(f"ERROR reading {WEATHER_IN}: {e}", file=sys.stderr); return 1

    # --- schema checks ---
    missing_f = REQ_FEATURE_COLS - set(feats.columns)
    missing_w = REQ_WEATHER_COLS - set(wx.columns)
    if missing_f:
        print(f"ERROR: features file missing columns: {sorted(missing_f)}", file=sys.stderr); return 1
    if missing_w:
        print(f"ERROR: weather file missing columns: {sorted(missing_w)}", file=sys.stderr); return 1

    # Force numeric types (coerce bad to NaN)
    feats["track"] = pd.to_numeric(feats["track"], errors="coerce")
    wx["wind_direction"] = pd.to_numeric(wx["wind_direction"], errors="coerce")
    wx["temperature"]    = pd.to_numeric(wx["temperature"], errors="coerce")
    wx["altitude_m"]     = pd.to_numeric(wx["altitude_m"], errors="coerce")

    # --- merge ---
    merged = pd.merge(feats, wx, on="idx", how="inner", suffixes=("", "_wx"))

    # --- compute ΔT ISA ---
    t_isa_c = isa_temperature_c_from_alt_m(merged["altitude_m"])
    merged["delta_t_isa"] = merged["temperature"] - t_isa_c  # °C

    # --- angles in radians ---
    # Keep a copy of track in degrees for internal math, then overwrite with radians
    track_deg = merged["track"].astype(float) % 360.0
    track_rad = deg2rad(track_deg)
    merged["track"] = track_rad  # overwrite: now radians in [0, 2π)

    # Wind: FROM-direction (deg) → TOWARDS (rad)
    wind_from_deg = merged["wind_direction"].astype(float) % 360.0
    wind_from_rad = deg2rad(wind_from_deg)
    wind_towards_rad = (wind_from_rad + PI) % TAU

    # Relative angle (track vs wind_towards), wrapped to (-π, π]
    delta_rad = wrap_rad_minuspi_pi(track_rad - wind_towards_rad)

    # XGBoost-friendly components (unitless; encode angle with sin/cos)
    merged["headwind"]  = np.cos(delta_rad)  # +1 tail, -1 head, 0 cross
    merged["crosswind"] = np.sin(delta_rad)  # signed left/right component

    # --- tidy up ---
    # Drop categorical wind text if present; user requested replacement by numeric features
    if "wind_cardinal" in merged.columns:
        merged = merged.drop(columns=["wind_cardinal"])

    # Optional: also drop any legacy 'wind_angle' if it exists
    if "wind_angle" in merged.columns:
        merged = merged.drop(columns=["wind_angle"])

    # --- save ---
    try:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(OUT_PATH, index=False)
    except Exception as e:
        print(f"ERROR writing {OUT_PATH}: {e}", file=sys.stderr); return 1

    print(
        f"Done.\n"
        f"  features rows: {len(feats)}\n"
        f"  weather  rows: {len(wx)}\n"
        f"  merged   rows: {len(merged)}\n"
        f"Wrote: {OUT_PATH}"
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
