#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_extract_interval_points.py

Extracts per-interval flight points:
- Standard path: use real samples inside [start, end] if there is at least one
  "complete" row (lat, lon, alt, groundspeed, track, vertical_rate all present).
- Fallback (inside flight span, but no complete interior rows): choose boundary
  anchors from the nearest rows that DO have all six fields and synthesize exactly
  three rows (start/mid/end) using interpolation rules:
    lat/lon/alt -> linear; track -> circular mean (mid); groundspeed/vertical_rate -> arithmetic mean (mid).
- Outside flight span: 3 rows at start/mid/end from nearest valid row; set alt=0, speeds/rates=0.

Usage:
  python 02_extract_interval_points.py \
      --root prc-2025-datasets \
      --features data/processed/features_intervals.csv \
      --outdir data/processed/Intervals \
      [--limit 100] [--preview] [--overwrite]
"""

import argparse
import os
from pathlib import Path
import math
from typing import List, Optional, Tuple
import re
import numpy as np
import pandas as pd
from tqdm import tqdm


# -----------------------------
# Helpers
# -----------------------------
LAT_SYNS = ["latitude", "lat"]
LON_SYNS = ["longitude", "lon", "long"]
ALT_SYNS = ["altitude", "alt", "baro_altitude", "geoaltitude"]
GS_SYNS  = ["groundspeed", "gs"]
TRK_SYNS = ["track", "heading", "true_track"]
VR_SYNS  = ["vertical_rate", "vertrate", "baro_rate", "roc", "rocd"]
TYPE_SYNS = ["typecode", "icao_typecode", "aircraft_type", "type"]
MACH_SYNS = ["mach"]
TAS_SYNS = ["TAS", "tas"]
CAS_SYNS = ["CAS", "cas"]
SRC_SYNS = ["source"]

REQ_GROUPS = (LAT_SYNS, LON_SYNS, ALT_SYNS, GS_SYNS, TRK_SYNS, VR_SYNS)

OUTPUT_COLS = [
    "timestamp", "flight_id", "typecode",
    "latitude", "longitude", "altitude",
    "groundspeed", "track", "vertical_rate",
    "mach", "TAS", "CAS", "source"
]


def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def first_non_null(df: pd.DataFrame, cols: List[str]) -> Optional[str]:
    """Return the first available column name among cols that has at least one non-null value."""
    for c in cols:
        if c in df.columns and df[c].notna().any():
            return c
    return None


def to_naive_datetime(s: pd.Series) -> pd.Series:
    s2 = pd.to_datetime(s, errors="coerce", utc=True)
    # Drop tz to compare with naive interval times (features are usually naive)
    return s2.dt.tz_convert(None)


def parse_dt(x: str) -> pd.Timestamp:
    s = str(x).strip()
    if not s:
        return pd.NaT

    # ISO 8601-like: YYYY-MM-DD ...
    if re.match(r'^\d{4}-\d{2}-\d{2}', s):
        # Use explicit format to silence the warning
        try:
            return pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        except ValueError:
            try:
                return pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", errors="coerce")
            except ValueError:
                return pd.to_datetime(s, errors="coerce", dayfirst=False)

    # EU style: DD.MM.YYYY ...
    for fmt in ("%d.%m.%Y %H:%M:%S.%f", "%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M"):
        try:
            return pd.to_datetime(s, format=fmt, errors="coerce")
        except ValueError:
            pass

    # Fallback
    return pd.to_datetime(s, errors="coerce")


def angle_circ_mean_deg(a: float, b: float) -> float:
    """Circular mean of two angles in degrees, handling NaNs."""
    if pd.isna(a) and pd.isna(b):
        return np.nan
    if pd.isna(a):
        return b
    if pd.isna(b):
        return a
    a_rad = math.radians(a % 360)
    b_rad = math.radians(b % 360)
    x = (math.cos(a_rad) + math.cos(b_rad)) / 2.0
    y = (math.sin(a_rad) + math.sin(b_rad)) / 2.0
    ang = math.degrees(math.atan2(y, x)) % 360
    return ang


def lin_interp(vL, vR, tL, tR, tq):
    """Linear interpolation for scalar values with time stamps; if tL==tR or value missing, fallback."""
    if pd.isna(vL) and pd.isna(vR):
        return np.nan
    if tL == tR or pd.isna(vL) or pd.isna(vR):
        # If one side is missing OR no span, snap to the available one
        return vL if not pd.isna(vL) else vR
    frac = (tq - tL) / (tR - tL)
    return vL + frac * (vR - vL)


def nearest_row(df: pd.DataFrame, t: pd.Timestamp) -> pd.Series:
    """Return the row with timestamp closest to t."""
    idx = (df["timestamp"] - t).abs().argsort().iloc[0]
    return df.iloc[idx]


def choose_valid_anchors(f_valid: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp) -> Tuple[pd.Series, pd.Series]:
    """Pick left (<= t0) and right (>= t1) from valid rows. If one side is empty, fallback to extreme valid row."""
    left_df  = f_valid.loc[f_valid["timestamp"] <= t0]
    right_df = f_valid.loc[f_valid["timestamp"] >= t1]

    if not left_df.empty:
        left = left_df.tail(1).iloc[0]
    else:
        left = f_valid.head(1).iloc[0]

    if not right_df.empty:
        right = right_df.head(1).iloc[0]
    else:
        right = f_valid.tail(1).iloc[0]

    return left, right


def build_row(timestr: pd.Timestamp,
              flight_id: str,
              typecode_val: Optional[str],
              lat: float, lon: float, alt: float,
              gs: float, trk: float, vr: float,
              mach: float = np.nan, tas: float = np.nan, cas: float = np.nan,
              source: str = None) -> dict:
    return {
        "timestamp": timestr,
        "flight_id": flight_id,
        "typecode": typecode_val,
        "latitude": lat,
        "longitude": lon,
        "altitude": alt,
        "groundspeed": gs,
        "track": trk,
        "vertical_rate": vr,
        "mach": mach,
        "TAS": tas,
        "CAS": cas,
        "source": source,
    }


# -----------------------------
# Core logic
# -----------------------------
def process_interval(row, flight_df_cache, args, outdir: Path):
    idx = int(row["idx"])
    flight_id = str(row["flight_id"])

    # Parse interval bounds
    t0 = parse_dt(row["start"])
    t1 = parse_dt(row["end"])
    if pd.isna(t0) or pd.isna(t1):
        raise ValueError(f"Bad start/end for idx={idx}")

    # Prepare output
    out_path = outdir / f"{idx}_flight_data.csv"
    if out_path.exists() and not args.overwrite:
        return "skipped_exists"

    # Load flight DF (cached)
    if flight_id not in flight_df_cache:
        parquet_path = Path(args.root) / "flights_train" / f"{flight_id}.parquet"
        if not parquet_path.exists():
            return f"missing_parquet:{flight_id}"
        try:
            fdf = pd.read_parquet(parquet_path, engine="pyarrow")
        except Exception:
            # As a fallback try fastparquet if available
            fdf = pd.read_parquet(parquet_path)
        # Ensure timestamp
        if "timestamp" not in fdf.columns:
            # common alternatives
            for alt_ts in ["time", "datetime", "DateTime", "ts"]:
                if alt_ts in fdf.columns:
                    fdf = fdf.rename(columns={alt_ts: "timestamp"})
                    break
        fdf["timestamp"] = to_naive_datetime(fdf["timestamp"])
        fdf = fdf.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        flight_df_cache[flight_id] = fdf
    else:
        fdf = flight_df_cache[flight_id]

    if fdf.empty:
        return f"empty_flight:{flight_id}"

    # Discover columns
    lat_col = pick_col(fdf, LAT_SYNS)
    lon_col = pick_col(fdf, LON_SYNS)
    alt_col = pick_col(fdf, ALT_SYNS)
    gs_col  = pick_col(fdf, GS_SYNS)
    trk_col = pick_col(fdf, TRK_SYNS)
    vr_col  = pick_col(fdf, VR_SYNS)
    type_col = first_non_null(fdf, TYPE_SYNS)
    mach_col = pick_col(fdf, MACH_SYNS)
    tas_col  = pick_col(fdf, TAS_SYNS)
    cas_col  = pick_col(fdf, CAS_SYNS)
    src_col  = pick_col(fdf, SRC_SYNS)

    # Required columns actually present
    required_cols = [c for c in [lat_col, lon_col, alt_col, gs_col, trk_col, vr_col] if c]

    # Interval subset
    sub = fdf[(fdf["timestamp"] >= t0) & (fdf["timestamp"] <= t1)].copy()

    # Option A: Treat as empty if there are NO complete rows inside
    sub_complete = sub.dropna(subset=required_cols) if required_cols else sub

    # Decide overlap
    first_ts = fdf["timestamp"].iloc[0]
    last_ts  = fdf["timestamp"].iloc[-1]
    overlaps = not (t1 < first_ts or t0 > last_ts)

    # If there is at least one complete row inside the window -> standard path
    if not sub_complete.empty:
        # Standard path: keep real samples (as-is)
        out_df = sub.sort_values("timestamp").copy()

        # Ensure required output columns exist / fill from other cols if possible
        # Add flight_id if missing, fill typecode from first non-null across DF
        if "flight_id" not in out_df.columns:
            out_df["flight_id"] = flight_id

        if "typecode" not in out_df.columns:
            type_val = fdf[type_col].dropna().iloc[0] if type_col else np.nan
            out_df["typecode"] = type_val

        # Ensure canonical column names exist (derive/rename if needed)
        rename_map = {}
        if lat_col and lat_col != "latitude": rename_map[lat_col] = "latitude"
        if lon_col and lon_col != "longitude": rename_map[lon_col] = "longitude"
        if alt_col and alt_col != "altitude": rename_map[alt_col] = "altitude"
        if gs_col and gs_col != "groundspeed": rename_map[gs_col] = "groundspeed"
        if trk_col and trk_col != "track": rename_map[trk_col] = "track"
        if vr_col and vr_col != "vertical_rate": rename_map[vr_col] = "vertical_rate"
        if mach_col and mach_col != "mach": rename_map[mach_col] = "mach"
        if tas_col and tas_col != "TAS": rename_map[tas_col] = "TAS"
        if cas_col and cas_col != "CAS": rename_map[cas_col] = "CAS"
        if src_col and src_col != "source": rename_map[src_col] = "source"
        out_df = out_df.rename(columns=rename_map)

        # Restrict columns for output (keep missing as NaN)
        for c in OUTPUT_COLS:
            if c not in out_df.columns:
                out_df[c] = np.nan
        out_df = out_df[OUTPUT_COLS]

        out_df.to_csv(out_path, index=False)
        return "saved_standard"

    # Fallbacks:
    if overlaps:
        # Inside flight span BUT no complete interior rows -> interpolate 3 rows

        # Use only rows with all required columns non-null as anchor candidates
        f_valid = fdf.dropna(subset=required_cols).sort_values("timestamp") if required_cols else fdf

        if f_valid.empty:
            # No row has all six fields — fall back to nearest-anything approach (treat like outside)
            mode = "extrapolation"
            base = nearest_row(fdf.dropna(subset=[c for c in [lat_col, lon_col] if c]), (t0 + (t1 - t0) / 2))
            type_val = (base[type_col] if type_col and pd.notna(base.get(type_col, np.nan)) else
                        (fdf[type_col].dropna().iloc[0] if type_col and fdf[type_col].notna().any() else np.nan))
            rows = []
            for tq in [t0, t0 + (t1 - t0) / 2, t1]:
                rows.append(build_row(
                    tq, flight_id, type_val,
                    float(base.get(lat_col, np.nan)), float(base.get(lon_col, np.nan)), 0.0,
                    0.0, float(base.get(trk_col, np.nan)) if trk_col else np.nan, 0.0,
                    np.nan, np.nan, np.nan, mode
                ))
            out_df = pd.DataFrame(rows, columns=OUTPUT_COLS)
            out_df.to_csv(out_path, index=False)
            return "saved_fallback_no_valid_anchors"

        # Choose anchors around the window
        left, right = choose_valid_anchors(f_valid, t0, t1)

        # Times for 3 points
        tmid = t0 + (t1 - t0) / 2

        # Extract anchor values
        def g(s, col): return s.get(col, np.nan) if col else np.nan

        type_val = (
            g(left, "typecode") if "typecode" in left.index and pd.notna(g(left, "typecode"))
            else (fdf[type_col].dropna().iloc[0] if type_col and fdf[type_col].notna().any() else np.nan)
        )

        # Start values (copy from left)
        lat_s = float(g(left, lat_col));  lon_s = float(g(left, lon_col));  alt_s = float(g(left, alt_col))
        gs_s  = float(g(left, gs_col));   trk_s = float(g(left, trk_col));  vr_s  = float(g(left, vr_col))

        # End values (copy from right)
        lat_e = float(g(right, lat_col)); lon_e = float(g(right, lon_col)); alt_e = float(g(right, alt_col))
        gs_e  = float(g(right, gs_col));  trk_e = float(g(right, trk_col)); vr_e  = float(g(right, vr_col))

        tL = left["timestamp"]; tR = right["timestamp"]

        # Interpolate positions/alt
        lat_m = float(lin_interp(lat_s, lat_e, tL, tR, tmid))
        lon_m = float(lin_interp(lon_s, lon_e, tL, tR, tmid))
        alt_m = float(lin_interp(alt_s, alt_e, tL, tR, tmid))

        # Mid signals
        gs_m  = float(np.nanmean([gs_s, gs_e]))
        vr_m  = float(np.nanmean([vr_s, vr_e]))
        trk_m = float(angle_circ_mean_deg(trk_s, trk_e))

        # Build rows
        rows = [
            build_row(t0,   flight_id, type_val, lat_s, lon_s, alt_s, gs_s, trk_s, vr_s, source="interpolation"),
            build_row(tmid, flight_id, type_val, lat_m, lon_m, alt_m, gs_m, trk_m, vr_m, source="interpolation"),
            build_row(t1,   flight_id, type_val, lat_e, lon_e, alt_e, gs_e, trk_e, vr_e, source="interpolation"),
        ]

        out_df = pd.DataFrame(rows, columns=OUTPUT_COLS)
        out_df.to_csv(out_path, index=False)
        return "saved_fallback_interpolation"

    else:
        # Entire window outside the flight span -> extrapolate 3 rows from nearest valid row
        # Prefer a row with all required fields; if none, fall back to nearest with lat/lon at least.
        base_source = fdf
        if required_cols:
            f_valid_any = fdf.dropna(subset=required_cols)
            base_source = f_valid_any if not f_valid_any.empty else fdf.dropna(subset=[c for c in [lat_col, lon_col] if c])

        if base_source.empty:
            return f"no_position_data:{flight_id}"

        # Choose nearest to the midpoint
        tmid = t0 + (t1 - t0) / 2
        base = nearest_row(base_source, tmid)

        # Compose three rows; position/track from base, alt/speeds/rates = 0
        type_val = (base.get("typecode", np.nan) if "typecode" in base.index and pd.notna(base.get("typecode"))
                    else (fdf[type_col].dropna().iloc[0] if type_col and fdf[type_col].notna().any() else np.nan))

        rows = []
        for tq in [t0, tmid, t1]:
            rows.append(build_row(
                tq, flight_id, type_val,
                float(base.get(lat_col, np.nan)), float(base.get(lon_col, np.nan)), 0.0,
                0.0, float(base.get(trk_col, np.nan)) if trk_col else np.nan, 0.0,
                np.nan, np.nan, np.nan, "extrapolation"
            ))

        out_df = pd.DataFrame(rows, columns=OUTPUT_COLS)
        out_df.to_csv(out_path, index=False)
        return "saved_outside_span"


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="prc-2025-datasets", help="Dataset root with flights_train/")
    ap.add_argument("--features", default="data/processed/features_intervals.csv", help="CSV with idx,flight_id,start,end,...")
    ap.add_argument("--outdir", default="data/processed/Intervals", help="Directory to write [idx]_flight_data.csv")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N intervals")
    ap.add_argument("--preview", action="store_true", help="Do not write, just simulate")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load features file
    feats = pd.read_csv(args.features)
    # Normalize column names if necessary
    rename_map = {}
    for c in feats.columns:
        lc = c.strip().lower()
        if lc == "idx" and c != "idx": rename_map[c] = "idx"
        if lc == "flight_id" and c != "flight_id": rename_map[c] = "flight_id"
        if lc == "start" and c != "start": rename_map[c] = "start"
        if lc == "end" and c != "end": rename_map[c] = "end"
    if rename_map:
        feats = feats.rename(columns=rename_map)

    missing = [c for c in ["idx", "flight_id", "start", "end"] if c not in feats.columns]
    if missing:
        raise ValueError(f"Features CSV is missing required columns: {missing}")

    if args.limit:
        feats = feats.head(args.limit)

    flight_df_cache = {}
    results = {"saved_standard": 0, "saved_fallback_interpolation": 0,
               "saved_fallback_no_valid_anchors": 0, "saved_outside_span": 0,
               "skipped_exists": 0, "errors": 0}

    for _, r in tqdm(feats.iterrows(), total=len(feats), desc="Intervals"):
        try:
            if args.preview:
                # Dry run: just attempt processing without writing (checks paths & logic)
                status = process_interval(r, flight_df_cache, argparse.Namespace(**{**vars(args), "overwrite": True}), outdir)
            else:
                status = process_interval(r, flight_df_cache, args, outdir)
            if status in results:
                results[status] += 1
            else:
                # other statuses (missing parquet, empty flight, etc.)
                results["errors"] += 1
        except Exception as e:
            results["errors"] += 1
            print(f"[idx={int(r['idx'])}] ERROR: {e}")

    print("\nSummary:")
    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
