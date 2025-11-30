#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_extract_interval_points.py

Purpose
-------
Extract raw trajectory points for an interval [start, end] from a flight parquet.

Behavior
--------
1) STANDARD PATH:
   - If >= 3 rows exist inside the interval that have latitude+longitude,
     write ALL of them as-is (other fields may be NaN).

2) FALLBACK PATH (exactly 3 rows: start/mid/end):
   - Pick left (<= start) and right (>= end) anchors that have lat/lon.
   - If anchors are distinct, do linear interpolation for lat/lon/alt/gs/track/vr.
   - **If anchors effectively collapse** (same timestamp, or within ~1 km, or only one
     side has a valid position), use **dead-reckoning** from the single anchor:
       position = propagate using groundspeed (knots) + track (deg)
       altitude  = propagate using vertical rate (fpm)
     Mark rows with source="dead_reckoning".

Output
------
data/processed/Intervals/{idx}_flight_data.csv with columns:
timestamp,flight_id,typecode,latitude,longitude,altitude,groundspeed,track,vertical_rate,mach,TAS,CAS,source
"""

import argparse
import math
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd

# Try pyarrow if available (faster); fall back to pandas.
try:
    import pyarrow.parquet as pq
except Exception:
    pq = None

# -------------------------------
# Config
# -------------------------------
FEATURES_CSV_DEFAULT = "data/processed/features_intervals.csv"
FLIGHTS_DIR_DEFAULT  = "prc-2025-datasets/flights_train"
OUT_DIR_DEFAULT      = "data/processed/Intervals"

# Output schema
OUTPUT_COLS = [
    "timestamp", "flight_id", "typecode",
    "latitude", "longitude", "altitude",
    "groundspeed", "track", "vertical_rate",
    "mach", "TAS", "CAS", "source"
]

# Parquet column names
TS_COL   = "timestamp"
LAT_COL  = "latitude"
LON_COL  = "longitude"
ALT_COL  = "altitude"
GS_COL   = "groundspeed"
TRK_COL  = "track"
VR_COL   = "vertical_rate"
TYPE_COL = "typecode"
SRC_COL  = "source"

# Geo/tolerance constants
EARTH_R_M   = 6371000.0
KNOT_TO_MPS = 0.514444
DIST_EPS_M  = 1000.0   # treat anchors closer than 1 km as "same place"

# Debug switch
DEBUG = False  # set True to print why a branch is chosen


# -------------------------------
# Time helpers
# -------------------------------
def ensure_dt_naive_utc(ts: pd.Timestamp) -> pd.Timestamp:
    """Convert tz-aware to UTC then drop tz; leave naive as-is."""
    if ts.tz is not None:
        return ts.tz_convert("UTC").tz_localize(None)
    return ts


def parse_timestamp_any(s) -> pd.Timestamp:
    """Parse many timestamp formats; normalize to naive UTC."""
    if isinstance(s, pd.Timestamp):
        return ensure_dt_naive_utc(s)

    ts = pd.to_datetime(s, errors="coerce", utc=True)
    if pd.notna(ts):
        return ensure_dt_naive_utc(ts)

    ts = pd.to_datetime(s, errors="coerce", utc=True, dayfirst=True, infer_datetime_format=True)
    if pd.notna(ts):
        return ensure_dt_naive_utc(ts)

    ts = pd.to_datetime(s, errors="raise")
    return ensure_dt_naive_utc(ts)


# -------------------------------
# IO helpers
# -------------------------------
def read_features_row(features_csv: str, idx: int) -> pd.Series:
    df = pd.read_csv(features_csv)

    # Find idx column
    if "idx" in df.columns:
        idx_col = "idx"
    elif "id" in df.columns:
        idx_col = "id"
    elif "ID" in df.columns:
        idx_col = "ID"
    else:
        idx_col = df.columns[0]

    row = df.loc[df[idx_col] == idx]
    if row.empty:
        raise ValueError(f"idx={idx} not found in {features_csv}")
    row = row.iloc[0]

    # Find start/end columns
    start_col, end_col = None, None
    for cand in ["start", "interval_start", "start_time"]:
        if cand in df.columns: start_col = cand; break
    for cand in ["end", "interval_end", "end_time"]:
        if cand in df.columns: end_col = cand; break
    if start_col is None or end_col is None:
        for c in df.columns:
            cl = c.lower()
            if start_col is None and "start" in cl: start_col = c
            if end_col is None and "end" in cl:     end_col   = c
    if start_col is None or end_col is None:
        raise ValueError("Could not locate start/end columns in features CSV.")

    if "flight_id" not in df.columns:
        raise ValueError("Column 'flight_id' not found in features CSV.")

    return pd.Series({
        "idx": idx,
        "flight_id": str(row["flight_id"]),
        "start": parse_timestamp_any(row[start_col]),
        "end":   parse_timestamp_any(row[end_col]),
    })


def load_flight_parquet(fpath: str, columns: List[str]) -> pd.DataFrame:
    if pq is None:
        return pd.read_parquet(fpath, columns=[c for c in columns if c is not None])
    table = pq.read_table(fpath, columns=[c for c in columns if c is not None])
    return table.to_pandas(types_mapper=lambda dt: dt)


def save_csv(df_out: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)


# -------------------------------
# Math / geo helpers
# -------------------------------
def gc_distance_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters."""
    if not all(np.isfinite([lat1, lon1, lat2, lon2])):
        return np.inf
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = φ2 - φ1
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return 2 * EARTH_R_M * math.asin(math.sqrt(a))


def geo_dest(lat_deg: float, lon_deg: float, bearing_deg: float, dist_m: float):
    """Destination point from (lat, lon), bearing (deg), distance (m)."""
    φ1 = math.radians(lat_deg); λ1 = math.radians(lon_deg)
    θ = math.radians((bearing_deg or 0.0) % 360.0)
    δ = dist_m / EARTH_R_M
    sinφ1, cosφ1 = math.sin(φ1), math.cos(φ1)
    sinδ,  cosδ  = math.sin(δ),  math.cos(δ)
    sinφ2 = sinφ1 * cosδ + cosφ1 * sinδ * math.cos(θ)
    φ2 = math.asin(sinφ2)
    λ2 = λ1 + math.atan2(math.sin(θ) * sinδ * cosφ1, cosδ - sinφ1 * sinφ2)
    lat2 = math.degrees(φ2)
    lon2 = ((math.degrees(λ2) + 540.0) % 360.0) - 180.0
    return lat2, lon2


def dead_reckon(lat0, lon0, alt0, gs_kn, track_deg, vr_fpm, t_anchor, tq):
    """Propagate from a single anchor using GS/track; VR (fpm) adjusts altitude."""
    dt = (tq - t_anchor).total_seconds()  # seconds
    if not all(np.isfinite(x) for x in [lat0, lon0]) or not np.isfinite(gs_kn) or not np.isfinite(track_deg):
        return float(lat0), float(lon0), float(alt0)
    d_m = abs(dt) * (gs_kn * KNOT_TO_MPS)  # meters
    bearing = (track_deg + (180.0 if dt < 0 else 0.0)) % 360.0
    lat1, lon1 = geo_dest(float(lat0), float(lon0), float(bearing), float(d_m))
    alt1 = float(alt0) + (float(vr_fpm) if np.isfinite(vr_fpm) else 0.0) * (dt / 60.0)
    return lat1, lon1, alt1


def lin_interp(tq: pd.Timestamp, tL: pd.Timestamp, tR: pd.Timestamp, vL: float, vR: float) -> float:
    """Linear interpolation for scalars; if tL==tR or NaNs, prefer vL then vR."""
    if tL == tR or not (np.isfinite(vL) and np.isfinite(vR)):
        return float(vL) if np.isfinite(vL) else float(vR)
    span = (tR - tL).total_seconds()
    w = (tq - tL).total_seconds() / span
    return float(vL) * (1.0 - w) + float(vR) * w


# -------------------------------
# Anchor selection
# -------------------------------
def choose_anchors_latlon(df: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp) -> Tuple[pd.Series, pd.Series]:
    """Pick left (<= t0) and right (>= t1) anchors among rows having lat/lon."""
    valid = df.dropna(subset=[LAT_COL, LON_COL]).sort_values(TS_COL)
    if valid.empty:
        base = {
            TS_COL: t0, LAT_COL: np.nan, LON_COL: np.nan, ALT_COL: np.nan,
            GS_COL: np.nan, TRK_COL: np.nan, VR_COL: np.nan, TYPE_COL: np.nan, SRC_COL: np.nan
        }
        s = pd.Series(base)
        return s, s

    left  = valid[valid[TS_COL] <= t0].tail(1)
    right = valid[valid[TS_COL] >= t1].head(1)
    if left.empty:  left  = valid.head(1)
    if right.empty: right = valid.tail(1)
    return left.iloc[0], right.iloc[0]


# -------------------------------
# Row builder
# -------------------------------
def build_row(tstamp: pd.Timestamp, flight_id: str, typecode, lat, lon, alt, gs, trk, vr,
              mach=np.nan, TAS=np.nan, CAS=np.nan, source="actual") -> dict:
    return {
        "timestamp": tstamp.strftime("%Y-%m-%d %H:%M:%S.%f").rstrip("0").rstrip("."),
        "flight_id": flight_id,
        "typecode": typecode if pd.notna(typecode) else np.nan,
        "latitude": float(lat) if np.isfinite(lat) else np.nan,
        "longitude": float(lon) if np.isfinite(lon) else np.nan,
        "altitude": float(alt) if np.isfinite(alt) else np.nan,
        "groundspeed": float(gs) if np.isfinite(gs) else np.nan,
        "track": float(trk) if np.isfinite(trk) else np.nan,
        "vertical_rate": float(vr) if np.isfinite(vr) else np.nan,
        "mach": float(mach) if np.isfinite(mach) else np.nan,
        "TAS": float(TAS) if np.isfinite(TAS) else np.nan,
        "CAS": float(CAS) if np.isfinite(CAS) else np.nan,
        "source": source
    }


# -------------------------------
# Core logic
# -------------------------------
def process_idx(features_csv: str, flights_dir: str, out_dir: str, idx: int) -> str:
    # 1) features row
    feat = read_features_row(features_csv, idx)
    flight_id = str(feat["flight_id"])
    t0, t1 = feat["start"], feat["end"]
    if t1 < t0:
        t0, t1 = t1, t0  # ensure ordering

    out_path = Path(out_dir) / f"{idx}_flight_data.csv"

    # 2) read flight parquet
    pq_path = Path(flights_dir) / f"{flight_id}.parquet"
    if not pq_path.exists():
        raise FileNotFoundError(f"Parquet not found: {pq_path}")

    cols_needed = [TS_COL, LAT_COL, LON_COL, ALT_COL, GS_COL, TRK_COL, VR_COL, TYPE_COL, SRC_COL]
    fdf = load_flight_parquet(str(pq_path), [c for c in cols_needed if c is not None])

    # Parse timestamps & sort
    fdf[TS_COL] = pd.to_datetime(fdf[TS_COL], errors="coerce", utc=True).dt.tz_convert(None)
    fdf = fdf.sort_values(TS_COL).reset_index(drop=True)

    # 3) STANDARD PATH: take all rows inside [t0, t1] that at least have lat/lon
    sub = fdf[(fdf[TS_COL] >= t0) & (fdf[TS_COL] <= t1)]
    sub = sub.dropna(subset=[LAT_COL, LON_COL])

    if len(sub) >= 3:
        type_val = sub[TYPE_COL].dropna().iloc[0] if TYPE_COL in sub.columns and sub[TYPE_COL].notna().any() else np.nan
        rows = []
        for _, r in sub.iterrows():
            rows.append(
                build_row(
                    r[TS_COL], flight_id, type_val if pd.notna(r.get(TYPE_COL, np.nan)) else type_val,
                    r.get(LAT_COL, np.nan), r.get(LON_COL, np.nan), r.get(ALT_COL, np.nan),
                    r.get(GS_COL, np.nan), r.get(TRK_COL, np.nan), r.get(VR_COL, np.nan),
                    mach=r.get("mach", np.nan), TAS=r.get("TAS", np.nan), CAS=r.get("CAS", np.nan),
                    source=str(r.get(SRC_COL, "actual")) if pd.notna(r.get(SRC_COL, np.nan)) else "actual"
                )
            )
        out_df = pd.DataFrame(rows, columns=OUTPUT_COLS)
        save_csv(out_df, out_path)
        if DEBUG:
            print(f"[SNAPSHOT] idx={idx} wrote {len(out_df)} rows inside window.")
        return f"saved_snapshot_{len(out_df)}"

    # 4) FALLBACK PATH: synthesize exactly 3 rows (start/mid/end)
    tmid = t0 + (t1 - t0) / 2
    left, right = choose_anchors_latlon(fdf, t0, t1)

    # typecode resolution
    type_val = np.nan
    if TYPE_COL in fdf.columns and fdf[TYPE_COL].notna().any():
        type_val = fdf[TYPE_COL].dropna().iloc[0]
    if pd.notna(left.get(TYPE_COL, np.nan)):
        type_val = left.get(TYPE_COL, type_val)

    # --- Coerce to float early (robust against object/Decimal dtypes) ---
    def num(x):
        try:
            return float(x)
        except Exception:
            return np.nan

    latL, lonL, altL = num(left.get(LAT_COL, np.nan)),  num(left.get(LON_COL, np.nan)),  num(left.get(ALT_COL, np.nan))
    gsL,  trkL,  vrL  = num(left.get(GS_COL, np.nan)),   num(left.get(TRK_COL, np.nan)),  num(left.get(VR_COL, np.nan))
    latR, lonR, altR = num(right.get(LAT_COL, np.nan)), num(right.get(LON_COL, np.nan)), num(right.get(ALT_COL, np.nan))
    gsR,  trkR,  vrR  = num(right.get(GS_COL, np.nan)),  num(right.get(TRK_COL, np.nan)), num(right.get(VR_COL, np.nan))
    tL, tR = left.get(TS_COL, t0), right.get(TS_COL, t1)

    left_has_pos  = np.isfinite(latL) and np.isfinite(lonL)
    right_has_pos = np.isfinite(latR) and np.isfinite(lonR)
    dist_lr = gc_distance_m(latL, lonL, latR, lonR) if (left_has_pos and right_has_pos) else np.inf

    anchors_collapse = (
        (pd.notna(tL) and pd.notna(tR) and tL == tR) or
        (left_has_pos and right_has_pos and dist_lr < DIST_EPS_M) or
        (left_has_pos and not right_has_pos) or
        (right_has_pos and not left_has_pos)
    )

    # Final guard: if linear endpoints would still yield practically flat coords, force DR
    FORCE_DR_IF_FLAT = True
    if not anchors_collapse and FORCE_DR_IF_FLAT:
        same_lat = (np.isfinite(latL) and np.isfinite(latR) and abs(latL - latR) < 1e-10)
        same_lon = (np.isfinite(lonL) and np.isfinite(lonR) and abs(lonL - lonR) < 1e-10)
        if same_lat and same_lon:
            anchors_collapse = True
            if DEBUG:
                print("[DR-DEBUG] Forced DR: linear endpoints flat.")

    if DEBUG:
        print(f"[DR-DEBUG] idx={idx} t0={t0} t1={t1}")
        print(f"[DR-DEBUG] tL={tL} tR={tR} left_has_pos={left_has_pos} right_has_pos={right_has_pos} dist_lr={dist_lr:.2f} m collapse={anchors_collapse}")
        print(f"[DR-DEBUG] left(lat,lon,gs,trk,vr)=({latL},{lonL},{gsL},{trkL},{vrL}) right=({latR},{lonR},{gsR},{trkR},{vrR})")

    rows = []

    if anchors_collapse:
        # choose propagation anchor (prefer the one that has a position)
        use_right = (right_has_pos and not left_has_pos)
        a_lat = latR if use_right else latL
        a_lon = lonR if use_right else lonL
        a_alt = altR if use_right else altL
        a_gs  = gsR  if use_right else gsL
        a_trk = trkR if use_right else trkL
        a_vr  = vrR  if use_right else vrL
        a_t   = tR if use_right else tL

        for tq in [t0, tmid, t1]:
            lat_q, lon_q, alt_q = dead_reckon(a_lat, a_lon, a_alt, a_gs, a_trk, a_vr, a_t, tq)
            rows.append(build_row(tq, flight_id, type_val, lat_q, lon_q, alt_q, a_gs, a_trk, a_vr, source="dead_reckoning"))

        out_df = pd.DataFrame(rows, columns=OUTPUT_COLS)
        save_csv(out_df, out_path)
        if DEBUG:
            print(f"[DR] idx={idx} wrote 3 rows via dead_reckoning.")
        return "saved_fallback_dead_reckon"

    # Else: distinct anchors -> linear interpolation
    def interp_row(tq: pd.Timestamp):
        lat_q = lin_interp(tq, tL, tR, latL, latR)
        lon_q = lin_interp(tq, tL, tR, lonL, lonR)
        alt_q = lin_interp(tq, tL, tR, altL, altR)
        gs_q  = lin_interp(tq, tL, tR, gsL,  gsR)
        trk_q = lin_interp(tq, tL, tR, trkL, trkR)
        vr_q  = lin_interp(tq, tL, tR, vrL,  vrR)
        return build_row(tq, flight_id, type_val, lat_q, lon_q, alt_q, gs_q, trk_q, vr_q, source="interpolation")

    for tq in [t0, tmid, t1]:
        rows.append(interp_row(tq))

    out_df = pd.DataFrame(rows, columns=OUTPUT_COLS)
    save_csv(out_df, out_path)
    if DEBUG:
        print(f"[LIN-INT] idx={idx} wrote 3 rows via interpolation.")
    return "saved_fallback_interpolation"


# -------------------------------
# CLI
# -------------------------------
def main():
    ap = argparse.ArgumentParser(description="Extract interval trajectory points (with robust DR fallback).")
    ap.add_argument("--idx", type=int, required=True, help="Interval index to process.")
    ap.add_argument("--features", type=str, default=FEATURES_CSV_DEFAULT, help="Path to features_intervals.csv")
    ap.add_argument("--flights_dir", type=str, default=FLIGHTS_DIR_DEFAULT, help="Directory with flights_train/*.parquet")
    ap.add_argument("--out_dir", type=str, default=OUT_DIR_DEFAULT, help="Output directory for interval CSVs")
    ap.add_argument("--debug", action="store_true", help="Print debug info")
    args = ap.parse_args()

    global DEBUG
    DEBUG = bool(args.debug)

    try:
        status = process_idx(args.features, args.flights_dir, args.out_dir, args.idx)
        print(f"[OK] idx={args.idx} -> {status}")
    except Exception as e:
        print(f"[ERROR] idx={args.idx}: {e}")
        raise


if __name__ == "__main__":
    main()
