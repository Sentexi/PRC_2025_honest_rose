# scripts/02_extract_interval_points.py
# Usage:
#   python scripts/02_extract_interval_points.py \
#       --root prc-2025-datasets \
#       --features data/processed/features_intervals.csv \
#       --outdir data/processed/Intervals \
#       --preview 3

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import sys
from math import radians, sin, cos, asin, sqrt, atan2

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # tqdm optional
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

# ------------------ helpers ------------------

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between (lat1, lon1) and (lat2, lon2) in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2.0)**2 + cos(lat1) * cos(lat2) * sin(dlon/2.0)**2
    c = 2 * asin(min(1.0, sqrt(a)))
    return R * c

def _num(v):
    """Safe numeric cast to float with NaN for non-finites."""
    try:
        f = float(v)
        return f if np.isfinite(f) else np.nan
    except Exception:
        return np.nan

def _cmean_deg(a, b):
    """Circular mean of two headings in degrees. Falls back to the finite one."""
    a = _num(a); b = _num(b)
    if np.isfinite(a) and np.isfinite(b):
        ar = radians(a); br = radians(b)
        x = cos(ar) + cos(br)
        y = sin(ar) + sin(br)
        if x == 0 and y == 0:
            return np.nan
        ang = np.degrees(atan2(y, x))
        return ang % 360.0
    return a if np.isfinite(a) else (b if np.isfinite(b) else np.nan)

def _first_existing(cols, options):
    """Return the first column in 'options' that exists in cols."""
    for name in options:
        if name in cols:
            return name
    return None

def _ci_exact(cols, name):
    """Case-insensitive exact-name match; returns actual column or None."""
    name_l = name.lower()
    for c in cols:
        if c.lower() == name_l:
            return c
    return None

def _find_signal(cols, canonical, synonyms):
    """Return the column name for a signal (case-insensitive, tries synonyms)."""
    c = _ci_exact(cols, canonical)
    if c:
        return c
    for s in synonyms:
        c = _ci_exact(cols, s)
        if c:
            return c
    return None

def _choose_anchors_with_bounds(fdf: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp):
    """
    Boundary-aware anchor selection.

    Returns:
      mode: 'inside' (normal), 'oob_before', 'oob_after'
      left, right: Series rows (original index) for anchors when mode == 'inside'.
                   Otherwise (oob_*), returns (None, None).
    """
    if fdf.empty:
        return 'inside', None, None

    f = fdf.sort_values("timestamp")
    first_ts = f["timestamp"].iloc[0]
    last_ts  = f["timestamp"].iloc[-1]

    # Both endpoints before first
    if t1 < first_ts:
        return 'oob_before', None, None
    # Both endpoints after last
    if t0 > last_ts:
        return 'oob_after', None, None

    # Left anchor (start side)
    if t0 < first_ts:
        left = f.iloc[0]
    else:
        left_df = f.loc[f["timestamp"] <= t0]
        left = (left_df.tail(1) if not left_df.empty else f.head(1)).iloc[0]

    # Right anchor (end side)
    if t1 > last_ts:
        right = f.iloc[-1]
    else:
        right_df = f.loc[f["timestamp"] >= t1]
        right = (right_df.head(1) if not right_df.empty else f.tail(1)).iloc[0]

    # Ensure chronological order for interpolation
    if left["timestamp"] <= right["timestamp"]:
        L, R = left, right
    else:
        L, R = right, left

    return 'inside', fdf.loc[L.name], fdf.loc[R.name]

# ------------------ I/O ------------------

def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["start", "end"]:
        df[c] = pd.to_datetime(df[c], errors="coerce", utc=False)
    required = {"idx", "flight_id", "start", "end", "avg_speed_km_per_min"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in features CSV: {missing}")
    df["idx"] = pd.to_numeric(df["idx"], errors="coerce").astype("Int64")
    df["avg_speed_km_per_min"] = pd.to_numeric(df["avg_speed_km_per_min"], errors="coerce")
    return df

def load_flight_parquet(root: Path, flight_id: str) -> pd.DataFrame:
    fpath = root / "flights_train" / f"{flight_id}.parquet"
    if not fpath.exists():
        raise FileNotFoundError(f"Parquet not found for flight_id '{flight_id}': {fpath}")
    df = pd.read_parquet(fpath)  # engine auto (pyarrow/fastparquet)
    if "timestamp" not in df.columns:
        raise KeyError(f"'timestamp' column missing in {fpath}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)
    return df

# ------------------ main ------------------

def main():
    ap = argparse.ArgumentParser(description="Extract trajectory points per fuel interval into CSV files.")
    ap.add_argument("--root", default="prc-2025-datasets", help="Dataset root containing flights_train/")
    ap.add_argument("--features", default=r"data/processed/features_intervals.csv", help="Input features_intervals.csv")
    ap.add_argument("--outdir", default=r"data/processed/Intervals", help="Output directory for [idx]_flight_data.csv")
    ap.add_argument("--preview", type=int, default=0, help="Show first N rows of each extracted subset (0=off)")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N intervals AFTER skipping existing (0=all)")
    ap.add_argument("--overwrite", action="store_true", help="If set, re-create files even if they already exist")
    args = ap.parse_args()

    root = Path(args.root)
    feats_path = Path(args.features)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        feats = load_features(feats_path)
    except Exception as e:
        print(f"[!] Failed to load features: {e}", file=sys.stderr)
        sys.exit(1)

    # Skip output files that already exist
    if not args.overwrite:
        def exists_for_idx(i):
            try:
                return (outdir / f"{int(i)}_flight_data.csv").exists()
            except Exception:
                return False
        mask_exists = feats["idx"].apply(exists_for_idx)
        skipped_existing = int(mask_exists.sum())
        feats = feats.loc[~mask_exists].copy()
    else:
        skipped_existing = 0

    feats = feats.sort_values(["flight_id", "start"])
    if args.limit and args.limit > 0:
        feats = feats.head(args.limit).copy()

    total_to_process = len(feats)
    if total_to_process == 0:
        print(f"Nothing to do. Existing files skipped: {skipped_existing}.")
        print(f"Output dir: {outdir}")
        return

    flights_cache = {}
    total_saved = 0
    errors = 0

    with tqdm(total=total_to_process, desc="Extracting intervals", unit="interval") as pbar:
        for flight_id, g in feats.groupby("flight_id", sort=False):

            try:
                if flight_id not in flights_cache:
                    flights_cache[flight_id] = load_flight_parquet(root, flight_id)
                fdf = flights_cache[flight_id]
            except Exception as e:
                errors += len(g)
                pbar.update(len(g))
                print(f"[!] Error loading parquet for flight_id={flight_id}: {e}", file=sys.stderr)
                continue

            # Column discovery on this flight parquet
            lat_col = _first_existing(fdf.columns, ["latitude", "lat"])
            lon_col = _first_existing(fdf.columns, ["longitude", "lon"])
            alt_col = _first_existing(fdf.columns, ["altitude", "alt", "baro_altitude"])
            have_latlon = (lat_col is not None) and (lon_col is not None)

            # Ensure numeric dtype for interpolation (handles decimals/strings)
            for c in [lat_col, lon_col, alt_col]:
                if c is not None and c in fdf.columns:
                    fdf[c] = pd.to_numeric(fdf[c], errors="coerce")

            # Extra signals (case-insensitive)
            gs_col   = _find_signal(fdf.columns, "groundspeed",
                                    ["gs", "ground_speed", "ground_speed_kts", "groundspeed_kts", "groundspeed_kmh"])
            trk_col  = _find_signal(fdf.columns, "track", ["heading", "true_track", "course"])
            vr_col   = _find_signal(fdf.columns, "vertical_rate", ["verticalSpeed", "vertical_speed", "roc_fpm", "roc_mps"])
            mach_col = _find_signal(fdf.columns, "mach", ["mach_number"])
            tas_col  = _find_signal(fdf.columns, "TAS", ["tas", "true_airspeed", "true_airspeed_kts", "tas_kts"])
            cas_col  = _find_signal(fdf.columns, "CAS", ["cas", "calibrated_airspeed", "cas_kts"])
            src_col  = _find_signal(fdf.columns, "source", ["data_source", "src"])

            typecode_col = _find_signal(fdf.columns, "typecode",
                                        ["aircraft_type", "icao_type", "aircraft_icao", "type"])
            # features fallback for typecode
            feats_type_fallback = None
            if "aircraft_type" in g.columns:
                feats_type_fallback = "aircraft_type"
            elif "typecode" in g.columns:
                feats_type_fallback = "typecode"

            for _, row in g.iterrows():
                idx = int(row["idx"])
                t0, t1 = row["start"], row["end"]
                tmid = t0 + (t1 - t0) / 2
                out_path = outdir / f"{idx}_flight_data.csv"

                if out_path.exists() and not args.overwrite:
                    pbar.update(1)
                    continue

                try:
                    # Fast path: actual samples inside interval
                    mask = (fdf["timestamp"] >= t0) & (fdf["timestamp"] <= t1)
                    sub = fdf.loc[mask].copy()

                    if len(sub) > 0:
                        if "flight_id" not in sub.columns:
                            sub["flight_id"] = flight_id
                        if "typecode" not in sub.columns:
                            if typecode_col:
                                sub["typecode"] = fdf.loc[sub.index, typecode_col]
                            else:
                                sub["typecode"] = row[feats_type_fallback] if feats_type_fallback else np.nan
                        sub.to_csv(out_path, index=False)
                        total_saved += 1

                        if args.preview > 0:
                            print(f"\n[{flight_id}] idx={idx}  rows={len(sub)}  -> {out_path}")
                            with pd.option_context("display.width", 200, "display.max_columns", 20):
                                print(sub.head(args.preview))
                    else:
                        # Boundary-aware anchor selection
                        mode, left, right = _choose_anchors_with_bounds(fdf, t0, t1)
                        
                        # Enforce “anchors must have complete signals” only for fallback interpolation
                        if mode == "inside":
                            # Columns that must be present & non-null on anchors
                            required_cols = [c for c in [lat_col, lon_col, alt_col, gs_col, trk_col, vr_col] if c]
                            if required_cols:
                                # Keep only rows where all required columns are non-NaN
                                f_valid = fdf.dropna(subset=required_cols).sort_values("timestamp")
                                if not f_valid.empty:
                                    # Re-pick LEFT (<= t0) and RIGHT (>= t1) from valid rows
                                    left_df  = f_valid.loc[f_valid["timestamp"] <= t0]
                                    right_df = f_valid.loc[f_valid["timestamp"] >= t1]

                                    if not left_df.empty:
                                        left = left_df.tail(1).iloc[0]
                                    else:
                                        # no valid row on the left side — fallback to earliest valid
                                        left = f_valid.head(1).iloc[0]

                                    if not right_df.empty:
                                        right = right_df.head(1).iloc[0]
                                    else:
                                        # no valid row on the right side — fallback to latest valid
                                        right = f_valid.tail(1).iloc[0]

                        # Columns to ensure in output
                        cols = list(fdf.columns)
                        for essential in [
                            "timestamp", "latitude", "longitude", "altitude",
                            "flight_id", "typecode",
                            "groundspeed", "track", "vertical_rate", "mach", "TAS", "CAS", "source"
                        ]:
                            if essential not in cols:
                                cols.append(essential)

                        # Out-of-bounds on both sides -> 3 fixed rows using closest datapoint
                        if mode in ("oob_before", "oob_after"):
                            f_sorted = fdf.sort_values("timestamp")
                            closest = f_sorted.iloc[0] if mode == "oob_before" else f_sorted.iloc[-1]

                            # Resolve typecode
                            typecode_val = None
                            if typecode_col:
                                typecode_val = closest[typecode_col] if typecode_col in closest.index else np.nan
                            if (typecode_val is None) or (isinstance(typecode_val, float) and np.isnan(typecode_val)):
                                typecode_val = row[feats_type_fallback] if feats_type_fallback else np.nan

                            # Prepare row builder
                            def oob_row(ts):
                                d = {c: np.nan for c in cols}
                                d["timestamp"] = ts
                                d["flight_id"] = flight_id
                                d["typecode"]  = typecode_val

                                # Lat/Lon from closest if available
                                if have_latlon:
                                    lat_v = _num(closest[lat_col]); lon_v = _num(closest[lon_col])
                                    d[lat_col] = lat_v; d[lon_col] = lon_v
                                    if lat_col != "latitude":
                                        d["latitude"] = lat_v
                                    if lon_col != "longitude":
                                        d["longitude"] = lon_v
                                else:
                                    d["latitude"] = np.nan
                                    d["longitude"] = np.nan

                                # Altitude = 0 (also mirror into alt_col if different)
                                d["altitude"] = 0.0
                                if alt_col and alt_col != "altitude":
                                    d[alt_col] = 0.0

                                # Track from closest if present; otherwise NaN
                                trk_closest = _num(closest[trk_col]) if trk_col else np.nan
                                d["track"] = trk_closest

                                # Zero signals
                                d["groundspeed"]   = 0.0
                                d["vertical_rate"] = 0.0
                                d["mach"]          = 0.0
                                d["TAS"]           = 0.0
                                d["CAS"]           = 0.0

                                # Source marker
                                d["source"] = "interpolation"
                                return d

                            times = [t0, tmid, t1]
                            out = pd.DataFrame([oob_row(ts) for ts in times], columns=cols)
                            out.to_csv(out_path, index=False)
                            total_saved += 1

                            if args.preview > 0:
                                print(f"\n[{flight_id}] idx={idx}  OOB->{mode}  -> {out_path}")
                                with pd.option_context("display.width", 200, "display.max_columns", 20):
                                    print(out.head(args.preview))
                            pbar.update(1)
                            continue  # next interval

                        # --- Normal interpolation path (inside range, side-constrained anchors) ---
                        # Column discovery for geometry again (already done), compute endpoints
                        if have_latlon:
                            latL, lonL = _num(left[lat_col]), _num(right[lat_col])  # careful: will fix below
                            # Oops: need latR/lonR properly
                            latL, lonL = _num(left[lat_col]), _num(left[lon_col])
                            latR, lonR = _num(right[lat_col]), _num(right[lon_col])
                            seg_km = haversine_km(latL, lonL, latR, lonR)
                        else:
                            latL = lonL = latR = lonR = np.nan
                            seg_km = 0.0

                        if (alt_col is not None) and pd.notna(left[alt_col]) and pd.notna(right[alt_col]):
                            altL = _num(left[alt_col]); altR = _num(right[alt_col])
                        else:
                            altL = altR = np.nan

                        # Speed constraint from features; fallback to segment-average speed
                        minutes_in_interval = max((t1 - t0).total_seconds() / 60.0, 0.0)
                        v_km_per_min = _num(row.get("avg_speed_km_per_min", np.nan))
                        if not np.isfinite(v_km_per_min) or v_km_per_min < 0:
                            dt_seg_min = max((right["timestamp"] - left["timestamp"]).total_seconds() / 60.0, 1e-9)
                            v_km_per_min = seg_km / dt_seg_min if dt_seg_min > 0 else 0.0

                        if seg_km <= 0:
                            s_start = s_mid = s_end = 0.0
                        else:
                            # Distance dictated by v * Δt, centered around the time-projected midpoint
                            target_d = max(0.0, v_km_per_min) * minutes_in_interval
                            d = min(max(target_d, 0.0), seg_km)

                            tb, ta = left["timestamp"], right["timestamp"]
                            if pd.isna(tb) or pd.isna(ta) or tb == ta:
                                s_mid = 0.5
                            else:
                                s_mid = float((tmid - tb) / (ta - tb))
                                s_mid = min(max(s_mid, 0.0), 1.0)

                            half_frac = (d / seg_km) / 2.0
                            s_start = s_mid - half_frac
                            s_end   = s_mid + half_frac

                            # Clamp to [0, 1] with minimal re-centering
                            if s_start < 0.0:
                                shift = -s_start
                                s_start = 0.0
                                s_end = min(1.0, s_end + shift)
                            if s_end > 1.0:
                                shift = s_end - 1.0
                                s_end = 1.0
                                s_start = max(0.0, s_start - shift)
                            s_mid = 0.5 * (s_start + s_end)

                        def interp_lin(a, b, s):
                            a = _num(a); b = _num(b)
                            if np.isfinite(a) and np.isfinite(b):
                                return a + (b - a) * float(s)
                            return a if np.isfinite(a) else (b if np.isfinite(b) else np.nan)

                        def val(rowp, col):
                            if col is None:
                                return np.nan
                            return rowp[col] if col in rowp.index else np.nan

                        # Endpoint signals
                        gs_L   = _num(val(left,  gs_col));   gs_R   = _num(val(right, gs_col))
                        trk_L  = _num(val(left,  trk_col));  trk_R  = _num(val(right, trk_col))
                        vr_L   = _num(val(left,  vr_col));   vr_R   = _num(val(right, vr_col))
                        mach_L = _num(val(left,  mach_col)); mach_R = _num(val(right, mach_col))
                        tas_L  = _num(val(left,  tas_col));  tas_R  = _num(val(right, tas_col))
                        cas_L  = _num(val(left,  cas_col));  cas_R  = _num(val(right, cas_col))

                        # Source constant from left/right if present (kept same)
                        src_val = val(left, src_col)
                        if (isinstance(src_val, float) and np.isnan(src_val)) or src_val is None:
                            src_val = val(right, src_col)

                        # Typecode resolution
                        typecode_val = None
                        if typecode_col:
                            typecode_val = val(left, typecode_col)
                            if (isinstance(typecode_val, float) and np.isnan(typecode_val)) or typecode_val is None:
                                typecode_val = val(right, typecode_col)
                        if (typecode_val is None) or (isinstance(typecode_val, float) and np.isnan(typecode_val)):
                            typecode_val = row[feats_type_fallback] if feats_type_fallback else np.nan

                        # Build 3 rows -> list -> DataFrame (avoids concat warning)
                        times = [t0, tmid, t1]
                        ss = [s_start, s_mid, s_end]
                        rows_out = []

                        for ts, s in zip(times, ss):
                            d = {c: np.nan for c in cols}
                            d["timestamp"] = ts
                            d["flight_id"] = flight_id
                            d["typecode"]  = typecode_val

                            # Position
                            if have_latlon:
                                lat_v = interp_lin(latL, latR, s)
                                lon_v = interp_lin(lonL, lonR, s)
                                d[lat_col] = lat_v
                                d[lon_col] = lon_v
                                if lat_col != "latitude":
                                    d["latitude"] = lat_v
                                if lon_col != "longitude":
                                    d["longitude"] = lon_v
                            else:
                                d["latitude"] = np.nan
                                d["longitude"] = np.nan

                            # Altitude
                            if (alt_col is not None) and np.isfinite(altL) and np.isfinite(altR):
                                alt_v = interp_lin(altL, altR, s)
                                d[alt_col] = alt_v
                                if alt_col != "altitude":
                                    d["altitude"] = alt_v
                            else:
                                d["altitude"] = np.nan

                            # Signals: start=left, end=right, mid=averages (track circular)
                            if ts == t0:
                                d["groundspeed"]   = gs_L
                                d["track"]         = trk_L
                                d["vertical_rate"] = vr_L
                                d["mach"]          = mach_L
                                d["TAS"]           = tas_L
                                d["CAS"]           = cas_L
                                d["source"]        = src_val
                            elif ts == t1:
                                d["groundspeed"]   = gs_R
                                d["track"]         = trk_R
                                d["vertical_rate"] = vr_R
                                d["mach"]          = mach_R
                                d["TAS"]           = tas_R
                                d["CAS"]           = cas_R
                                d["source"]        = src_val
                            else:
                                d["groundspeed"]   = np.nanmean([gs_L, gs_R])
                                d["track"]         = _cmean_deg(trk_L, trk_R)
                                d["vertical_rate"] = np.nanmean([vr_L, vr_R])
                                d["mach"]          = np.nanmean([mach_L, mach_R])
                                d["TAS"]           = np.nanmean([tas_L, tas_R])
                                d["CAS"]           = np.nanmean([cas_L, cas_R])
                                d["source"]        = src_val

                            rows_out.append(d)

                        out = pd.DataFrame(rows_out, columns=cols)
                        out.to_csv(out_path, index=False)
                        total_saved += 1

                        if args.preview > 0:
                            print(f"\n[{flight_id}] idx={idx}  rows=0 -> interpolated 3 rows  -> {out_path}")
                            with pd.option_context("display.width", 200, "display.max_columns", 20):
                                print(out.head(args.preview))

                except Exception as e:
                    errors += 1
                    print(f"[!] Error processing idx={idx} (flight_id={flight_id}): {e}", file=sys.stderr)
                finally:
                    pbar.update(1)

    print(f"\nDone. Intervals written: {total_saved}. Skipped existing: {skipped_existing}. Errors: {errors}.")
    print(f"Output dir: {outdir}")

if __name__ == "__main__":
    main()
