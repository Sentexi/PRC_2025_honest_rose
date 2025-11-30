#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
12_integrity_check_outliers.py

Sanity & integrity checks for PRC fuel interval labels. Writes a Markdown report
and CSV extracts (routes & flight_ids most affected, and concrete examples).

Inputs:
- prc-2025-datasets/fuel_train.parquet
  required cols: ['idx','flight_id','start','end','fuel_kg']
- prc-2025-datasets/flightlist_train.parquet
  route cols (any of): origin: ['origin_icao', 'origin']
                       dest:   ['dest_icao', 'destination_icao', 'dest']
  optional: 'aircraft_type'

Outputs (under data/diagnostics/):
- integrity_report.md
- anomalies_rows.csv
- top_routes_by_anomaly.csv
- top_flightids_by_anomaly.csv
- extreme_examples.csv
"""

import argparse
import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd

# ---------- IO helpers ----------
def read_parquet_pyarrow(path, columns=None):
    try:
        return pd.read_parquet(path, engine="pyarrow", columns=columns)
    except Exception as e:
        print(f"ERROR reading parquet: {path}\n{e}", file=sys.stderr)
        sys.exit(1)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# ---------- Parsing helpers ----------
def safe_to_datetime_utc(series):
    """
    Convert a Series to tz-naive UTC datetimes.
    - Parse with utc=True (handles naive & tz-aware).
    - Convert to UTC, drop tzinfo.
    - Unparseables -> NaT.
    """
    s = pd.to_datetime(series, errors="coerce", utc=True)
    # s is tz-aware (UTC) after utc=True; strip tz
    return s.dt.tz_convert("UTC").dt.tz_localize(None)

def coerce_fuel_to_float(series):
    """
    Make sure fuel_kg becomes float, handling strings with German decimal commas.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("float64")
    # convert to string, replace ',' -> '.', then to float
    s = series.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

def pick_first_present(df, candidates):
    """Return first existing column name from `candidates` or None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def pct(n, d):
    return (100.0 * n / d) if d else 0.0

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Integrity checks for fuel interval labels.")
    parser.add_argument("--fuel-parquet", default="prc-2025-datasets/fuel_train.parquet")
    parser.add_argument("--flightlist-parquet", default="prc-2025-datasets/flightlist_train.parquet")
    parser.add_argument("--outdir", default="data/diagnostics")
    # thresholds (tunable)
    parser.add_argument("--min-dt-s", type=float, default=30.0, help="Minimum interval seconds to trust (default: 30s).")
    parser.add_argument("--max-fuel-kg", type=float, default=30000.0, help="Max plausible fuel per interval in kg (default: 30,000).")
    parser.add_argument("--max-rate-kgmin", type=float, default=220.0, help="Max plausible fuel rate kg/min (default: 220).")
    parser.add_argument("--examples", type=int, default=25, help="How many extreme examples to save.")
    parser.add_argument("--simulate-divide-1000", action="store_true",
                        help="Simulate dividing fuel_kg by 1000 on 1000-multiple rows in the report footer.")
    args = parser.parse_args()

    ensure_dir(args.outdir)

    # Load fuel intervals
    df_fuel = read_parquet_pyarrow(args.fuel_parquet)
    required = ["idx", "flight_id", "start", "end", "fuel_kg"]
    missing = [c for c in required if c not in df_fuel.columns]
    if missing:
        print(f"ERROR: Missing columns in {args.fuel_parquet}: {missing}", file=sys.stderr)
        sys.exit(1)

    df = df_fuel.copy()

    # Timestamps & durations
    df["start_dt"] = safe_to_datetime_utc(df["start"])
    df["end_dt"]   = safe_to_datetime_utc(df["end"])
    df["dt_s"] = (df["end_dt"] - df["start_dt"]).dt.total_seconds()
    df["dt_min"] = df["dt_s"] / 60.0

    # Fuel numeric (handle decimal commas if present)
    df["fuel_kg"] = coerce_fuel_to_float(df["fuel_kg"])
    total_rows = len(df)

    # Load flightlist to enrich route
    df_flist = read_parquet_pyarrow(args.flightlist_parquet)

    origin_col = pick_first_present(df_flist, ["origin_icao", "origin"])
    dest_col   = pick_first_present(df_flist, ["dest_icao", "destination_icao", "dest"])
    ac_col     = pick_first_present(df_flist, ["aircraft_type"])

    merge_cols = ["flight_id"]
    if origin_col: merge_cols.append(origin_col)
    if dest_col:   merge_cols.append(dest_col)
    if ac_col:     merge_cols.append(ac_col)

    df_routes = df_flist[merge_cols].drop_duplicates()

    df = df.merge(df_routes, on="flight_id", how="left")

    # Build 'route' string robustly
    if origin_col and dest_col:
        df["route"] = df[origin_col].fillna("?").astype(str) + " - " + df[dest_col].fillna("?").astype(str)
    else:
        df["route"] = "UNKNOWN"

    # Derived: fuel rate
    df["fuel_kg_min"] = df["fuel_kg"] / df["dt_min"]

    # Rules / flags
    rules = {}
    rules["A_dt_missing"]     = df["dt_s"].isna() | df["dt_min"].isna()
    rules["B_dt_negative"]    = df["dt_s"] < 0
    rules["C_dt_too_small"]   = (df["dt_s"] >= 0) & (df["dt_s"] < args.min_dt_s)
    rules["D_dt_zero"]        = df["dt_s"] == 0
    rules["E_fuel_nan"]       = df["fuel_kg"].isna()
    rules["F_fuel_negative"]  = df["fuel_kg"] < 0
    rules["G_fuel_too_big"]   = df["fuel_kg"] > args.max_fuel_kg
    rules["H_rate_nan"]       = df["fuel_kg_min"].isna()
    rules["I_rate_negative"]  = df["fuel_kg_min"] < 0
    rules["J_rate_too_big"]   = df["fuel_kg_min"] > args.max_rate_kgmin
    rules["K_multiple_1000"]  = df["fuel_kg"].notna() & ((df["fuel_kg"] % 1000) == 0) & (df["fuel_kg"] >= 1000)
    rules["L_start_eq_end"]   = df["start_dt"].notna() & df["end_dt"].notna() & (df["start_dt"] == df["end_dt"])

    flag_cols = list(rules.keys())
    for k, v in rules.items():
        df[k] = v

    df["any_flag"] = df[flag_cols].any(axis=1)
    anomalies = df[df["any_flag"]].copy()
    n_anom = len(anomalies)

    # Group helpers
    df["_one"] = 1

    # Route-level aggregation
    route_agg = (
        df.groupby(["route"], dropna=False)
          .agg(
              rows=("_one", "sum"),
              anomalies=("any_flag", "sum"),
              share_anom=("any_flag", "mean"),
              mean_dt_s=("dt_s", "mean"),
              p95_dt_s=("dt_s", lambda s: np.nanpercentile(s, 95)),
              mean_fuel_kg=("fuel_kg", "mean"),
              p95_fuel_kg=("fuel_kg", lambda s: np.nanpercentile(s, 95)),
              k_1000=("K_multiple_1000","sum"),
              dt_zero=("D_dt_zero","sum"),
              dt_small=("C_dt_too_small","sum"),
              too_big=("G_fuel_too_big","sum"),
              rate_too_big=("J_rate_too_big","sum"),
          )
          .reset_index()
          .sort_values(["anomalies","share_anom","rows"], ascending=[False, False, False])
    )

    # FlightID-level aggregation
    fid_agg = (
        df.groupby(["flight_id"], dropna=False)
          .agg(
              rows=("_one","sum"),
              anomalies=("any_flag","sum"),
              share_anom=("any_flag","mean"),
              k_1000=("K_multiple_1000","sum"),
              dt_zero=("D_dt_zero","sum"),
              dt_small=("C_dt_too_small","sum"),
              too_big=("G_fuel_too_big","sum"),
              rate_too_big=("J_rate_too_big","sum"),
          )
          .reset_index()
          .sort_values(["anomalies","share_anom","rows"], ascending=[False, False, False])
    )

    # Extreme examples
    extremes = df.copy()
    extremes["rank_key"] = (
        np.where(extremes["K_multiple_1000"], 2, 0)
        + np.where(extremes["G_fuel_too_big"], 3, 0)
        + np.where(extremes["J_rate_too_big"], 3, 0)
        + np.where(extremes["D_dt_zero"] | extremes["L_start_eq_end"], 2, 0)
    )
    extremes = extremes.sort_values(
        ["rank_key","fuel_kg","fuel_kg_min","dt_s"], ascending=[False, False, False, True]
    ).head(args.examples)

    # Save CSV extracts
    ensure_dir(args.outdir)
    anomalies_out = os.path.join(args.outdir, "anomalies_rows.csv")
    routes_out    = os.path.join(args.outdir, "top_routes_by_anomaly.csv")
    fids_out      = os.path.join(args.outdir, "top_flightids_by_anomaly.csv")
    extremes_out  = os.path.join(args.outdir, "extreme_examples.csv")

    keep_cols = ["idx","flight_id"]
    if origin_col: keep_cols.append(origin_col)
    if dest_col:   keep_cols.append(dest_col)
    keep_cols += ["route","start_dt","end_dt","dt_s","dt_min","fuel_kg","fuel_kg_min"] + flag_cols

    anomalies[keep_cols].to_csv(anomalies_out, index=False)
    route_agg.to_csv(routes_out, index=False)
    fid_agg.to_csv(fids_out, index=False)
    extremes[keep_cols].to_csv(extremes_out, index=False)

    # Markdown report
    report_path = os.path.join(args.outdir, "integrity_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# PRC Fuel Interval Integrity Report\n\n")
        f.write(f"_Generated: {datetime.utcnow().isoformat(timespec='seconds')}Z_\n\n")
        f.write("**Source files:**\n\n")
        f.write(f"- `{args.fuel_parquet}`\n")
        f.write(f"- `{args.flightlist_parquet}`\n\n")
        f.write("## Summary\n\n")
        f.write(f"- Total rows: **{total_rows:,}**\n")
        f.write(f"- Rows with any anomaly: **{n_anom:,}** ({pct(n_anom, total_rows):.2f}%)\n\n")
        f.write("### Thresholds\n")
        f.write(f"- Min interval seconds: `{args.min_dt_s}`\n")
        f.write(f"- Max plausible fuel/interval (kg): `{args.max_fuel_kg}`\n")
        f.write(f"- Max plausible fuel rate (kg/min): `{args.max_rate_kgmin}`\n\n")

        # Per-rule counts
        f.write("## Rule Hits\n\n")
        f.write("| Rule | Description | Count | Share |\n|---|---|---:|---:|\n")
        rule_desc = {
            "A_dt_missing":    "Duration missing/NaN",
            "B_dt_negative":   "Duration negative",
            "C_dt_too_small":  f"Duration < {args.min_dt_s}s",
            "D_dt_zero":       "Duration == 0s",
            "E_fuel_nan":      "fuel_kg is NaN",
            "F_fuel_negative": "fuel_kg < 0",
            "G_fuel_too_big":  f"fuel_kg > {args.max_fuel_kg}",
            "H_rate_nan":      "fuel_kg_min is NaN",
            "I_rate_negative":  "fuel_kg_min < 0",
            "J_rate_too_big":   f"fuel_kg_min > {args.max_rate_kgmin}",
            "K_multiple_1000":  "fuel_kg multiple of 1000 (≥1000) → gram-upscale suspicion",
            "L_start_eq_end":   "start == end (identical timestamps)",
        }
        for k in flag_cols:
            c = int(df[k].sum())
            f.write(f"| `{k}` | {rule_desc.get(k,k)} | {c:,} | {pct(c,total_rows):.2f}% |\n")
        f.write("\n")

        f.write("## Plausible Hypotheses\n\n")
        f.write("1. **Unit upscale (kg→g)** on a minority of records (flag `K_multiple_1000`).\n")
        f.write("2. **Near-zero/zero durations** (flags `C_dt_too_small`/`D_dt_zero`) from timestamp equality/rounding.\n")
        f.write("3. **Rate explosions** driven by (1)+(2) (flag `J_rate_too_big`).\n")
        f.write("4. **Route concentration** of anomalies inflating tail metrics.\n\n")

        # Top routes table (head)
        f.write("## Top Routes by Anomalies (head)\n\n")
        f.write("| Route | Rows | Anom | Share% | ×1000 | dt=0 | dt<min | fuel>max | rate>max |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for _, r in route_agg.head(15).iterrows():
            f.write(f"| {r['route']} | {int(r['rows']):,} | {int(r['anomalies']):,} | {100*r['share_anom']:.2f} | "
                    f"{int(r['k_1000']):,} | {int(r['dt_zero']):,} | {int(r['dt_small']):,} | "
                    f"{int(r['too_big']):,} | {int(r['rate_too_big']):,} |\n")
        f.write("\n(Full CSV: `top_routes_by_anomaly.csv`)\n\n")

        # Top flight IDs table (head)
        f.write("## Top Flight IDs by Anomalies (head)\n\n")
        f.write("| Flight ID | Rows | Anom | Share% | ×1000 | dt=0 | dt<min | fuel>max | rate>max |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for _, r in fid_agg.head(15).iterrows():
            f.write(f"| {r['flight_id']} | {int(r['rows']):,} | {int(r['anomalies']):,} | {100*r['share_anom']:.2f} | "
                    f"{int(r['k_1000']):,} | {int(r['dt_zero']):,} | {int(r['dt_small']):,} | "
                    f"{int(r['too_big']):,} | {int(r['rate_too_big']):,} |\n")
        f.write("\n(Full CSV: `top_flightids_by_anomaly.csv`)\n\n")

        # Extreme examples preview
        f.write("## Extreme Examples (preview)\n\n")
        f.write("| idx | flight_id | route | start | end | dt_s | fuel_kg | fuel_kg_min | ×1000 | dt=0 |\n")
        f.write("|---:|---|---|---|---|---:|---:|---:|---:|---:|\n")
        for _, r in extremes.iterrows():
            f.write(f"| {int(r['idx'])} | {r['flight_id']} | {r.get('route','?')} | "
                    f"{r['start_dt']} | {r['end_dt']} | {r['dt_s']:.1f} | {r['fuel_kg']:.0f} | "
                    f"{(r['fuel_kg_min'] if np.isfinite(r['fuel_kg_min']) else np.nan):.1f} | "
                    f"{int(r['K_multiple_1000'])} | {int(r['D_dt_zero'])} |\n")
        f.write("\n(Full CSV: `extreme_examples.csv`)\n\n")

        if args.simulate_divide_1000:
            sim = df.copy()
            mask = sim["K_multiple_1000"].fillna(False)
            sim.loc[mask, "fuel_kg_sim"] = sim.loc[mask, "fuel_kg"] / 1000.0
            sim.loc[~mask, "fuel_kg_sim"] = sim.loc[~mask, "fuel_kg"]
            sim["fuel_kg_min_sim"] = sim["fuel_kg_sim"] / sim["dt_min"]
            j_rate_sim = (sim["fuel_kg_min_sim"] > args.max_rate_kgmin).sum()
            g_fuel_sim = (sim["fuel_kg_sim"] > args.max_fuel_kg).sum()
            f.write("## What-if: divide-by-1000 simulation\n\n")
            f.write(f"- Rows flagged `K_multiple_1000`: **{int(mask.sum()):,}**\n")
            f.write(f"- After /1000: rows still `fuel_kg > max`: **{g_fuel_sim:,}**\n")
            f.write(f"- After /1000: rows still `fuel_kg_min > max`: **{j_rate_sim:,}**\n\n")

        f.write("## Recommendations\n\n")
        f.write(f"- Quarantine rows with `dt_s < {args.min_dt_s}s` or `start==end`.\n")
        f.write("- Inspect rows flagged `K_multiple_1000`; if raw source is kg, divide-by-1000 repair is warranted.\n")
        f.write("- Keep internal numerics in SI (kg, seconds). Only format for Excel at export boundaries.\n")

    print("✅ Integrity scan complete.")
    print(f"- Report: {report_path}")
    print(f"- Anomalies CSV: {anomalies_out}")
    print(f"- Top routes CSV: {routes_out}")
    print(f"- Top flight IDs CSV: {fids_out}")
    print(f"- Extreme examples CSV: {extremes_out}")

if __name__ == "__main__":
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 200)
    main()
