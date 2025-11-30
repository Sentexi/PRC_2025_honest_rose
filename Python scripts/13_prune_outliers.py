#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
13_prune_outliers.py

Create a *pruned* copy of an already processed intervals CSV by removing rows
with implausible / corrupted labels. Input is expected to live under:
    data/processed/<filename>.csv
and output will be written to:
    data/processed/<filename>_pruned.csv

We DO NOT change any values — only drop faulty rows.

Default rules (tunable via CLI flags):
- fuel_kg_min must be finite and > 0
- computed duration from start/end must be > 0
- interval_min must closely match computed duration (|Δ| ≤ 0.25 min)
- fuel_kg_min must be ≤ max plausible rate (default: 220 kg/min)

Usage:
    python scripts/13_prune_outliers.py --filename features_intervals_v3
    # optional:
    python scripts/13_prune_outliers.py --filename features_intervals_v3 --max-rate-kgmin 200 --dt-tolerance 0.2
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser(description="Prune clearly faulty rows from features_intervals CSV.")
    ap.add_argument("--filename", required=True, help="Base filename under data/processed (with or without .csv).")
    ap.add_argument("--sep", default=",", help="CSV delimiter (default ',').")
    ap.add_argument("--decimal", default=".", help="Decimal separator (default '.').")
    ap.add_argument("--max-rate-kgmin", type=float, default=220.0, help="Max plausible fuel_kg_min (default 220).")
    ap.add_argument("--dt-tolerance", type=float, default=0.25, help="Allowed |computed_dt - interval_min| in minutes (default 0.25).")
    ap.add_argument("--min-interval-min", type=float, default=0.5, help="Minimum acceptable interval_min (default 0.5).")
    args = ap.parse_args()

    in_name = args.filename
    if in_name.lower().endswith(".csv"):
        in_name = in_name[:-4]
    in_path = os.path.join("data", "processed", f"{in_name}.csv")
    out_path = os.path.join("data", "processed", f"{in_name}_pruned.csv")

    if not os.path.exists(in_path):
        print(f"ERROR: input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    # Read CSV (allowing for German-style decimal on demand)
    try:
        df = pd.read_csv(in_path, sep=args.sep, decimal=args.decimal)
    except Exception as e:
        print(f"ERROR reading CSV: {in_path}\n{e}", file=sys.stderr)
        sys.exit(1)

    n0 = len(df)

    # Required columns (we won’t modify semantics beyond these)
    required_cols = ["start", "end", "interval_min", "fuel_kg_min"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns in {in_path}: {missing}", file=sys.stderr)
        sys.exit(1)

    # Parse times & compute actual duration in minutes
    start_dt = pd.to_datetime(df["start"], errors="coerce", utc=True)
    end_dt   = pd.to_datetime(df["end"],   errors="coerce", utc=True)
    dt_min_calc = (end_dt - start_dt).dt.total_seconds() / 60.0

    # Coerce numerics we rely on
    interval_min = pd.to_numeric(df["interval_min"], errors="coerce")
    fuel_kg_min  = pd.to_numeric(df["fuel_kg_min"], errors="coerce")

    # Build flags (drop if ANY is True)
    flag_nan_rate   = ~np.isfinite(fuel_kg_min)
    flag_nonpos     = fuel_kg_min <= 0
    flag_nan_dt     = ~np.isfinite(dt_min_calc)
    flag_zero_dt    = (dt_min_calc == 0)
    flag_small_int  = (interval_min < args.min_interval_min) | ~np.isfinite(interval_min)
    flag_dt_mismatch= np.abs(dt_min_calc - interval_min) > args.dt_tolerance
    flag_rate_high  = fuel_kg_min > args.max_rate_kgmin

    # Combine: only rows with any clear fault are removed
    drop_mask = (
        flag_nan_rate | flag_nonpos |
        flag_nan_dt   | flag_zero_dt |
        flag_small_int |
        flag_dt_mismatch |
        flag_rate_high
    )

    kept = df.loc[~drop_mask].copy()
    ndrop = int(drop_mask.sum())

    # Write output (keep same delimiter/decimal style)
    try:
        kept.to_csv(out_path, index=False, sep=args.sep, decimal=args.decimal)
    except Exception as e:
        print(f"ERROR writing CSV: {out_path}\n{e}", file=sys.stderr)
        sys.exit(1)

    # Console summary
    print("✅ Pruning complete.")
    print(f"- Input : {in_path}")
    print(f"- Output: {out_path}")
    print(f"- Rows in   : {n0:,}")
    print(f"- Rows kept : {len(kept):,}")
    print(f"- Rows dropped (faulty): {ndrop:,}")

    # Quick breakdown (helpful for sanity)
    if ndrop > 0:
        reasons = {
            "nan_rate"  : int(flag_nan_rate.sum()),
            "nonpos"    : int(flag_nonpos.sum()),
            "nan_dt"    : int(flag_nan_dt.sum()),
            "zero_dt"   : int(flag_zero_dt.sum()),
            "small_int" : int(flag_small_int.sum()),
            "dt_mismatch": int(flag_dt_mismatch.sum()),
            "rate_high" : int(flag_rate_high.sum()),
        }
        print("- Drop reasons (counts, non-exclusive):")
        for k, v in reasons.items():
            if v:
                print(f"  • {k}: {v:,}")

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 200)
    main()
