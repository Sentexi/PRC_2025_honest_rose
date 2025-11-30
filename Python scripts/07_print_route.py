#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
07_print_route.py

Usage:
    python 07_print_route.py --idx 123

What it does:
- Reads data/processed/features_intervals.csv
- Finds the row with idx == --idx and extracts flight_id
- Opens prc-2025-datasets/flights_train/{flight_id}.parquet (via pyarrow)
- Prints the entire file to console
- Saves it as data/temp/{idx}_{flight_id}.csv in a German-Excel-friendly format:
  - sep=';'
  - decimal=','
  - encoding='utf-8-sig'
"""

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description="Print and export a flight route by idx.")
    parser.add_argument("--idx", type=int, required=True, help="Index (idx) to look up in features_intervals.csv")
    args = parser.parse_args()

    root = Path.cwd()
    features_csv = root / "data" / "processed" / "features_intervals.csv"

    # --- Load features_intervals.csv
    df_features = pd.read_csv(features_csv)

    if "idx" not in df_features.columns:
        raise KeyError("Column 'idx' not found in features_intervals.csv")
    if "flight_id" not in df_features.columns:
        raise KeyError("Column 'flight_id' not found in features_intervals.csv")

    # --- Locate row by idx (numeric match, fallback to string)
    mask = pd.to_numeric(df_features["idx"], errors="coerce").eq(args.idx)
    if not mask.any():
        mask = df_features["idx"].astype(str).eq(str(args.idx))
    if not mask.any():
        raise ValueError(f"No row found with idx == {args.idx}")
    row = df_features.loc[mask].iloc[0]

    flight_id = str(row["flight_id"]).strip()
    if not flight_id:
        raise ValueError(f"Empty flight_id for idx {args.idx}")

    parquet_path = root / "prc-2025-datasets" / "flights_train" / f"{flight_id}.parquet"

    # --- Read Parquet using PyArrow
    table = pq.read_table(parquet_path)
    df_flight = table.to_pandas()

    # --- Print full dataframe to console (careful: may be large)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    print(df_flight.to_string(index=False))

    # --- Save CSV in German-Excel-friendly format
    out_dir = root / "data" / "temp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.idx}_{flight_id}.csv"

    # If you want a consistent datetime format in CSV, set date_format below.
    df_flight.to_csv(
        out_path,
        index=False,
        sep=';',
        decimal=',',
        encoding='utf-8-sig',
        na_rep='',
        date_format='%Y-%m-%d %H:%M:%S'
    )

    print(f"\nSaved CSV (German Excel friendly) to: {out_path}")


if __name__ == "__main__":
    main()
