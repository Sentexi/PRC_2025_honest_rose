#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Schnelle Berechnung von alt_diff, vs_mean_fpm & alt_mean_ft mit parallelem I/O (SUBMISSION-Version).

- Liest  data/processed/submission_intervals.csv
- Für jedes idx: data/processed/Intervals_submission/[idx]_flight_data.csv
  -> streamt nur 'timestamp','altitude' (keine Datums-Parse, keine Sortierung)
  -> nimmt erste und letzte NICHT-NA-Höhe in Dateireihenfolge
- Schreibt data/processed/submission_intervals_v1.csv
- points_file_exists als erste Spalte

Tipp: --workers an CPU/SSD anpassen (I/O-bound, 8–16 oft gut).
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial


def alt_diff_stream(points_path: str, chunksize: int = 200_000) -> float:
    """
    Streamt die Punkte-CSV und ermittelt:
    alt_diff = last_alt_non_null - first_alt_non_null
    - Nur usecols ['timestamp','altitude']
    - Keine Datums-Konvertierung (Reihenfolge = Dateireihenfolge)
    - Robust gegen sehr große Dateien
    """
    if not os.path.exists(points_path):
        return np.nan

    first_alt = None
    last_alt = None

    try:
        usecols = ["timestamp", "altitude"]
        for chunk in pd.read_csv(
            points_path,
            usecols=usecols,
            dtype={"timestamp": "string", "altitude": "float64"},
            chunksize=chunksize
        ):
            s = chunk["altitude"].dropna()
            if s.empty:
                continue
            if first_alt is None:
                first_alt = float(s.iloc[0])
            last_alt = float(s.iloc[-1])
    except Exception:
        return np.nan

    if first_alt is None or last_alt is None:
        return np.nan
    return last_alt - first_alt


def alt_mean_stream(points_path: str, chunksize: int = 200_000) -> float:
    """
    Liefert die mittlere Flughöhe (ft) über alle gültigen Punkte im File.
    """
    if not os.path.exists(points_path):
        return np.nan

    sum_alt = 0.0
    cnt_alt = 0

    try:
        for chunk in pd.read_csv(
            points_path,
            usecols=["timestamp", "altitude"],
            dtype={"timestamp": "string", "altitude": "float64"},
            chunksize=chunksize
        ):
            s = chunk["altitude"].dropna()
            if s.empty:
                continue
            sum_alt += float(s.sum())
            cnt_alt += int(s.shape[0])
    except Exception:
        return np.nan

    if cnt_alt == 0:
        return np.nan
    return sum_alt / cnt_alt


def process_one(idx_val, points_dir, chunksize: int = 200_000):
    """Hilfsfunktion für parallele Ausführung."""
    points_path = os.path.join(points_dir, f"{idx_val}_flight_data.csv")
    exists = os.path.exists(points_path)
    if not exists:
        return (0, np.nan)
    return (1, alt_diff_stream(points_path, chunksize=chunksize))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--intervals-csv", default="data/processed/submission_intervals.csv",
                        help="Pfad zu data/processed/submission_intervals.csv")
    parser.add_argument("--points-dir", default="data/processed/Intervals_submission",
                        help="Ordner mit [idx]_flight_data.csv (Submission)")
    parser.add_argument("--out-csv", default="data/processed/submission_intervals_v1.csv",
                        help="Zieldatei (Submission)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Anzahl Threads für paralleles I/O (I/O-bound)")
    parser.add_argument("--chunksize", type=int, default=200_000,
                        help="CSV-Chunksize fürs Streaming der Punktdateien")
    args = parser.parse_args()

    if not os.path.exists(args.intervals_csv):
        print(f"ERROR: Datei nicht gefunden: {args.intervals_csv}", file=sys.stderr)
        sys.exit(1)

    # Haupt-CSV laden (parse_dates nicht nötig für diese Aufgabe)
    df = pd.read_csv(args.intervals_csv, low_memory=False)
    if "idx" not in df.columns:
        print("ERROR: Spalte 'idx' fehlt in submission_intervals.csv", file=sys.stderr)
        sys.exit(1)

    n = len(df)
    idx_values = df["idx"].tolist()

    # Parallel I/O vorbereiten
    worker_fn = partial(process_one, points_dir=args.points_dir, chunksize=args.chunksize)

    points_exists = np.zeros(n, dtype=int)
    alt_diffs     = np.full(n, np.nan, dtype=float)
    alt_means     = np.full(n, np.nan, dtype=float)

    # 1) alt_diff + exists parallel
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(worker_fn, idx_values[i]): i for i in range(n)}
        for fut in tqdm(as_completed(futures), total=n, unit="interval", desc="alt_diff"):
            i = futures[fut]
            try:
                exists_flag, alt_diff_val = fut.result()
            except Exception:
                exists_flag, alt_diff_val = 0, np.nan
            points_exists[i] = exists_flag
            alt_diffs[i]     = alt_diff_val

    # 2) alt_mean_ft parallel (nur für existierende Files)
    mean_jobs = [i for i in range(n) if points_exists[i] == 1]
    if mean_jobs:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures2 = {
                ex.submit(
                    alt_mean_stream,
                    os.path.join(args.points_dir, f"{idx_values[i]}_flight_data.csv"),
                    args.chunksize
                ): i for i in mean_jobs
            }
            for fut in tqdm(as_completed(futures2), total=len(futures2), unit="interval", desc="alt_mean"):
                i = futures2[fut]
                try:
                    alt_means[i] = fut.result()
                except Exception:
                    alt_means[i] = np.nan

    # vs_mean_fpm nur wenn interval_min vorhanden und >0
    vs_mean = np.full(n, np.nan, dtype=float)
    if "interval_min" in df.columns:
        interval_min = pd.to_numeric(df["interval_min"], errors="coerce")
        valid = interval_min > 0
        vs_mean[valid] = alt_diffs[valid] / interval_min[valid]

    # Ausgabe zusammensetzen: points_file_exists als erste Spalte
    df_out = df.copy()
    df_out.insert(0, "points_file_exists", points_exists)
    df_out["alt_diff"]     = alt_diffs
    df_out["vs_mean_fpm"]  = vs_mean
    df_out["alt_mean_ft"]  = alt_means

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df_out.to_csv(args.out_csv, index=False)
    print(f"Fertig. Gespeichert als: {args.out_csv}")


if __name__ == "__main__":
    main()
