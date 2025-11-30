# scripts/02_extract_interval_points.py
# Usage (run from "2025"):
#   python scripts/02_extract_interval_points.py ^
#       --root prc-2025-datasets ^
#       --features data\processed\features_intervals.csv ^
#       --outdir data\processed\Intervals ^
#       --preview 3

from pathlib import Path
import argparse
import pandas as pd
import sys
from tqdm import tqdm  # pip install tqdm

def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # robust datetime parsing
    for c in ["start", "end"]:
        df[c] = pd.to_datetime(df[c], errors="coerce", utc=False)
    # minimal schema check
    required = {"idx", "flight_id", "start", "end"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in features CSV: {missing}")
    # idx integer for filenames
    df["idx"] = pd.to_numeric(df["idx"], errors="coerce").astype("Int64")
    return df

def load_flight_parquet(root: Path, flight_id: str) -> pd.DataFrame:
    fpath = root / "flights_train" / f"{flight_id}.parquet"
    if not fpath.exists():
        raise FileNotFoundError(f"Parquet not found for flight_id '{flight_id}': {fpath}")
    df = pd.read_parquet(fpath, engine="pyarrow")
    if "timestamp" not in df.columns:
        raise KeyError(f"'timestamp' column missing in {fpath}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)
    return df

def main():
    ap = argparse.ArgumentParser(description="Extract trajectory points per fuel interval into CSV files.")
    ap.add_argument("--root", default="prc-2025-datasets", help="Dataset root containing flights_train/")
    ap.add_argument("--features", default=r"data\processed\features_intervals.csv", help="Input features_intervals.csv")
    ap.add_argument("--outdir", default=r"data\processed\Intervals", help="Output directory for [idx]_flight_data.csv")
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

    # --- Redundancy check: skip intervals that already have an output file ---
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

    # Optional limit after skipping
    feats = feats.sort_values(["flight_id", "start"])
    if args.limit and args.limit > 0:
        feats = feats.head(args.limit).copy()

    total_to_process = len(feats)
    if total_to_process == 0:
        print(f"Nothing to do. Existing files skipped: {skipped_existing}.")
        print(f"Output dir: {outdir}")
        return

    # Process grouped by flight_id (to cache parquet loads)
    flights_cache = {}
    total_saved = 0
    errors = 0

    with tqdm(total=total_to_process, desc="Extracting intervals", unit="interval") as pbar:
        for flight_id, g in feats.groupby("flight_id", sort=False):
            # lazily load & cache the parquet for this flight
            try:
                if flight_id not in flights_cache:
                    flights_cache[flight_id] = load_flight_parquet(root, flight_id)
                fdf = flights_cache[flight_id]
            except Exception as e:
                # count all rows in this group as errors and advance the bar
                errors += len(g)
                pbar.update(len(g))
                print(f"[!] Error loading parquet for flight_id={flight_id}: {e}", file=sys.stderr)
                continue

            # iterate intervals
            for _, row in g.iterrows():
                idx = int(row["idx"])
                t0, t1 = row["start"], row["end"]
                out_path = outdir / f"{idx}_flight_data.csv"

                # Redundancy guard (in case of race/overwrite toggle)
                if out_path.exists() and not args.overwrite:
                    pbar.update(1)
                    continue

                try:
                    mask = (fdf["timestamp"] >= t0) & (fdf["timestamp"] <= t1)
                    sub = fdf.loc[mask].copy()
                    sub.to_csv(out_path, index=False)
                    total_saved += 1

                    if args.preview > 0:
                        print(f"\n[{flight_id}] idx={idx}  rows={len(sub)}  -> {out_path}")
                        if len(sub):
                            with pd.option_context("display.width", 200, "display.max_columns", 20):
                                print(sub.head(args.preview))
                except Exception as e:
                    errors += 1
                    print(f"[!] Error processing idx={idx} (flight_id={flight_id}): {e}", file=sys.stderr)
                finally:
                    pbar.update(1)

    print(f"\nDone. Intervals written: {total_saved}. Skipped existing: {skipped_existing}. Errors: {errors}.")
    print(f"Output dir: {outdir}")

if __name__ == "__main__":
    main()
