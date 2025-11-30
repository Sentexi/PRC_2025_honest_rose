#!/usr/bin/env python3
# 10_construct_weather_dataset.py
#
# Build a weather dataset by calling 06_prepare_weather_dataset.get_weather_for_idx(idx)
# for all idx values listed in data/processed/features_intervals_v3.csv.
#
# - Skips idx that already exist in data/processed/weather_data.csv
# - Respects Open-Meteo free-tier limits (default: 550/min; change with --per-minute)
# - Uses tqdm for progress
# - Expands CSV header dynamically if new keys appear later

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path
import importlib.util
import sys
import re

try:
    from tqdm import tqdm
except Exception:
    # Minimal fallback if tqdm is not installed
    def tqdm(x, **kwargs):
        return x

# -------- Settings (paths default to your layout) --------
DEFAULT_FEATURES_CSV = Path("data/processed/submission_intervals_v3.csv")
DEFAULT_OUT_CSV = Path("data/processed/weather_data_submission.csv")
DEFAULT_SIX_PATH = Path(__file__).resolve().parent / "06A_prepare_weather_dataset_submission.py"


# -------- Utilities --------
def load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def sniff_delimiter(csv_path: Path, default=","):
    try:
        sample = csv_path.read_text(encoding="utf-8", errors="ignore")[:8192]
        dialect = csv.Sniffer().sniff(sample)
        return dialect.delimiter
    except Exception:
        return default


def read_idx_from_features(features_csv: Path):
    """Return a list of idx values from features_intervals_v3.csv (any delimiter)."""
    if not features_csv.exists():
        raise FileNotFoundError(f"Features file not found: {features_csv}")

    delim = sniff_delimiter(features_csv)
    idxs = []
    with features_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        if "idx" not in (reader.fieldnames or []):
            raise ValueError(f"'idx' column not found in {features_csv}")
        for row in reader:
            val = row.get("idx")
            if val is None or val == "":
                continue
            try:
                idxs.append(int(val))
            except Exception:
                # tolerate non-integer values; skip
                continue
    return idxs


def read_existing_idx_set(out_csv: Path):
    """Return a set of idx already present in weather_data.csv (comma delimiter)."""
    if not out_csv.exists():
        return set()
    with out_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "idx" not in reader.fieldnames:
            return set()
        s = set()
        for row in reader:
            try:
                s.add(int(row["idx"]))
            except Exception:
                continue
    return s


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_row(idx: int, payload: dict):
    """Return one flat dict for CSV: idx first + payload (None -> '')"""
    row = {"idx": int(idx)}
    for k, v in payload.items():
        # skip duplicate idx if present in payload
        if k == "idx":
            continue
        if v is None:
            row[k] = ""
        elif isinstance(v, (dict, list, tuple)):
            row[k] = json.dumps(v, ensure_ascii=False)  # defensive
        else:
            row[k] = v
    return row


def current_fieldnames(out_csv: Path):
    if not out_csv.exists():
        return None
    with out_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames


def unified_header(existing_fields, new_row_fields):
    """Return header with 'idx' first, then all other fields (existing first order preserved,
    new ones appended sorted for stability)."""
    base = []
    if existing_fields:
        base = [*existing_fields]
    if not base:
        base = ["idx"]

    # Ensure 'idx' is first
    if "idx" in base:
        base = ["idx"] + [c for c in base if c != "idx"]
    else:
        base = ["idx"] + base

    # Add any new fields
    extras = [c for c in new_row_fields if c not in base]
    if extras:
        extras_sorted = sorted([c for c in extras if c != "idx"])
        base += extras_sorted
    return base


def rewrite_with_new_header(out_csv: Path, new_header):
    """Re-write existing CSV with an expanded header; keep previous rows."""
    if not out_csv.exists():
        return
    with out_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        old_rows = list(reader)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_header)
        writer.writeheader()
        for r in old_rows:
            writer.writerow({k: r.get(k, "") for k in new_header})


def append_row(out_csv: Path, header, row_dict):
    file_exists = out_csv.exists()
    with out_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row_dict.get(k, "") for k in header})


class RateLimiter:
    """Sliding-window per-minute limiter."""
    def __init__(self, per_minute: int = 550):
        self.per_minute = max(1, int(per_minute))
        self.calls = deque()  # monotonic timestamps

    def wait_turn(self):
        now = time.monotonic()
        window = 60.0
        # Drop timestamps older than a minute
        while self.calls and (now - self.calls[0]) >= window:
            self.calls.popleft()

        # If we already hit the window cap, sleep until space frees
        while len(self.calls) >= self.per_minute:
            earliest = self.calls[0]
            to_sleep = window - (now - earliest)
            if to_sleep > 0:
                # sleep in small chunks so ctrl+c feels responsive
                time.sleep(min(to_sleep, 1.0))
                now = time.monotonic()
                while self.calls and (now - self.calls[0]) >= window:
                    self.calls.popleft()
            else:
                break

        # Record this call
        self.calls.append(time.monotonic())


def call_with_retries(fn, *args, retries=3, backoff=1.5, **kwargs):
    """Call a function with simple retry/backoff."""
    attempt = 0
    while True:
        try:
            return fn(*args, **kwargs)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            attempt += 1
            if attempt > retries:
                raise
            time.sleep(backoff ** attempt)


# -------- Main --------
def parse_args():
    p = argparse.ArgumentParser(description="Construct weather_data.csv by calling 06.get_weather_for_idx for each idx.")
    p.add_argument("--features", type=str, default=str(DEFAULT_FEATURES_CSV),
                   help="Path to features_intervals_v3.csv (default: data/processed/features_intervals_v3.csv)")
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT_CSV),
                   help="Output CSV path (default: data/processed/weather_data.csv)")
    p.add_argument("--six", type=str, default=str(DEFAULT_SIX_PATH),
                   help="Path to 06_prepare_weather_dataset.py (default: alongside this script)")
    p.add_argument("--per-minute", type=int, default=550,
                   help="Max API calls per minute (free tier allows up to 600/min). Default 550.")
    p.add_argument("--resume", action="store_true", default=True,
                   help="Skip idx already present in output CSV (default on).")
    return p.parse_args()


def main():
    args = parse_args()

    features_csv = Path(args.features).resolve()
    out_csv = Path(args.out).resolve()
    six_path = Path(args.six).resolve()

    # Import 06 module
    if not six_path.exists():
        # common fallback (project-root/scripts)
        alt = Path.cwd() / "scripts" / "06_prepare_weather_dataset.py"
        if alt.exists():
            six_path = alt.resolve()
        else:
            raise FileNotFoundError(f"Could not find 06_prepare_weather_dataset.py at {six_path} or {alt}")

    mod06 = load_module_from_path("mod06_weather", six_path)
    if not hasattr(mod06, "get_weather_for_idx"):
        raise AttributeError("06_prepare_weather_dataset.py must expose get_weather_for_idx(idx:int)->dict")

    # Read inputs and already-done idx
    idx_list = read_idx_from_features(features_csv)
    done = read_existing_idx_set(out_csv) if args.resume else set()

    # Prepare output dir
    ensure_parent(out_csv)

    # Get current header if any
    header = current_fieldnames(out_csv)

    limiter = RateLimiter(per_minute=args.per_minute)

    # Iterate
    to_process = [i for i in idx_list if i not in done]
    if not to_process:
        print("Nothing to do. All idx values already present.")
        return

    for idx in tqdm(to_process, desc="Building weather_data", unit="idx"):
        # Rate limit
        limiter.wait_turn()

        # Call with retry (handles transient API hiccups)
        try:
            payload = call_with_retries(mod06.get_weather_for_idx, idx, retries=3, backoff=1.6)
        except Exception as e:
            # Log and continue; we don't write a partial row
            sys.stderr.write(f"[WARN] idx={idx}: {e}\n")
            continue
            
        # NEW: collapse *_XXXhPa columns to single base columns
        payload = collapse_pressure_level_columns(payload)    

        row = normalize_row(idx, payload)

        # Ensure header is up-to-date (dynamic keys)
        new_header = unified_header(header, row.keys())
        if header != new_header:
            # If file exists and header grows, rewrite with expanded header
            if out_csv.exists() and header is not None:
                rewrite_with_new_header(out_csv, new_header)
            header = new_header

        # Append row
        append_row(out_csv, header, row)

    print(f"Done. Wrote/updated: {out_csv}")

def collapse_pressure_level_columns(payload: dict) -> dict:
    """
    Collapse keys like temperature_300hPa -> temperature (prefer chosen_level_hpa if several exist).
    Keeps chosen_level_hpa and chosen_level_ref_alt_m so you know the context.
    """
    pattern = re.compile(r'^(temperature|relative_humidity|wind_speed|wind_direction|geopotential_height)_(\d+)hPa$')
    chosen = str(payload.get("chosen_level_hpa") or "")
    collapsed = {}
    keep = {}

    for k, v in payload.items():
        m = pattern.match(k)
        if m:
            base, lvl = m.group(1), m.group(2)
            # Prefer the chosen level; otherwise first seen wins
            if chosen and lvl == chosen:
                collapsed[base] = v
            elif base not in collapsed:
                collapsed[base] = v
        else:
            keep[k] = v

    # Overwrite/insert the collapsed base keys
    keep.update(collapsed)
    return keep

if __name__ == "__main__":
    # Make relative imports work
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
