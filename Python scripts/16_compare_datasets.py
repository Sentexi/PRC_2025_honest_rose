#!/usr/bin/env python3
"""
016_compare_datasets.py

Compares the distributions of every shared column in two CSV files:
- data/processed/features_intervals_v4.csv
- data/processed/submission_intervals_v4.csv

Creates an output folder "Data set analysis" (by default next to the CSVs)
with charts for each column:
- Categorical columns: grouped bar chart of category percentages (features vs. submission)
- Numeric columns: histogram with aligned bin edges & percentages (features vs. submission)

Usage (defaults assume repo layout):
    python 016_compare_datasets.py \
        --features data/processed/features_intervals_v4.csv \
        --submission data/processed/submission_intervals_v4.csv

Optional flags:
    --output-dir <path>           # Where to write charts (default: <csv_dir>/Data set analysis)
    --max-categories 40           # Cap for categorical levels (rest grouped as 'Other')
    --numeric-bins auto           # 'auto' | 'fd' | int
    --dpi 160                     # Figure DPI
    --style default               # 'default' | 'ggplot' | 'seaborn-v0_8' (if installed)

Dependencies: pandas, numpy, matplotlib
"""

from __future__ import annotations
import argparse
import math
import os
import re
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------- Helpers ------------------------- #

def sanitize_filename(name: str) -> str:
    name = re.sub(r"[/\\\n\r\t]", "_", str(name))
    name = re.sub(r"[^\w\-\.\s\[\]\(\),]", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:180]  # keep filenames reasonable


def safe_to_numeric(s: pd.Series) -> Tuple[pd.Series, float]:
    coerced = pd.to_numeric(s, errors="coerce")
    fraction_numeric = coerced.notna().mean() if len(coerced) else 0.0
    return coerced, fraction_numeric


def is_numeric_like(a: pd.Series, b: pd.Series, thresh: float = 0.9) -> bool:
    a_num, fa = safe_to_numeric(a)
    b_num, fb = safe_to_numeric(b)
    return (fa >= thresh) and (fb >= thresh)


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def pick_bins(data: np.ndarray, strategy: str | int = "auto") -> np.ndarray:
    """Return bin edges suitable for np.histogram.
    Uses combined data from both datasets to ensure aligned edges and clamps to a
    reasonable number of bins to keep plots readable.
    """
    data = data[np.isfinite(data)]
    if data.size == 0:
        return np.array([0.0, 1.0])
    vmin, vmax = float(np.nanmin(data)), float(np.nanmax(data))
    if vmin == vmax:
        # Single-value column -> create a tiny window around it
        eps = max(1e-9, abs(vmin) * 1e-6)
        return np.array([vmin - eps, vmax + eps])

    # If an integer is requested, respect it directly (user control)
    if isinstance(strategy, int):
        bins = max(2, int(strategy))
        return np.linspace(vmin, vmax, bins + 1)

    # Otherwise, start from a data-driven suggestion
    st = str(strategy).lower()
    try:
        base = "fd" if st in {"fd", "freedman-diaconis"} else "auto"
        edges = np.histogram_bin_edges(data, bins=base)
    except Exception:
        # robust fallback: sqrt rule
        bins = max(2, int(np.sqrt(data.size)))
        edges = np.linspace(vmin, vmax, bins + 1)

    # Clamp to keep charts compact
    MAX_BINS = 30
    if len(edges) - 1 > MAX_BINS:
        edges = np.linspace(vmin, vmax, MAX_BINS + 1)

    # Final safety
    if len(edges) < 3:
        bins = max(2, min(MAX_BINS, int(np.sqrt(data.size))))
        edges = np.linspace(vmin, vmax, bins + 1)

    return edges


def format_bin_labels(edges: np.ndarray) -> list[str]:
    labels = []
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        labels.append(f"[{a:.3g}, {b:.3g}{')' if i < len(edges) - 2 else ']'}")
    return labels


# ------------------------- Plotters ------------------------- #

def plot_categorical(column: str,
                     f_counts: pd.Series,
                     s_counts: pd.Series,
                     out_dir: str,
                     dpi: int = 160,
                     style: str = "default") -> None:
    # Align categories
    all_idx = pd.Index(sorted(set(f_counts.index).union(set(s_counts.index)), key=lambda x: str(x)))
    f = f_counts.reindex(all_idx, fill_value=0.0)
    s = s_counts.reindex(all_idx, fill_value=0.0)

    x = np.arange(len(all_idx), dtype=float)
    width = 0.45

    plt.style.use(style if style in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(max(8, len(all_idx) * 0.35), 5), dpi=dpi)
    ax.bar(x - width / 2, f.values * 100, width, label="features")
    ax.bar(x + width / 2, s.values * 100, width, label="submission")
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in all_idx], rotation=45, ha="right")
    ax.set_ylabel("Anteil am Datensatz (%)")
    ax.set_title(f"Categorical distribution: {column}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fname = os.path.join(out_dir, f"{sanitize_filename(column)}__categorical.png")
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


def plot_numeric(column: str,
                 f_vals: np.ndarray,
                 s_vals: np.ndarray,
                 edges: np.ndarray,
                 out_dir: str,
                 dpi: int = 160,
                 style: str = "default") -> None:
    # hist counts normalized to probability (percentage later)
    f_counts, _ = np.histogram(f_vals, bins=edges)
    s_counts, _ = np.histogram(s_vals, bins=edges)

    f_pct = f_counts / max(1, f_vals.size)
    s_pct = s_counts / max(1, s_vals.size)

    labels = format_bin_labels(edges)
    x = np.arange(len(labels), dtype=float)
    width = 0.45

    plt.style.use(style if style in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.35), 5), dpi=dpi)
    ax.bar(x - width / 2, f_pct * 100, width, label="features")
    ax.bar(x + width / 2, s_pct * 100, width, label="submission")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Anteil am Datensatz je Bin (%)")
    ax.set_title(f"Numeric histogram: {column}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fname = os.path.join(out_dir, f"{sanitize_filename(column)}__numeric.png")
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


# ------------------------- Core logic ------------------------- #

def build_categorical_series(s: pd.Series, max_categories: int) -> pd.Series:
    # Normalize values to strings to avoid issues with mixed types
    s = s.astype("object").astype("string")
    total = max(1, s.size)
    vc = s.value_counts(dropna=False)  # include NaN as category
    # create explicit NaN label
    vc.index = vc.index.fillna("<NA>")

    if len(vc) > max_categories:
        top = vc.iloc[:max_categories - 1]
        other = pd.Series({"Other": int(vc.iloc[max_categories - 1 :].sum())})
        vc = pd.concat([top, other])

    pct = (vc / total).sort_index()
    return pct


def analyze_column(col: str,
                   df_f: pd.DataFrame,
                   df_s: pd.DataFrame,
                   out_dir: str,
                   max_categories: int,
                   numeric_bins: str | int,
                   dpi: int,
                   style: str) -> None:
    s_f = df_f[col]
    s_s = df_s[col]

    # Decide numeric vs categorical robustly
    if is_numeric_like(s_f, s_s):
        f_num, _ = safe_to_numeric(s_f)
        s_num, _ = safe_to_numeric(s_s)
        f_vals = f_num.to_numpy(dtype=float)
        s_vals = s_num.to_numpy(dtype=float)
        comb = np.concatenate([f_vals[np.isfinite(f_vals)], s_vals[np.isfinite(s_vals)]])
        edges = pick_bins(comb, numeric_bins)
        plot_numeric(col, f_vals[np.isfinite(f_vals)], s_vals[np.isfinite(s_vals)], edges, out_dir, dpi, style)
    else:
        f_pct = build_categorical_series(s_f, max_categories)
        s_pct = build_categorical_series(s_s, max_categories)
        plot_categorical(col, f_pct, s_pct, out_dir, dpi, style)


def main():
    ap = argparse.ArgumentParser(description="Compare two dataset CSVs (features vs submission)")
    ap.add_argument("--features", default=os.path.join("data", "processed", "features_intervals_v4.csv"))
    ap.add_argument("--submission", default=os.path.join("data", "processed", "submission_intervals_v4.csv"))
    ap.add_argument("--output-dir", default=None, help="Output directory (default: next to CSVs, 'Data set analysis')")
    ap.add_argument("--max-categories", type=int, default=40)
    ap.add_argument("--numeric-bins", default="auto", help="'auto' | 'fd' | int (e.g., 30)")
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--style", default="default", help="Matplotlib style name (default, ggplot, seaborn-v0_8, ...) ")
    args = ap.parse_args()

    # Interpret numeric_bins
    try:
        nbins: str | int = int(args.numeric_bins)
    except ValueError:
        nbins = str(args.numeric_bins).lower()

    # Load CSVs
    df_f = pd.read_csv(args.features)
    df_s = pd.read_csv(args.submission)

    # Columns in common
    shared_cols = [c for c in df_f.columns if c in set(df_s.columns)]
    if not shared_cols:
        raise SystemExit("No shared columns between the two CSVs.")

    # Output dir
    if args.output_dir:
        out_dir = args.output_dir
    else:
        # default: sibling dir of the features CSV
        csv_base_dir = os.path.dirname(os.path.abspath(args.features))
        out_dir = os.path.join(csv_base_dir, "Data set analysis")

    ensure_output_dir(out_dir)

    print(f"Comparing {len(shared_cols)} shared columns. Output -> {out_dir}")

    # Iterate columns
    for i, col in enumerate(shared_cols, 1):
        subdir = out_dir  # single flat folder as requested
        ensure_output_dir(subdir)
        try:
            analyze_column(col, df_f, df_s, subdir, args.max_categories, nbins, args.dpi, args.style)
            print(f"[{i}/{len(shared_cols)}] ✓ {col}")
        except Exception as e:
            print(f"[{i}/{len(shared_cols)}] ✗ {col} -> {e}")

    # Optional: write a small manifest
    try:
        manifest_path = os.path.join(out_dir, "_manifest.txt")
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write("016_compare_datasets.py – output manifest\n")
            f.write(f"Features CSV: {os.path.abspath(args.features)}\n")
            f.write(f"Submission CSV: {os.path.abspath(args.submission)}\n")
            f.write(f"Columns compared: {len(shared_cols)}\n")
            f.write(f"Max categories: {args.max_categories}\n")
            f.write(f"Numeric bins: {args.numeric_bins}\n")
            f.write(f"Style: {args.style}\n")
    except Exception:
        pass


if __name__ == "__main__":
    main()
