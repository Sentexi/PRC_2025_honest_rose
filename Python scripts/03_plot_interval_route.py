from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def load_interval_row(features_path: Path, idx: int):
    df = pd.read_csv(features_path)
    for c in ["start", "end", "takeoff", "landed"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=False)
    if "idx" not in df.columns:
        raise ValueError("Column 'idx' not found in features_intervals.csv")
    df["idx"] = pd.to_numeric(df["idx"], errors="coerce")
    row = df.loc[df["idx"] == idx]
    if row.empty:
        raise ValueError(f"No row in features_intervals.csv with idx={idx}")
    row = row.iloc[0]
    for c in ["flight_id", "start", "end"]:
        if c not in row or pd.isna(row[c]):
            raise ValueError(f"Missing required field '{c}' for idx={idx}")
    return row

def load_flight_trajectory(root: Path, flight_id: str) -> pd.DataFrame:
    fpath = root / "flights_train" / f"{flight_id}.parquet"
    if not fpath.exists():
        raise FileNotFoundError(f"Trajectory parquet not found: {fpath}")
    want = ["timestamp","latitude","longitude","altitude","groundspeed","source"]
    try:
        df = pd.read_parquet(fpath, engine="pyarrow", columns=want)
    except Exception:
        df = pd.read_parquet(fpath, engine="pyarrow")
    need = {"timestamp","latitude","longitude"}
    if not need.issubset(df.columns):
        raise KeyError(f"Required columns missing in {fpath} (need {need})")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)
    df = df.dropna(subset=["timestamp","latitude","longitude"]).sort_values("timestamp")
    return df

def show_route_for_interval(idx: int, features_path: Path, root: Path,
                            dpi: int = 150, markevery: int = 1, dot_size: float = 1.6):
    row = load_interval_row(features_path, idx)
    flight_id = str(row["flight_id"])
    t0, t1 = row["start"], row["end"]

    traj = load_flight_trajectory(root, flight_id)

    # Full flight
    lat_all = traj["latitude"].to_numpy()
    lon_all = traj["longitude"].to_numpy()

    # Interval subset
    mask = (traj["timestamp"] >= t0) & (traj["timestamp"] <= t1)
    sub = traj.loc[mask]
    lat_sub = sub["latitude"].to_numpy()
    lon_sub = sub["longitude"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # --- Linien + Punkte (Granularität sichtbar) ---
    # Gesamtroute: dünne Linie + kleine Punkte (markevery für Performance wählbar)
    ax.plot(
        lon_all, lat_all,
        linewidth=0.4, alpha=0.85, zorder=1,
        marker='.', markersize=dot_size, markevery=markevery,
        label="Full route"
    )

    # Intervall: etwas dicker + Punkte durchgehend (markevery=1)
    if len(sub) > 0:
        ax.plot(
            lon_sub, lat_sub,
            linewidth=2.2, alpha=0.95, zorder=3,
            marker='.', markersize=dot_size+0.4, markevery=1,
            label=f"Interval [{t0} → {t1}]"
        )
        # Start/End-Marker des Intervalls
        ax.scatter([lon_sub[0]], [lat_sub[0]], s=28, marker="o", zorder=4)
        ax.scatter([lon_sub[-1]], [lat_sub[-1]], s=28, marker="x", zorder=4)

    # Optional: Start/End der Gesamtroute markieren
    if len(traj) > 0:
        ax.scatter([lon_all[0]], [lat_all[0]], s=18, marker="^", zorder=2)
        ax.scatter([lon_all[-1]], [lat_all[-1]], s=18, marker="v", zorder=2)

    # Achsen & Titel
    ax.set_title(f"Flight route with highlighted interval\nflight_id={flight_id} | idx={idx}", pad=12)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")

    # Seitenverhältnis (grobe Projektion)
    if len(traj) > 1:
        mid_lat = np.nanmedian(lat_all)
        scale = np.cos(np.deg2rad(mid_lat))
        if scale > 0:
            ax.set_aspect(1.0 / scale)

    # --- Legende außerhalb platzieren (überlappt nicht das Canvas) ---
    # Platz für Legende unten lassen
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    leg = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2, frameon=True, framealpha=1.0
    )

    plt.show()  # nur anzeigen

def main():
    ap = argparse.ArgumentParser(description="Show full flight route with interval highlighted (points + lines).")
    ap.add_argument("--idx", type=int, required=True, help="Interval idx from features_intervals.csv")
    ap.add_argument("--root", default="prc-2025-datasets", help="Dataset root containing flights_train/")
    ap.add_argument("--features", default=r"data\processed\features_intervals.csv", help="Path to features_intervals.csv")
    ap.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    ap.add_argument("--markevery", type=int, default=1, help="Plot only every Nth point for the full route (granularity vs. speed)")
    ap.add_argument("--dot-size", type=float, default=1.6, help="Marker size for points")
    args = ap.parse_args()
    try:
        show_route_for_interval(
            idx=args.idx,
            features_path=Path(args.features),
            root=Path(args.root),
            dpi=args.dpi,
            markevery=max(1, args.markevery),
            dot_size=args.dot_size
        )
    except Exception as e:
        print(f"[!] Failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
