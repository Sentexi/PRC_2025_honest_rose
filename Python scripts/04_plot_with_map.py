# scripts/04_plot_with_map.py
# Usage:
#   python scripts/04_plot_with_map.py --idx 0 ^
#     --root prc-2025-datasets ^
#     --features data\processed\features_intervals.csv ^
#     --dpi 160 --dot-size 1.6 --markevery 2

from pathlib import Path
import argparse, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- cartopy import & helpers ---
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.geodesic import Geodesic
except Exception as e:
    print("[!] Cartopy not available. Install with: pip install cartopy shapely", file=sys.stderr)
    raise

def load_interval(features_path: Path, idx: int):
    df = pd.read_csv(features_path)
    for c in ["start","end","takeoff","landed"]:
        if c in df.columns: df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ["idx"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    row = df.loc[df["idx"]==idx]
    if row.empty: raise ValueError(f"idx {idx} not found in {features_path}")
    row = row.iloc[0]
    req = ["flight_id","start","end","origin_icao","origin_lat","origin_lon","dest_lat","dest_lon"]
    miss = [c for c in req if c not in row or pd.isna(row[c])]
    if miss:
        raise ValueError(f"Missing required fields for idx={idx}: {miss}")
    return row

def load_traj(root: Path, flight_id: str) -> pd.DataFrame:
    fpath = root / "flights_train" / f"{flight_id}.parquet"
    if not fpath.exists(): raise FileNotFoundError(f"Trajectory parquet not found: {fpath}")
    want = ["timestamp","latitude","longitude"]
    try:
        df = pd.read_parquet(fpath, engine="pyarrow", columns=want)
    except Exception:
        df = pd.read_parquet(fpath, engine="pyarrow")
    for c in ["timestamp"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df = df.dropna(subset=["timestamp","latitude","longitude"]).sort_values("timestamp")
    return df

def label_box(ax, x, y, text, ha="center", va="center"):
    ax.text(
        x, y, text,
        fontsize=10, weight="bold",
        ha=ha, va=va, zorder=6,
        transform=ccrs.PlateCarree(),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", lw=1, alpha=0.9)
    )

def compute_extent(lats, lons, pad_deg=3):
    lat_min, lat_max = np.nanmin(lats), np.nanmax(lats)
    lon_min, lon_max = np.nanmin(lons), np.nanmax(lons)
    # pad & handle near-vertical/horizontal routes
    if not np.isfinite([lat_min, lat_max, lon_min, lon_max]).all():
        return (-20, 20, -10, 10)
    return (lon_min-pad_deg, lon_max+pad_deg, lat_min-pad_deg, lat_max+pad_deg)

def show_map_for_interval(idx: int, root: Path, features: Path, dpi=160, dot_size=1.6, markevery=2):
    row = load_interval(features, idx)
    flight_id = str(row["flight_id"])
    t0, t1 = row["start"], row["end"]

    traj = load_traj(root, flight_id)
    mask = (traj["timestamp"] >= t0) & (traj["timestamp"] <= t1)
    sub  = traj.loc[mask]

    # data arrays
    lat_all = traj["latitude"].to_numpy(); lon_all = traj["longitude"].to_numpy()
    lat_sub = sub["latitude"].to_numpy();  lon_sub = sub["longitude"].to_numpy()

    # figure + map
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(8.3, 6.0), dpi=dpi)
    ax = plt.axes(projection=proj)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # background (clean, dezent)
    ax.add_feature(cfeature.LAND.with_scale('110m'), facecolor="#f5f5f5", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale('110m'), facecolor="#eaf2fa", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.6, edgecolor="0.45", zorder=1)
    ax.add_feature(cfeature.BORDERS.with_scale('110m'), linewidth=0.4, edgecolor="0.65", zorder=1)

    # gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.8", alpha=0.8)
    gl.right_labels = False; gl.top_labels = False

    # extent
    ext = compute_extent(np.r_[lat_all, row["origin_lat"], row["dest_lat"]],
                         np.r_[lon_all, row["origin_lon"], row["dest_lon"]],
                         pad_deg=5)
    ax.set_extent(ext, crs=proj)

    # full route: line + small points
    ax.plot(lon_all, lat_all,
            transform=proj, zorder=2,
            linewidth=0.4, alpha=0.95,
            marker=".", markersize=dot_size, markevery=max(1, markevery),
            label="full route")

    # interval highlight
    if len(sub) > 0:
        ax.plot(lon_sub, lat_sub,
                transform=proj, zorder=4,
                linewidth=1.0, alpha=0.98,
                marker=".", markersize=dot_size+0.4, markevery=1,
                label=f"interval {str(t0)[:16]} → {str(t1)[:16]}")
        # start/end markers of interval
        ax.scatter([lon_sub[0]],[lat_sub[0]], transform=proj, s=28, marker="o", zorder=5)
        ax.scatter([lon_sub[-1]],[lat_sub[-1]], transform=proj, s=28, marker="x", zorder=5)

    # geodesic arc (GC) from origin to dest (ästhetische Kurve)
    try:
        gc = Geodesic().inverse([row["origin_lon"], row["origin_lat"]],
                                [row["dest_lon"],   row["dest_lat"]], npts=100)
        gc_lon = gc[:,0]; gc_lat = gc[:,1]
        ax.plot(gc_lon, gc_lat, transform=proj, color="C0", linewidth=1.0, alpha=0.35, zorder=3)
    except Exception:
        pass

    # origin/dest badges
    label_box(ax, row["origin_lon"], row["origin_lat"], str(row["origin_icao"]))
    # dest ICAO is in route_key but wir haben dest coords:
    label_box(ax, row["dest_lon"], row["dest_lat"],
              str(row.get("route_key","")).split("→")[1] if "route_key" in row and "→" in row["route_key"] else "DEST")

    # start/ende gesamte route markieren
    if len(traj):
        ax.scatter([lon_all[0]],[lat_all[0]], transform=proj, s=18, marker="^", zorder=3)
        ax.scatter([lon_all[-1]],[lat_all[-1]], transform=proj, s=18, marker="v", zorder=3)

    # Legend außerhalb (überlappt nicht)
    # Platz unten reservieren
    plt.tight_layout(rect=(0, 0.08, 1, 0.95))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=1, frameon=True)

    # Title
    ax.set_title(f"{flight_id}  |  idx={idx}", pad=10)
    plt.show()

def main():
    ap = argparse.ArgumentParser(description="Plot flight with world map background and highlighted interval.")
    ap.add_argument("--idx", type=int, required=True)
    ap.add_argument("--root", default="prc-2025-datasets")
    ap.add_argument("--features", default=r"data\processed\features_intervals.csv")
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--dot-size", type=float, default=1.6)
    ap.add_argument("--markevery", type=int, default=2)
    args = ap.parse_args()
    show_map_for_interval(
        idx=args.idx,
        root=Path(args.root),
        features=Path(args.features),
        dpi=args.dpi,
        dot_size=args.dot_size,
        markevery=args.markevery
    )

if __name__ == "__main__":
    main()
