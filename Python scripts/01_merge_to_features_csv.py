from pathlib import Path
import argparse
import numpy as np
import pandas as pd

# ---------------- Geometry ----------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# ---------------- Load & Clean ----------------
def load_sources(root: Path):
    # flightlist: only the needed columns (no *_name fields)
    fl_cols = ["flight_id","flight_date","aircraft_type","takeoff","landed","origin_icao","destination_icao"]
    fl = pd.read_parquet(root / "flightlist_train.parquet", engine="pyarrow", columns=fl_cols)
    fl = fl.rename(columns={"destination_icao": "dest_icao"})

    fu = pd.read_parquet(root / "fuel_train.parquet", engine="pyarrow")
    ap = pd.read_parquet(root / "apt.parquet",  engine="pyarrow")
    return fl, fu, ap

def clean_types(flightlist: pd.DataFrame, fuel: pd.DataFrame, apt: pd.DataFrame):
    flightlist["flight_date"] = pd.to_datetime(flightlist["flight_date"], errors="coerce")
    for c in ["takeoff", "landed"]:
        flightlist[c] = pd.to_datetime(flightlist[c], errors="coerce")
    for c in ["origin_icao","dest_icao","aircraft_type","flight_id"]:
        flightlist[c] = flightlist[c].astype(str)

    fuel["start"] = pd.to_datetime(fuel["start"], errors="coerce")
    fuel["end"]   = pd.to_datetime(fuel["end"],   errors="coerce")
    fuel["flight_id"] = fuel["flight_id"].astype(str)

    apt["icao"] = apt["icao"].astype(str)
    return flightlist, fuel, apt

# ---------------- Airport join & route features ----------------
def join_airports_to_flightlist(flightlist: pd.DataFrame, apt: pd.DataFrame) -> pd.DataFrame:
    ap = apt.rename(columns={"icao":"icao","longitude":"lon","latitude":"lat","elevation":"elev_ft"})

    fl = (flightlist
          .merge(ap.add_prefix("origin_"), left_on="origin_icao", right_on="origin_icao", how="left")
          .merge(ap.add_prefix("dest_"),   left_on="dest_icao",   right_on="dest_icao",   how="left"))

    fl["gc_distance_km"]   = haversine_km(fl["origin_lat"], fl["origin_lon"], fl["dest_lat"], fl["dest_lon"])
    fl["elev_delta_ft"]    = fl["dest_elev_ft"] - fl["origin_elev_ft"]
    fl["flight_duration_min"] = (fl["landed"] - fl["takeoff"]).dt.total_seconds() / 60.0

    # calendar strings
    fl["month"] = fl["flight_date"].dt.month_name()
    fl["dow"]   = fl["flight_date"].dt.day_name().str.lower()
    return fl

# ---------------- Build interval-level features ----------------
def build_interval_features(root: Path) -> pd.DataFrame:
    flightlist, fuel, apt = load_sources(root)
    flightlist, fuel, apt = clean_types(flightlist, fuel, apt)
    flx = join_airports_to_flightlist(flightlist, apt)

    df = fuel.merge(flx, on="flight_id", how="left", validate="many_to_one")

    # Interval timing
    df["interval_min"] = (df["end"] - df["start"]).dt.total_seconds() / 60.0
    df["t_since_takeoff_min"] = (df["start"] - df["takeoff"]).dt.total_seconds() / 60.0
    df["t_to_landing_min"]    = (df["landed"] - df["end"]).dt.total_seconds() / 60.0
    midpoint = df["start"] + (df["end"] - df["start"]) / 2
    df["pct_elapsed_mid"] = (
        (midpoint - df["takeoff"]).dt.total_seconds() /
        (df["landed"] - df["takeoff"]).dt.total_seconds()
    ) * 100.0

    # Hours (UTC)
    df["start_hour_utc"] = df["start"].dt.hour
    df["end_hour_utc"]   = df["end"].dt.hour

    # Avg speed over whole flight
    df["avg_speed_km_per_min"] = df["gc_distance_km"] / df["flight_duration_min"]

    # ---------- CHANGE: compute fuel_kg_min and drop original fuel_kg ----------
    valid = df["interval_min"].astype(float) > 0
    df["fuel_kg_min"] = np.where(valid, df["fuel_kg"] / df["interval_min"], np.nan)
    if "fuel_kg" in df.columns:
        df = df.drop(columns=["fuel_kg"])

    # Column order
    front = [
        "idx","flight_id","start","end","interval_min","fuel_kg_min",
        "takeoff","landed","flight_duration_min",
        "t_since_takeoff_min","t_to_landing_min","pct_elapsed_mid",
        "start_hour_utc","end_hour_utc",
        "aircraft_type",
        "origin_icao","dest_icao",
        "gc_distance_km","elev_delta_ft",
        "origin_lon","origin_lat","origin_elev_ft",
        "dest_lon","dest_lat","dest_elev_ft",
        "month","dow",
        "avg_speed_km_per_min",
    ]
    front = [c for c in front if c in df.columns]
    df = df[front + [c for c in df.columns if c not in front]]

    # ---------- CHANGE: round only non-integer values; never round lat/lon ----------
    latlon_skip = {"origin_lon","origin_lat","dest_lon","dest_lat"}
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Work only on float columns that are not lat/lon
    float_cols = [c for c in num_cols if np.issubdtype(df[c].dtype, np.floating) and c not in latlon_skip]

    for c in float_cols:
        s = df[c]
        # mask of values that are NOT integers (allow for NaN)
        mask = s.notna() & ((s % 1) != 0)
        df.loc[mask, c] = s.loc[mask].round(2)

    return df

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(
        description="Build interval-level features CSV (fuel_kg_min = fuel_kg/interval_min). Round only non-integer float values; never round lat/lon."
    )
    ap.add_argument("--root", default="prc-2025-datasets", help="Dataset root")
    ap.add_argument("--out",  default=r"data\processed\features_intervals.csv", help="Output CSV path")
    ap.add_argument("--head", type=int, default=8, help="Preview N rows")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    feats = build_interval_features(Path(args.root))
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print("\n=== Interval-level features built ===")
        print(f"Rows: {len(feats):,} | Cols: {feats.shape[1]}")
        print("\nHead:")
        print(feats.head(args.head))

    # Note: no float_format here — keeps integers unformatted
    feats.to_csv(out, index=False)
    print(f"\nSaved CSV → {out}\nDone.")

if __name__ == "__main__":
    main()
