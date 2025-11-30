#!/usr/bin/env python3
# 06_prepare_weather_dataset.py
# Free, no-key weather lookup for flight-interval midpoints using Open-Meteo pressure-levels.

import argparse
import datetime as dt
import math
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import requests

# (hPa, ~geopotential height m ASL) coarse reference ladder
PRESSURE_LEVELS = [
    (1000, 110), (975, 320), (950, 500), (925, 800), (900, 1000),
    (850, 1500), (800, 1900), (700, 3000), (600, 4200), (500, 5600),
    (400, 7200), (300, 9200), (250, 10400), (200, 11800), (150, 13500),
    (100, 15800), (70, 17700), (50, 19300), (30, 22000),
]

WMO_CODES = {
    0: "Clear", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Drizzle (light)", 53: "Drizzle (moderate)", 55: "Drizzle (dense)",
    56: "Freezing drizzle (light)", 57: "Freezing drizzle (dense)",
    61: "Rain (slight)", 63: "Rain (moderate)", 65: "Rain (heavy)",
    66: "Freezing rain (light)", 67: "Freezing rain (heavy)",
    71: "Snow (slight)", 73: "Snow (moderate)", 75: "Snow (heavy)",
    77: "Snow grains",
    80: "Rain showers (slight)", 81: "Rain showers (moderate)", 82: "Rain showers (violent)",
    85: "Snow showers (slight)", 86: "Snow showers (heavy)",
    95: "Thunderstorm", 96: "Thunderstorm w/ slight hail", 99: "Thunderstorm w/ heavy hail",
}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sample weather at the midpoint of a flight interval CSV using Open-Meteo pressure-levels."
    )
    p.add_argument("--idx", type=int, required=True,
                   help="Index of the interval file: data/processed/Intervals_submission/[idx]_flight_data.csv")
    return p.parse_args()

def read_midpoint_row(csv_path: str) -> pd.Series:
    # Let pandas sniff the delimiter; use python engine for robustness
    df = pd.read_csv(csv_path, sep=None, engine="python")
    needed = {"timestamp", "latitude", "longitude", "altitude"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV must contain columns: {', '.join(sorted(needed))}. Missing: {', '.join(sorted(missing))}")
    # midpoint row (or the only row)
    mid_idx = len(df) // 2
    return df.iloc[mid_idx]

def parse_timestamp_utc(ts_str: str) -> dt.datetime:
    # Accept: "YYYY-mm-dd HH:MM:SS.ssssss... (UTC)" or without suffix
    clean = ts_str.replace(" (UTC)", "").strip()
    ts = pd.to_datetime(clean, utc=True)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.floor("us").to_pydatetime()

def to_meters_detect_unit(alt_value) -> float:
    """Return altitude in meters; convert if it looks like feet (typical ADS-B range)."""
    try:
        alt = float(alt_value)
    except Exception:
        return float("nan")
    if 2000 <= alt <= 60000:  # feet heuristic
        return alt * 0.3048
    return alt  # assume meters

def nearest_pressure_level_for_alt(alt_m: float) -> Tuple[int, int]:
    if not math.isfinite(alt_m):
        return 850, 1500
    return min(PRESSURE_LEVELS, key=lambda lv: abs(lv[1] - alt_m))

def build_openmeteo_params(lat: float, lon: float, t_utc: dt.datetime, level_hpa: int) -> dict:
    # Query a ±1 day span for robustness near day boundaries; we will choose the nearest time afterwards
    start_date = (t_utc - dt.timedelta(days=1)).date().isoformat()
    end_date   = (t_utc + dt.timedelta(days=1)).date().isoformat()
    lvl = f"{level_hpa}hPa"
    hourly_vars = [
        f"temperature_{lvl}",
        f"relative_humidity_{lvl}",
        f"wind_speed_{lvl}",
        f"wind_direction_{lvl}",
        f"geopotential_height_{lvl}",
        # contextual surface fields
        "precipitation", "rain", "showers", "snowfall", "surface_pressure", "weather_code",
    ]
    return {
        "latitude": f"{lat:.6f}",
        "longitude": f"{lon:.6f}",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",
        "timeformat": "iso8601",
        "hourly": ",".join(hourly_vars),
    }

def fetch_openmeteo(params: dict) -> dict:
    #url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    url = "https://customer-historical-forecast-api.open-meteo.com/v1/forecast"
    #url = "https://customer-archive-api.open-meteo.com/v1/archive"
    #r = requests.get(url, params=params, timeout=30)
    r = requests.get(url, params={**params, "apikey": "Im5gTaLGEmh1QlHs"})
    r.raise_for_status()
    return r.json()

def _ts_utc_value(dt_like) -> int:
    """Return UTC nanoseconds since epoch for any dt/datetime/Timestamp."""
    ts = pd.Timestamp(dt_like)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.value  # int64 ns

def nearest_time_index(times_list, target_dt_utc: dt.datetime) -> int:
    """
    Robust nearest-time finder:
    - Parse returned time strings with UTC
    - Compute absolute diffs in integer nanoseconds
    - Return argmin index
    """
    if not times_list:
        raise ValueError("Open-Meteo returned no times.")
    t_idx = pd.to_datetime(times_list, utc=True)
    target_ns = _ts_utc_value(target_dt_utc)
    diffs = np.abs(t_idx.asi8 - target_ns)  # ndarray[int64]
    return int(np.argmin(diffs))

def cardinal_dir(deg: float) -> str:
    if not math.isfinite(deg):
        return "n/a"
    dirs = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
    ix = int((deg % 360) / 22.5 + 0.5) % 16
    return dirs[ix]

def main():
    args = parse_args()
    csv_path = f"data/processed/Intervals_submission/{args.idx}_flight_data.csv"

    try:
        row = read_midpoint_row(csv_path)
    except Exception as e:
        print(f"ERROR reading interval file: {e}", file=sys.stderr); sys.exit(1)

    try:
        t_mid = parse_timestamp_utc(str(row["timestamp"]))
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        alt_m = to_meters_detect_unit(row["altitude"])
    except Exception as e:
        print(f"ERROR parsing fields: {e}", file=sys.stderr); sys.exit(1)

    level_hpa, level_alt_m_ref = nearest_pressure_level_for_alt(alt_m)
    params = build_openmeteo_params(lat, lon, t_mid, level_hpa)

    try:
        data = fetch_openmeteo(params)
    except Exception as e:
        print(f"ERROR fetching Open-Meteo data: {e}", file=sys.stderr); sys.exit(1)

    if "hourly" not in data or "time" not in data["hourly"]:
        print("No hourly data returned.", file=sys.stderr); sys.exit(1)

    times = data["hourly"]["time"]
    try:
        i = nearest_time_index(times, t_mid)  # nearest to true midpoint (no brittle string compare)
    except Exception as e:
        print(f"ERROR selecting nearest time: {e}", file=sys.stderr); sys.exit(1)

    lvl = f"{level_hpa}hPa"
    def get(var):
        arr = data["hourly"].get(var)
        return None if arr is None or i >= len(arr) else arr[i]

    # Flight-level variables
    T  = get(f"temperature_{lvl}")           # °C
    RH = get(f"relative_humidity_{lvl}")     # %
    WS = get(f"wind_speed_{lvl}")            # km/h
    WD = get(f"wind_direction_{lvl}")        # °
    Zg = get(f"geopotential_height_{lvl}")   # m ASL

    # Surface context
    SP = get("surface_pressure")             # hPa
    PR = get("precipitation")                # mm
    RN = get("rain")                         # mm
    SH = get("showers")                      # mm
    SF = get("snowfall")                     # cm (per API)
    WXC = get("weather_code")

    # ---- Output ----
    print("=" * 72)
    print(f"Flight interval idx: {args.idx}")
    print(f"Midpoint (UTC): {t_mid.isoformat().replace('+00:00','Z')}")
    print(f"Model time used: {times[i]}")
    print(f"Location: lat {lat:.5f}, lon {lon:.5f}")
    if math.isfinite(alt_m):
        alt_str = f"{alt_m:.0f} m"
        try:
            raw_alt = float(row["altitude"])
            if 2000 <= raw_alt <= 60000:
                alt_str += f" (from {raw_alt:.0f} ft)"
        except Exception:
            pass
    else:
        alt_str = "n/a"
    print(f"Aircraft altitude: {alt_str}")
    print(f"Chosen pressure level: {level_hpa} hPa (ref ~{level_alt_m_ref:,} m ASL)")
    if Zg is not None:
        print(f"Model geopotential height @ {level_hpa} hPa: {Zg:.0f} m ASL")

    print("\n-- Weather @ flight level --")
    print(f"Temperature:        {T if T is not None else 'n/a'} °C")
    ws_txt = f"{WS:.1f} km/h" if WS is not None else "n/a"
    wd_txt = f"{WD:.0f}° ({cardinal_dir(WD)})" if WD is not None else "n/a"
    print(f"Wind:               {ws_txt}  from {wd_txt}")
    print(f"Relative humidity:  {RH if RH is not None else 'n/a'} %")

    print("\n-- Context near surface below --")
    print(f"Surface pressure:   {SP if SP is not None else 'n/a'} hPa")
    print(f"Precip (past hr):   total={PR if PR is not None else 'n/a'} mm | rain={RN if RN is not None else 'n/a'} mm | "
          f"showers={SH if SH is not None else 'n/a'} mm | snowfall={SF if SF is not None else 'n/a'} cm")
    if WXC is not None:
        print(f"Weather code:       {int(WXC)} ({WMO_CODES.get(int(WXC), 'wmo code')})")

    print("=" * 72)
    print("Source: Open-Meteo Historical Forecast API (pressure-level variables).")
    
def get_weather_for_idx(idx: int) -> dict:
    """
    Programmatic API for other scripts:
    Given an interval idx, return a dictionary of weather data sampled
    at the interval midpoint (UTC) using the same logic as the CLI.
    """
    csv_path = f"data/processed/Intervals_submission/{idx}_flight_data.csv"

    # 1) Read the midpoint row
    row = read_midpoint_row(csv_path)

    # 2) Parse fields
    t_mid = parse_timestamp_utc(str(row["timestamp"]))
    lat = float(row["latitude"])
    lon = float(row["longitude"])
    alt_m = to_meters_detect_unit(row["altitude"])

    # 3) Choose pressure level & build request
    level_hpa, level_alt_m_ref = nearest_pressure_level_for_alt(alt_m)
    params = build_openmeteo_params(lat, lon, t_mid, level_hpa)

    # 4) Fetch from API
    data = fetch_openmeteo(params)
    if "hourly" not in data or "time" not in data["hourly"]:
        raise RuntimeError("No hourly data returned from Open-Meteo.")

    # 5) Select nearest model time
    times = data["hourly"]["time"]
    i = nearest_time_index(times, t_mid)
    lvl = f"{level_hpa}hPa"

    def get(var):
        arr = data["hourly"].get(var)
        return None if arr is None or i >= len(arr) else arr[i]

    # Flight-level variables
    result = {
        "idx": int(idx),
        "midpoint_utc": t_mid.isoformat().replace("+00:00", "Z"),
        "model_time_utc": times[i],
        "latitude": lat,
        "longitude": lon,
        "altitude_m": None if not math.isfinite(alt_m) else float(alt_m),
        "chosen_level_hpa": int(level_hpa),
        "chosen_level_ref_alt_m": int(level_alt_m_ref),

        # Flight level (pressure level) fields
        f"temperature_{lvl}": get(f"temperature_{lvl}"),
        f"relative_humidity_{lvl}": get(f"relative_humidity_{lvl}"),
        f"wind_speed_{lvl}": get(f"wind_speed_{lvl}"),
        f"wind_direction_{lvl}": get(f"wind_direction_{lvl}"),
        f"geopotential_height_{lvl}": get(f"geopotential_height_{lvl}"),

        # Surface context
        "surface_pressure": get("surface_pressure"),
        "precipitation": get("precipitation"),
        "rain": get("rain"),
        "showers": get("showers"),
        "snowfall": get("snowfall"),
        "weather_code": int(get("weather_code")) if get("weather_code") is not None else None,
        "weather_code_text": WMO_CODES.get(int(get("weather_code"))) if get("weather_code") is not None else None,
    }

    # Also add a convenience cardinal wind dir if available
    wd = result.get(f"wind_direction_{lvl}")
    result["wind_cardinal"] = cardinal_dir(wd) if wd is not None else None

    return result

if __name__ == "__main__":
    main()
