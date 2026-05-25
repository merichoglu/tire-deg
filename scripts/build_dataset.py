"""Fetch race data for 2023-2025, filter to clean laps, save a parquet.

Filtering rules (Layer 2 of pipeline):
- IsAccurate must be True
- TrackStatus must be exactly '1' (all-green flag)
- Lap must not be Deleted
- No PitInTime or PitOutTime on the lap (in/out laps)
- Drop lap 1 of the race (race start)
- Drop lap 1 of any non-first stint (outlap warmup, even if pit times not set)
- LapTime must be present
- Race must be dry (no Rainfall on any kept lap of that race)

Output columns are lap-level with track/weather context joined in.
"""

import sys
import time
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

CACHE_DIR = Path("cache")
OUT_PATH = Path("data/laps_clean.parquet")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

SEASONS = [2023, 2024, 2025]


def process_session(year: int, round_no: int, event_name: str) -> pd.DataFrame | None:
    """Load one race, return clean lap-level dataframe or None on failure / wet race."""
    try:
        session = fastf1.get_session(year, round_no, "R")
        session.load(laps=True, telemetry=False, weather=True, messages=False)
    except Exception as e:
        print(f"  ! load failed: {e}")
        return None

    laps = session.laps.copy()
    if laps.empty:
        return None

    # Weather: lap-level merge by nearest time
    weather = session.weather_data
    if weather is None or weather.empty:
        return None

    # Check if race was wet — disqualify only if a meaningful fraction of the race
    # window had rainfall. A single damp pre-race minute doesn't make a race wet.
    race_start = laps["LapStartTime"].min()
    race_end = laps["Time"].max()
    race_weather = weather[
        (weather["Time"] >= race_start) & (weather["Time"] <= race_end)
    ]
    if len(race_weather) > 0:
        wet_frac = race_weather["Rainfall"].mean()
        if wet_frac > 0.10:
            print(f"  - wet race ({wet_frac:.0%} of samples), skipped")
            return None

    # Convert LapTime to seconds (float)
    laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()

    # Filter mask
    keep = (
        laps["IsAccurate"]
        & (laps["TrackStatus"].astype(str) == "1")
        & (laps["Deleted"].astype(str).str.lower() != "true")
        & laps["PitInTime"].isna()
        & laps["PitOutTime"].isna()
        & laps["LapTimeSec"].notna()
        & (laps["LapNumber"] > 1)
    )
    laps = laps[keep].copy()
    # Drop rows missing any of the integer fields we cast below — Spain sessions
    # sometimes leave NaN TyreLife/Stint/LapNumber on edge-case laps.
    laps = laps.dropna(subset=["LapNumber", "Stint", "TyreLife"]).copy()
    if laps.empty:
        return None

    # Drop lap 1 of each stint (outlap warmup). After filter above, "first lap of stint
    # currently in dataframe" may not equal TyreLife==1, but TyreLife is the cleaner
    # signal: drop TyreLife==1 rows for non-first stints (used sets sometimes start >1).
    # For first stint, TyreLife==1 corresponds to race lap 1 which we already dropped.
    first_stint_per_driver = laps.groupby("Driver")["Stint"].transform("min")
    is_outlap = (laps["TyreLife"] == 1) & (laps["Stint"] != first_stint_per_driver)
    laps = laps[~is_outlap].copy()

    # Drop incident laps: lap time more than 10% above the driver's stint median.
    # IsAccurate=True + TrackStatus='1' can still let through laps slowed by traffic
    # immediately following a crash that hasn't yet triggered a yellow flag in the
    # data (e.g. lap 36 of Mexico 2023, where the field bunched around an incident
    # before the SC came out). Tyre degradation across ~30 laps adds <2% to lap time,
    # so a 10% cutoff is comfortably above the deg signal we want to keep.
    stint_median = laps.groupby(["Driver", "Stint"])["LapTimeSec"].transform("median")
    laps = laps[laps["LapTimeSec"] <= 1.10 * stint_median].copy()

    # Per-lap weather via merge_asof on Time
    weather_sorted = weather.sort_values("Time")
    laps_sorted = laps.sort_values("Time")
    laps_with_weather = pd.merge_asof(
        laps_sorted,
        weather_sorted[["Time", "AirTemp", "TrackTemp", "Humidity", "WindSpeed"]],
        on="Time",
        direction="nearest",
    )

    # Keep only the columns we need going forward
    out = pd.DataFrame(
        {
            "year": year,
            "round": round_no,
            "event": event_name,
            "driver": laps_with_weather["Driver"].values,
            "team": laps_with_weather["Team"].values,
            "lap_number": laps_with_weather["LapNumber"].astype(int).values,
            "stint": laps_with_weather["Stint"].astype(int).values,
            "compound": laps_with_weather["Compound"].values,
            "tyre_life": laps_with_weather["TyreLife"].astype(int).values,
            "fresh_tyre": laps_with_weather["FreshTyre"].astype(bool).values,
            "lap_time_s": laps_with_weather["LapTimeSec"].values,
            "air_temp": laps_with_weather["AirTemp"].values,
            "track_temp": laps_with_weather["TrackTemp"].values,
            "humidity": laps_with_weather["Humidity"].values,
            "wind_speed": laps_with_weather["WindSpeed"].values,
            "position": laps_with_weather["Position"].values,
        }
    )

    # Final compound filter: keep only dry compounds
    out = out[out["compound"].isin(["SOFT", "MEDIUM", "HARD"])].copy()
    return out


def main():
    all_frames = []
    started = time.time()
    for year in SEASONS:
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as e:
            print(f"[{year}] schedule fetch failed: {e}")
            continue

        # Round 0 is sometimes pre-season testing — already excluded above. Filter to
        # rounds with EventDate in the past (FastF1 will fail on future races).
        today = pd.Timestamp.now()
        schedule = schedule[schedule["EventDate"] <= today]

        for _, row in schedule.iterrows():
            round_no = int(row["RoundNumber"])
            event_name = row["EventName"]
            elapsed = time.time() - started
            print(f"[{year} R{round_no:02d}] {event_name} (t+{elapsed:.0f}s)")
            try:
                df = process_session(year, round_no, event_name)
            except Exception as e:
                print(f"  ! unexpected error: {e}")
                continue
            if df is not None and not df.empty:
                print(f"  + {len(df)} clean laps")
                all_frames.append(df)

    if not all_frames:
        print("No data collected.")
        sys.exit(1)

    full = pd.concat(all_frames, ignore_index=True)
    full.to_parquet(OUT_PATH, index=False)
    print()
    print(f"Saved {len(full):,} clean laps to {OUT_PATH}")
    print(f"Seasons: {sorted(full['year'].unique())}")
    print(f"Races:   {full.groupby('year')['event'].nunique().to_dict()}")
    print(f"Compounds: {full['compound'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
