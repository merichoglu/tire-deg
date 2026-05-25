"""Sanity-check the cleaned parquet.

Reports:
- Races saved per season
- Per-race lap counts (look for outliers — too few = filter too aggressive)
- Compound mix per race (soft under-representation check)
- Tyre-age distribution per compound (sample size at long stints)
- Per-stint compound proportions on lap 1 of stint, to check the outlap-drop bias
- Driver / team coverage
- Missing values per column
"""

from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/laps_clean.parquet")
df = pd.read_parquet(DATA_PATH)

print("=" * 70)
print(f"DATASET: {len(df):,} laps")
print("=" * 70)

print("\n-- Races per season --")
print(df.groupby("year")["event"].nunique())

print("\n-- Per-race lap counts (sorted) --")
race_counts = df.groupby(["year", "event"]).size().sort_values()
print("Smallest 8:")
print(race_counts.head(8).to_string())
print("Largest 4:")
print(race_counts.tail(4).to_string())

print("\n-- Compound mix overall --")
print(df["compound"].value_counts())
print("As fraction:")
print((df["compound"].value_counts(normalize=True) * 100).round(1))

print("\n-- Compound mix per race (top 5 most soft-heavy and most soft-light) --")
comp_per_race = (
    df.groupby(["year", "event"])["compound"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
    .mul(100)
    .round(1)
)
print("Most SOFT laps:")
print(comp_per_race.sort_values("SOFT", ascending=False).head(5))
print("Least SOFT laps:")
print(comp_per_race.sort_values("SOFT").head(5))

print("\n-- Tyre age coverage by compound --")
for c in ["SOFT", "MEDIUM", "HARD"]:
    s = df[df["compound"] == c]["tyre_life"]
    if s.empty:
        continue
    print(
        f"  {c:<7s}  n={len(s):>5d}  median={s.median():>4.0f}  "
        f"p95={s.quantile(0.95):>4.0f}  max={s.max():>4.0f}"
    )

print("\n-- Stint-position breakdown (1st / 2nd / 3rd+ stint of a driver's race) --")
print(df.groupby("stint")["compound"].value_counts().unstack(fill_value=0))

print("\n-- Driver and team counts --")
print(f"  Unique drivers: {df['driver'].nunique()}")
print(f"  Unique teams:   {df['team'].nunique()}")

print("\n-- Missing values --")
miss = df.isna().sum()
print(miss[miss > 0] if (miss > 0).any() else "  none")

print("\n-- Weather range across dataset --")
print(
    df[["air_temp", "track_temp", "humidity", "wind_speed"]]
    .agg(["min", "median", "max"])
    .round(1)
)

print("\n-- Spain check (was hitting wet-race / NaN bugs) --")
spain = df[df["event"].str.contains("Spanish", na=False)]
print(spain.groupby(["year", "event"]).size())
