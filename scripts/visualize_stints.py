"""Plot raw (uncorrected) lap-time-by-tyre-age curves for sanity-checking.

Produces three figures saved to figures/:
  1. stint_curves_by_compound.png — all stints overlaid, faceted by compound
  2. stint_curves_by_track.png — median curve per (track, compound)
  3. compound_counts.png — sample sizes per compound per stint position

These are RAW lap times (no fuel correction, no track evolution removed). The
expected shape: lap times trend UP with tyre age (degradation) but ALSO trend
DOWN over the race (fuel burn + track evolution). The two effects partially
cancel in the raw data — which is exactly why corrections are needed.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_PATH = Path("data/laps_clean.parquet")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

df = pd.read_parquet(DATA_PATH)
print(
    f"Loaded {len(df):,} laps across {df.groupby('year')['event'].nunique().to_dict()} races"
)

COMPOUND_COLORS = {"SOFT": "#d62728", "MEDIUM": "#ffcc00", "HARD": "#bbbbbb"}
COMPOUND_ORDER = ["SOFT", "MEDIUM", "HARD"]


# ---- Figure 1: median lap time vs tyre age, per compound ----
# Plot lap-time DELTA from the per-stint reference (median of laps 2-4 of the stint),
# averaged across all stints. This already partly removes per-race base-pace differences
# but does NOT remove fuel or track-evolution — so any downward slope is fuel + evo,
# any upward slope is residual deg minus those.
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
for ax, compound in zip(axes, COMPOUND_ORDER):
    sub = df[df["compound"] == compound].copy()
    # Within-stint reference: median of laps 2-4 of each stint (skip lap 1 outlap)
    ref = (
        sub[sub["tyre_life"].between(2, 4)]
        .groupby(["year", "round", "driver", "stint"])["lap_time_s"]
        .median()
        .rename("ref_pace")
    )
    sub = sub.merge(ref, left_on=["year", "round", "driver", "stint"], right_index=True)
    sub["delta_s"] = sub["lap_time_s"] - sub["ref_pace"]

    # Faint individual stints
    for _, stint in sub.groupby(["year", "round", "driver", "stint"]):
        if len(stint) < 5:
            continue
        ax.plot(
            stint["tyre_life"],
            stint["delta_s"],
            color=COMPOUND_COLORS[compound],
            alpha=0.05,
            linewidth=0.7,
        )
    # Quantile band + median
    grp = sub.groupby("tyre_life")["delta_s"]
    med = grp.median()
    p25 = grp.quantile(0.25)
    p75 = grp.quantile(0.75)
    keep_idx = med.index <= 45
    med, p25, p75 = med[keep_idx], p25[keep_idx], p75[keep_idx]
    ax.fill_between(
        med.index,
        p25.values,
        p75.values,
        color="black",
        alpha=0.15,
        label="IQR (p25-p75)",
    )
    ax.plot(med.index, med.values, color="black", linewidth=2.5, label="median")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlim(0, 45)
    ax.set_ylim(-4, 8)
    ax.set_title(f"{compound}  (n={len(sub):,} laps)", fontsize=13)
    ax.set_xlabel("Tyre age (laps)")
    if ax is axes[0]:
        ax.set_ylabel("Δ lap time vs stint's own laps 2-4 (s)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
fig.suptitle(
    "Within-stint pace change vs tyre age — uncorrected for fuel / track evolution",
    fontsize=14,
)
fig.tight_layout()
fig.savefig(FIG_DIR / "stint_curves_by_compound.png", dpi=140)
print(f"  wrote {FIG_DIR}/stint_curves_by_compound.png")


# ---- Figure 2: per-track within-stint deltas, one row per compound ----
# Use the same within-stint reference trick as Figure 1 so tracks are comparable.
# Pick the 8 tracks with the most laps; aggregate across years.
top_tracks = (
    df.groupby("event").size().sort_values(ascending=False).head(8).index.tolist()
)
short_name = lambda s: s.replace(" Grand Prix", "")

fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
cmap = plt.get_cmap("tab10")
for ax, compound in zip(axes, COMPOUND_ORDER):
    sub = df[(df["compound"] == compound) & (df["event"].isin(top_tracks))].copy()
    ref = (
        sub[sub["tyre_life"].between(2, 4)]
        .groupby(["year", "round", "driver", "stint"])["lap_time_s"]
        .median()
        .rename("ref_pace")
    )
    sub = sub.merge(ref, left_on=["year", "round", "driver", "stint"], right_index=True)
    sub["delta_s"] = sub["lap_time_s"] - sub["ref_pace"]

    for i, track in enumerate(top_tracks):
        ts = sub[sub["event"] == track]
        if ts.empty:
            continue
        med = ts.groupby("tyre_life")["delta_s"].median()
        med = med[med.index <= 40]
        if len(med) >= 5:
            ax.plot(
                med.index,
                med.values,
                label=short_name(track),
                color=cmap(i),
                linewidth=2.2,
            )
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_title(f"{compound}", fontsize=13)
    ax.set_ylabel("Median Δ vs laps 2-4 of stint (s)")
    ax.set_xlim(0, 40)
    ax.set_ylim(-3, 6)
    ax.legend(fontsize=9, loc="upper left", ncol=2)
    ax.grid(alpha=0.3)
axes[-1].set_xlabel("Tyre age (laps)")
fig.suptitle(
    "Per-track pace change by tyre age — raw (uncorrected for fuel / evolution)",
    fontsize=14,
)
fig.tight_layout()
fig.savefig(FIG_DIR / "stint_curves_by_track.png", dpi=140)
print(f"  wrote {FIG_DIR}/stint_curves_by_track.png")


# ---- Figure 4: the confound exposed ----
# Within each stint, the lap-time delta should rise with tyre age (deg) but ALSO
# fall with race progression (fuel burn + track evolution). Plot the median delta
# vs tyre age, faceted by which stint of the race (1st / 2nd / 3rd) the stint is.
# In a 1st stint: tyre age == race lap, so fuel-burn is large => downward bias.
# In a late stint: race has been going a while, fuel burn during the stint is smaller,
# so the curve should bend up more clearly.
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
stint_positions = [1, 2, 3]
for ax, sp in zip(axes, stint_positions):
    for compound in COMPOUND_ORDER:
        sub = df[(df["compound"] == compound) & (df["stint"] == sp)].copy()
        if sub.empty:
            continue
        ref = (
            sub[sub["tyre_life"].between(2, 4)]
            .groupby(["year", "round", "driver", "stint"])["lap_time_s"]
            .median()
            .rename("ref_pace")
        )
        sub = sub.merge(
            ref, left_on=["year", "round", "driver", "stint"], right_index=True
        )
        sub["delta_s"] = sub["lap_time_s"] - sub["ref_pace"]
        med = sub.groupby("tyre_life")["delta_s"].median()
        n = sub.groupby("tyre_life").size()
        # Only plot where we have >= 30 laps to avoid noise
        keep = (n >= 30) & (med.index <= 40)
        med = med[keep]
        ax.plot(
            med.index,
            med.values,
            color=COMPOUND_COLORS[compound],
            linewidth=2.5,
            label=f"{compound} (n={len(sub):,})",
        )
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_title(f"Stint #{sp} of race", fontsize=13)
    ax.set_xlabel("Tyre age (laps)")
    if ax is axes[0]:
        ax.set_ylabel("Median Δ vs laps 2-4 of stint (s)")
    ax.set_xlim(0, 40)
    ax.set_ylim(-3, 6)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
fig.suptitle(
    "Pace change by tyre age, faceted by stint position — exposes the fuel confound",
    fontsize=14,
)
fig.tight_layout()
fig.savefig(FIG_DIR / "fuel_confound.png", dpi=140)
print(f"  wrote {FIG_DIR}/fuel_confound.png")


# ---- Figure 3: sample size by compound and tyre age ----
fig, ax = plt.subplots(figsize=(10, 5))
for compound in COMPOUND_ORDER:
    sub = df[df["compound"] == compound]
    counts = sub.groupby("tyre_life").size()
    counts = counts[counts.index <= 50]
    ax.plot(
        counts.index,
        counts.values,
        label=compound,
        color=COMPOUND_COLORS[compound],
        linewidth=2,
    )
ax.set_xlabel("Tyre age (laps)")
ax.set_ylabel("Number of laps in dataset")
ax.set_title("Sample size by tyre age & compound — censoring shows up as the falloff")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "sample_sizes.png", dpi=120)
print(f"  wrote {FIG_DIR}/sample_sizes.png")

print("Done.")
