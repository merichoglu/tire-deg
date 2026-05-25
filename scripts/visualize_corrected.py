"""Plot the *corrected* pace_loss_s vs tyre age across the full dataset.

Uses data/laps_corrected.parquet (output of correct_all_races.py). Only laps
from races with fit_ok=True are included. The pace_loss_s column already has
fuel + track-evolution removed, so the remaining variation with tyre age IS
the tyre-degradation signal.

Figures written:
  pace_loss_by_compound.png  — overall deg curves per compound (median + IQR)
  pace_loss_by_track.png     — per-track deg curves (top tracks)
  pace_loss_by_temp.png      — sliced by track-temperature bin per compound
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path("data/laps_corrected.parquet")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

df = pd.read_parquet(DATA)
n_all = len(df)
df = df[df["fit_ok"]].copy()
print(
    f"Loaded {n_all:,} laps, kept {len(df):,} from fit_ok races "
    f"({df['event'].nunique()} events)"
)

# Express pace_loss relative to each stint's laps 2-4 so it represents
# in-stint degradation, not absolute pace level (which still encodes things
# like car speed / driver / weather).
ref = (
    df[df["tyre_life"].between(2, 4)]
    .groupby(["year", "round", "driver", "stint"])["pace_loss_s"]
    .mean()
    .rename("ref_paceloss")
    .reset_index()
)
df = df.merge(ref, how="left", on=["year", "round", "driver", "stint"])
df["deg_s"] = df["pace_loss_s"] - df["ref_paceloss"]
# Stints without a reference (e.g. very short SC stint) drop here:
df = df.dropna(subset=["deg_s"]).copy()
print(f"After stint-reference join: {len(df):,} laps")

COMPOUND_COLORS = {"SOFT": "#d62728", "MEDIUM": "#ffcc00", "HARD": "#666666"}
COMPOUND_ORDER = ["SOFT", "MEDIUM", "HARD"]


def median_iqr(values, x_col, y_col, x_max=45, min_n=10):
    grp = values.groupby(x_col)[y_col]
    med = grp.median()
    p25 = grp.quantile(0.25)
    p75 = grp.quantile(0.75)
    n = grp.size()
    keep = (n >= min_n) & (med.index <= x_max)
    return med[keep], p25[keep], p75[keep]


# ---- Figure 1: pace loss vs tyre age, per compound ----
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
for ax, compound in zip(axes, COMPOUND_ORDER):
    sub = df[df["compound"] == compound]
    if sub.empty:
        continue
    # Faint individual stints
    for _, stint in sub.groupby(["year", "round", "driver", "stint"]):
        if len(stint) < 5:
            continue
        ax.plot(
            stint["tyre_life"],
            stint["deg_s"],
            color=COMPOUND_COLORS[compound],
            alpha=0.04,
            linewidth=0.7,
        )
    med, p25, p75 = median_iqr(sub, "tyre_life", "deg_s", x_max=45)
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
    ax.set_ylim(-2, 5)
    ax.set_title(f"{compound}  (n={len(sub):,} laps)", fontsize=13)
    ax.set_xlabel("Tyre age (laps)")
    if ax is axes[0]:
        ax.set_ylabel("Corrected pace loss vs laps 2-4 of stint (s)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
fig.suptitle(
    "Corrected pace loss by tyre age — fuel + track-evolution removed (fit_ok races only)",
    fontsize=14,
)
fig.tight_layout()
fig.savefig(FIG_DIR / "pace_loss_by_compound.png", dpi=140)
print(f"  wrote {FIG_DIR}/pace_loss_by_compound.png")


# ---- Figure 2: per-track pace loss, top tracks ----
top_tracks = (
    df.groupby("event").size().sort_values(ascending=False).head(10).index.tolist()
)
short_name = lambda s: s.replace(" Grand Prix", "")

fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
cmap = plt.get_cmap("tab10")
for ax, compound in zip(axes, COMPOUND_ORDER):
    sub = df[(df["compound"] == compound) & (df["event"].isin(top_tracks))]
    for i, track in enumerate(top_tracks):
        ts = sub[sub["event"] == track]
        if ts.empty:
            continue
        med, _, _ = median_iqr(ts, "tyre_life", "deg_s", x_max=40, min_n=5)
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
    ax.set_ylabel("Median pace loss (s)")
    ax.set_xlim(0, 40)
    ax.set_ylim(-1, 4)
    ax.legend(fontsize=9, loc="upper left", ncol=2)
    ax.grid(alpha=0.3)
axes[-1].set_xlabel("Tyre age (laps)")
fig.suptitle(
    "Corrected pace loss by tyre age and track  (fit_ok races, multi-year)",
    fontsize=14,
)
fig.tight_layout()
fig.savefig(FIG_DIR / "pace_loss_by_track.png", dpi=140)
print(f"  wrote {FIG_DIR}/pace_loss_by_track.png")


# ---- Figure 3: pace loss vs tyre age, sliced by track temperature ----
# This is where the model will need to draw signal from later, so let's see
# if there's a visible temperature dependence.
TEMP_BINS = [(15, 28), (28, 38), (38, 60)]
TEMP_LABELS = ["Cool (<28°C)", "Warm (28–38°C)", "Hot (>38°C)"]
TEMP_COLORS = ["#2b8cbe", "#fdae61", "#d7191c"]

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
for ax, compound in zip(axes, COMPOUND_ORDER):
    sub = df[df["compound"] == compound]
    for (lo, hi), label, color in zip(TEMP_BINS, TEMP_LABELS, TEMP_COLORS):
        ts = sub[sub["track_temp"].between(lo, hi)]
        if ts.empty:
            continue
        med, _, _ = median_iqr(ts, "tyre_life", "deg_s", x_max=40)
        if len(med) >= 3:
            ax.plot(
                med.index,
                med.values,
                color=color,
                linewidth=2.5,
                label=f"{label}  (n={len(ts):,})",
            )
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_title(f"{compound}", fontsize=13)
    ax.set_xlabel("Tyre age (laps)")
    if ax is axes[0]:
        ax.set_ylabel("Median pace loss (s)")
    ax.set_xlim(0, 40)
    ax.set_ylim(-1, 4)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
fig.suptitle(
    "Corrected pace loss by tyre age, stratified by track-temperature bin",
    fontsize=14,
)
fig.tight_layout()
fig.savefig(FIG_DIR / "pace_loss_by_temp.png", dpi=140)
print(f"  wrote {FIG_DIR}/pace_loss_by_temp.png")

print("Done.")
