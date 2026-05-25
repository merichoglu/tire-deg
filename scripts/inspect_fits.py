"""Inspect the per-race fit quality from data/race_fits.parquet.

Reports the worst-fitting races by several criteria and produces a 2x2 figure:
  - R² histogram
  - Residual RMSE histogram
  - Scatter: R² vs RMSE (which races look bad on both metrics)
  - Per-compound deg slope distributions
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FITS = pd.read_parquet("data/race_fits.parquet")
FIG_DIR = Path("figures")

print(f"Loaded {len(FITS)} race fits")

# ---- Tables ----
print("\n== Worst R² (10 races) ==")
print(
    FITS.nsmallest(10, "r2")[
        [
            "year",
            "event",
            "r2",
            "resid_rmse",
            "evo_swing",
            "deg_soft",
            "deg_medium",
            "deg_hard",
        ]
    ].to_string(index=False)
)

print("\n== Worst residual RMSE (10 races) ==")
print(
    FITS.nlargest(10, "resid_rmse")[
        ["year", "event", "r2", "resid_rmse", "evo_swing", "n_laps"]
    ].to_string(index=False)
)

print("\n== Largest track-evolution swing (10 races) ==")
print(
    FITS.nlargest(10, "evo_swing")[
        ["year", "event", "evo_swing", "evo_min", "evo_max", "evo_end", "r2"]
    ].to_string(index=False)
)

print("\n== Negative deg slopes (suspicious) ==")
neg = FITS[(FITS["deg_soft"] < 0) | (FITS["deg_medium"] < 0) | (FITS["deg_hard"] < 0)]
print(
    neg[["year", "event", "r2", "resid_rmse", "deg_soft", "deg_medium", "deg_hard"]]
    .sort_values("r2")
    .to_string(index=False)
)


# ---- Plots ----
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
ax.hist(FITS["r2"], bins=20, color="C0", edgecolor="white")
ax.set_xlabel("R²")
ax.set_ylabel("Number of races")
ax.set_title(f"R² distribution  (median={FITS['r2'].median():.3f})")
ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.hist(FITS["resid_rmse"], bins=np.linspace(0, 6, 25), color="C1", edgecolor="white")
ax.set_xlabel("Residual RMSE (s)")
ax.set_ylabel("Number of races")
ax.set_title(f"Residual RMSE  (median={FITS['resid_rmse'].median():.2f}s)")
ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.scatter(FITS["r2"], FITS["resid_rmse"], alpha=0.6)
# Annotate the worst points
worst = FITS.nlargest(5, "resid_rmse")
for _, r in worst.iterrows():
    ax.annotate(
        f"{r['event'].split()[0][:8]} {str(r['year'])[-2:]}",
        (r["r2"], r["resid_rmse"]),
        fontsize=8,
        alpha=0.8,
    )
ax.set_xlabel("R²")
ax.set_ylabel("Residual RMSE (s)")
ax.set_title("R² vs Residual RMSE per race (worst outliers labelled)")
ax.grid(alpha=0.3)

ax = axes[1, 1]
deg_cols = ["deg_soft", "deg_medium", "deg_hard"]
colors = ["#d62728", "#ffcc00", "#999999"]
for col, color in zip(deg_cols, colors):
    vals = FITS[col].dropna()
    ax.hist(
        vals,
        bins=20,
        alpha=0.5,
        label=col.replace("deg_", "").upper(),
        color=color,
        edgecolor="black",
        linewidth=0.5,
    )
ax.axvline(0, color="k", linewidth=0.8)
ax.set_xlabel("Linear deg slope (s/lap of tyre age)")
ax.set_ylabel("Number of races")
ax.set_title("Per-race fitted deg slopes by compound")
ax.legend()
ax.grid(alpha=0.3)

fig.suptitle("Per-race fit quality dashboard", fontsize=15)
fig.tight_layout()
fig.savefig(FIG_DIR / "fit_quality_dashboard.png", dpi=140)
print(f"\n  wrote {FIG_DIR}/fit_quality_dashboard.png")
