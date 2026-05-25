"""Evaluate the OOF predictions from train_lgbm.py.

Reads data/oof_predictions.parquet and produces:
  - Console tables: overall MAE, per-compound, per-age-bucket, per-track
  - figures/v1_calibration.png — coverage vs nominal level + interval width
  - figures/v1_pred_vs_actual.png — predicted vs actual deg scatter, per compound
  - figures/v1_mae_breakdown.png — bar charts of MAE by compound and age bucket
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OOF = Path("data/oof_predictions.parquet")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

df = pd.read_parquet(OOF)
print(f"Loaded {len(df):,} OOF predictions")


# Helpers
def mae(x):
    return float(np.mean(np.abs(x["deg_s"] - x["pred"])))


def cov(x):
    return float(np.mean(x["in_interval"]))


def width(x):
    return float(np.mean(x["pred_upper"] - x["pred_lower"]))


# ---- Overall numbers ----
print()
print("=" * 60)
print("OVERALL")
print("=" * 60)
overall_mae = mae(df)
overall_cov = cov(df)
overall_width = width(df)
baseline_mae = float(np.mean(np.abs(df["deg_s"])))
print(f"  MAE:                 {overall_mae:.3f} s")
print(f"  Baseline (predict 0): {baseline_mae:.3f} s")
print(f"  Improvement vs baseline: {(baseline_mae-overall_mae)/baseline_mae*100:.1f}%")
print(f"  80% coverage:        {overall_cov:.1%}")
print(f"  80% interval width:  {overall_width:.3f} s")

# ---- By compound ----
print()
print("=" * 60)
print("BY COMPOUND")
print("=" * 60)
by_comp = df.groupby("compound", observed=True).apply(
    lambda x: pd.Series(
        {
            "n": len(x),
            "MAE": mae(x),
            "coverage": cov(x),
            "width": width(x),
            "baseline_MAE": float(np.mean(np.abs(x["deg_s"]))),
        }
    )
)
print(by_comp.round(3).to_string())

# ---- By tyre age bucket ----
print()
print("=" * 60)
print("BY TYRE AGE BUCKET")
print("=" * 60)
df["age_bucket"] = pd.cut(
    df["tyre_life"], bins=[0, 5, 15, 25, 100], labels=["1-5", "6-15", "16-25", "26+"]
)
by_age = df.groupby("age_bucket", observed=True).apply(
    lambda x: pd.Series(
        {
            "n": len(x),
            "MAE": mae(x),
            "coverage": cov(x),
            "width": width(x),
            "mean_deg": float(np.mean(x["deg_s"])),
        }
    )
)
print(by_age.round(3).to_string())

# ---- By compound × age ----
print()
print("=" * 60)
print("BY COMPOUND × AGE BUCKET (MAE)")
print("=" * 60)
ca = (
    df.groupby(["compound", "age_bucket"], observed=True)
    .apply(lambda x: pd.Series({"n": len(x), "MAE": mae(x)}))
    .unstack("age_bucket")
)
print(ca.round(3).to_string())

# ---- Per track (top/bottom by MAE) ----
print()
print("=" * 60)
print("PER TRACK — best and worst 5 by MAE")
print("=" * 60)
by_track = (
    df.groupby("event")
    .apply(
        lambda x: pd.Series(
            {"n": len(x), "MAE": mae(x), "mean_deg": float(np.mean(x["deg_s"]))}
        )
    )
    .sort_values("MAE")
)
print("Best:")
print(by_track.head(5).round(3).to_string())
print("Worst:")
print(by_track.tail(5).round(3).to_string())


# ---- Figure 1: calibration curve ----
# Sweep nominal levels and measure empirical coverage
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
nominal_levels = np.linspace(0.05, 0.95, 19)
emp_cov = []
for level in nominal_levels:
    # symmetric interval: rescale the conformal width to this level
    # we have 80% intervals already, can't rescale exactly without retraining
    # so plot empirical |error| distribution vs uniform target
    threshold = np.quantile(np.abs(df["deg_s"] - df["pred"]), level)
    emp_cov.append(level)
# Instead: just plot the actual |error| CDF and the implied coverage curve
abs_err = np.abs(df["deg_s"] - df["pred"]).values
sorted_err = np.sort(abs_err)
empirical_cov = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
ax.plot(empirical_cov, empirical_cov, "k--", linewidth=1, label="perfect calibration")
# We achieved 78.3% coverage at the 80% nominal level
# Show this single point as marker
ax.scatter(
    [0.80],
    [overall_cov],
    s=80,
    color="C3",
    zorder=5,
    label=f"80% conformal: {overall_cov:.1%} empirical",
)
ax.plot([0.80, 0.80], [0, overall_cov], "C3:", linewidth=0.8)
ax.plot([0, 0.80], [overall_cov, overall_cov], "C3:", linewidth=0.8)
ax.set_xlabel("Nominal coverage")
ax.set_ylabel("Empirical coverage")
ax.set_title("Interval calibration")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend()
ax.grid(alpha=0.3)

# Right panel: |error| distribution
ax = axes[1]
ax.hist(abs_err, bins=60, color="C0", edgecolor="white")
q80 = np.quantile(abs_err, 0.80)
ax.axvline(q80, color="C3", linewidth=2, label=f"80% quantile = {q80:.2f}s")
ax.axvline(
    np.median(abs_err),
    color="C2",
    linewidth=2,
    label=f"median = {np.median(abs_err):.2f}s",
)
ax.set_xlabel("|Prediction error| (s)")
ax.set_ylabel("Number of laps")
ax.set_title("Distribution of absolute prediction errors")
ax.set_xlim(0, np.quantile(abs_err, 0.99))
ax.legend()
ax.grid(alpha=0.3)

fig.suptitle("v1 LightGBM — calibration and error distribution", fontsize=14)
fig.tight_layout()
fig.savefig(FIG_DIR / "v1_calibration.png", dpi=140)
print(f"\nwrote {FIG_DIR}/v1_calibration.png")


# ---- Figure 2: predicted vs actual, per compound ----
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
COLORS = {"SOFT": "#d62728", "MEDIUM": "#ffcc00", "HARD": "#666666"}
for ax, compound in zip(axes, ["SOFT", "MEDIUM", "HARD"]):
    sub = df[df["compound"] == compound]
    ax.scatter(sub["deg_s"], sub["pred"], s=4, alpha=0.15, color=COLORS[compound])
    lims = [-3, 6]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual deg (s)")
    if ax is axes[0]:
        ax.set_ylabel("Predicted deg (s)")
    ax.set_title(f"{compound}  (n={len(sub):,}, MAE={mae(sub):.3f}s)")
    ax.grid(alpha=0.3)
fig.suptitle("v1 LightGBM — predicted vs actual deg, per compound", fontsize=14)
fig.tight_layout()
fig.savefig(FIG_DIR / "v1_pred_vs_actual.png", dpi=140)
print(f"wrote {FIG_DIR}/v1_pred_vs_actual.png")


# ---- Figure 3: MAE breakdown bar charts ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
comps = list(by_comp.index)
mae_vals = by_comp["MAE"].values
base_vals = by_comp["baseline_MAE"].values
x = np.arange(len(comps))
ax.bar(x - 0.2, base_vals, 0.4, label="baseline (predict 0)", color="#bbbbbb")
ax.bar(x + 0.2, mae_vals, 0.4, label="LightGBM v1", color="C0")
ax.set_xticks(x)
ax.set_xticklabels(comps)
ax.set_ylabel("MAE (s)")
ax.set_title("MAE by compound")
ax.legend()
ax.grid(alpha=0.3, axis="y")

ax = axes[1]
ages = list(by_age.index)
ax.bar(range(len(ages)), by_age["MAE"].values, color="C1")
ax.set_xticks(range(len(ages)))
ax.set_xticklabels(ages)
ax.set_xlabel("Tyre age bucket (laps)")
ax.set_ylabel("MAE (s)")
ax.set_title("MAE by tyre age bucket")
for i, (n, m) in enumerate(zip(by_age["n"].values, by_age["MAE"].values)):
    ax.text(i, m + 0.02, f"n={int(n):,}", ha="center", fontsize=9)
ax.grid(alpha=0.3, axis="y")

fig.suptitle("v1 LightGBM — MAE breakdowns", fontsize=14)
fig.tight_layout()
fig.savefig(FIG_DIR / "v1_mae_breakdown.png", dpi=140)
print(f"wrote {FIG_DIR}/v1_mae_breakdown.png")

print("\nDone.")
