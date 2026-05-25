"""Plot LightGBM gain importance across the 5 folds, with error bars."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

imp = pd.read_parquet("data/feature_importance.parquet")
FIG_DIR = Path("figures")

stats = (
    imp.groupby("feature")["gain"]
    .agg(["mean", "std"])
    .sort_values("mean", ascending=True)
)
total = stats["mean"].sum()
stats["pct"] = stats["mean"] / total * 100

fig, ax = plt.subplots(figsize=(10, 6))
y = np.arange(len(stats))
ax.barh(y, stats["mean"], xerr=stats["std"], color="C0", alpha=0.85)
ax.set_yticks(y)
ax.set_yticklabels(stats.index)
ax.set_xlabel("Mean gain importance across 5 folds (± std)")
ax.set_title("v1 LightGBM — feature importance")
for yi, (mean, pct) in enumerate(zip(stats["mean"], stats["pct"])):
    ax.text(mean, yi, f"  {pct:.1f}%", va="center", fontsize=9)
ax.grid(alpha=0.3, axis="x")
fig.tight_layout()
fig.savefig(FIG_DIR / "v1_feature_importance.png", dpi=140)
print(f"wrote {FIG_DIR}/v1_feature_importance.png")
