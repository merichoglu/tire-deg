"""Worked example: one race, predicted deg curves overlaid on actual stint data.

Uses the out-of-fold predictions from train_lgbm_quantile.py (CQR, the primary
model) so the predictions are genuinely held-out — the model never saw these
laps during training.

Race selection logic: pick the race in oof_predictions_quantile.parquet with
the most compound variety and enough laps per stint to tell a story. Falls back
to oof_predictions.parquet (conformal) if the quantile file is absent.

Output: figures/worked_example_<race>.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OOF_QUANTILE = Path("data/oof_predictions_quantile.parquet")
OOF_CONFORMAL = Path("data/oof_predictions.parquet")
OUT_DIR = Path("figures")

COMPOUND_COLOR = {"SOFT": "#e31a1c", "MEDIUM": "#ff7f00", "HARD": "#6a6a6a"}
COMPOUND_MARKER = {"SOFT": "o", "MEDIUM": "s", "HARD": "^"}


def _pick_race(df: pd.DataFrame) -> tuple[int, int, str]:
    """Return (year, round, event) for the race with the richest stint data."""
    summary = (
        df.groupby(["year", "round", "event"])
        .agg(
            n_compounds=("compound", "nunique"),
            n_laps=("tyre_life", "count"),
            max_age=("tyre_life", "max"),
        )
        .reset_index()
    )
    # Prefer races with 3 compounds, many laps, long stints
    summary["score"] = (
        summary["n_compounds"] * 300 + summary["n_laps"] + summary["max_age"] * 2
    )
    best = summary.sort_values("score", ascending=False).iloc[0]
    return int(best["year"]), int(best["round"]), str(best["event"])


def _stint_medians(sub: pd.DataFrame) -> pd.DataFrame:
    """Per-lap median of actual and predicted, grouped by compound + tyre_life."""
    return (
        sub.groupby(["compound", "tyre_life"])
        .agg(
            actual_med=("deg_s", "median"),
            pred_med=("pred", "median"),
            pred_lower_med=("pred_lower", "median"),
            pred_upper_med=("pred_upper", "median"),
            n=("deg_s", "count"),
        )
        .reset_index()
    )


def plot_race(df: pd.DataFrame, year: int, rnd: int, event: str) -> Path:
    sub = df[(df["year"] == year) & (df["round"] == rnd)].copy()
    compounds = [c for c in ["SOFT", "MEDIUM", "HARD"] if c in sub["compound"].values]

    mae = float(sub["abs_err"].mean())
    cov = float(sub["in_interval"].mean())

    fig, axes = plt.subplots(
        1, len(compounds), figsize=(5 * len(compounds), 5), sharey=True
    )
    if len(compounds) == 1:
        axes = [axes]

    fig.suptitle(
        f"{event} {year}  —  OOF predictions vs actual deg_s\n"
        f"MAE = {mae:.3f} s   80% coverage = {cov:.1%}",
        fontsize=12,
        y=1.01,
    )

    for ax, compound in zip(axes, compounds):
        csub = sub[sub["compound"] == compound].copy()
        medians = _stint_medians(csub)
        color = COMPOUND_COLOR[compound]

        # Scatter: individual lap actuals (light, small)
        ax.scatter(
            csub["tyre_life"],
            csub["deg_s"],
            color=color,
            alpha=0.15,
            s=12,
            zorder=1,
            label="Individual laps (actual)",
        )

        # Median actual line
        ax.plot(
            medians["tyre_life"],
            medians["actual_med"],
            color=color,
            linewidth=2,
            marker=COMPOUND_MARKER[compound],
            markersize=5,
            zorder=3,
            label="Actual deg (median)",
        )

        # Predicted median + 80% band
        ax.plot(
            medians["tyre_life"],
            medians["pred_med"],
            color="black",
            linewidth=1.8,
            linestyle="--",
            zorder=4,
            label="Predicted (median)",
        )
        ax.fill_between(
            medians["tyre_life"],
            medians["pred_lower_med"],
            medians["pred_upper_med"],
            color="black",
            alpha=0.12,
            zorder=2,
            label="80% prediction band",
        )

        compound_mae = float(csub["abs_err"].mean())
        compound_cov = float(csub["in_interval"].mean())
        ax.set_title(
            f"{compound}\nn={len(csub)}  MAE={compound_mae:.3f}s  cov={compound_cov:.1%}",
            fontsize=10,
        )
        ax.set_xlabel("Tyre age (laps)", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("deg_s (seconds above fresh pace)", fontsize=9)
        ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    slug = event.lower().replace(" ", "_").replace("grand_prix", "gp")
    out_path = OUT_DIR / f"worked_example_{year}_{slug}.png"
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _commentary(df: pd.DataFrame, year: int, rnd: int, event: str) -> None:
    sub = df[(df["year"] == year) & (df["round"] == rnd)].copy()
    print()
    print("=" * 60)
    print(f"WORKED EXAMPLE: {event} {year}")
    print("=" * 60)

    overall_mae = sub["abs_err"].mean()
    overall_cov = sub["in_interval"].mean()
    print(f"  Overall MAE:      {overall_mae:.3f} s")
    print(f"  80% coverage:     {overall_cov:.1%}")
    print()

    for compound in ["SOFT", "MEDIUM", "HARD"]:
        csub = sub[sub["compound"] == compound]
        if csub.empty:
            continue
        cmae = csub["abs_err"].mean()
        ccov = csub["in_interval"].mean()

        # Find the age where prediction error is largest (model disagrees most)
        worst_age = csub.loc[csub["abs_err"].idxmax(), "tyre_life"]
        worst_err = csub["abs_err"].max()

        # Trend direction: does predicted deg increase with age as expected?
        trend_actual = np.polyfit(csub["tyre_life"], csub["deg_s"], 1)[0]
        trend_pred = np.polyfit(csub["tyre_life"], csub["pred"], 1)[0]

        print(f"  {compound}:")
        print(f"    MAE={cmae:.3f}s  coverage={ccov:.1%}  n={len(csub)}")
        print(
            f"    Actual deg slope: {trend_actual:+.4f} s/lap  "
            f"Predicted: {trend_pred:+.4f} s/lap"
        )
        print(
            f"    Largest error at tyre_life={worst_age:.0f} laps "
            f"({worst_err:.3f}s)"
        )
        if abs(trend_actual - trend_pred) < 0.005:
            print("    Model captures the slope well.")
        elif trend_pred < trend_actual:
            print("    Model under-predicts degradation rate (too optimistic).")
        else:
            print("    Model over-predicts degradation rate (too pessimistic).")
        print()


def main():
    if OOF_QUANTILE.exists():
        df = pd.read_parquet(OOF_QUANTILE)
        model_label = "CQR (v1b)"
    elif OOF_CONFORMAL.exists():
        df = pd.read_parquet(OOF_CONFORMAL)
        model_label = "conformal (v1)"
    else:
        raise FileNotFoundError(
            "No OOF prediction file found. Run train_lgbm_quantile.py or "
            "train_lgbm.py first."
        )

    print(f"Using {model_label} OOF predictions ({len(df):,} laps)")

    year, rnd, event = _pick_race(df)
    print(f"Selected race: {event} {year} (round {rnd})")

    out_path = plot_race(df, year, rnd, event)
    print(f"Saved plot to {out_path}")

    _commentary(df, year, rnd, event)


if __name__ == "__main__":
    main()
