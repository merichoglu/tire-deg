"""Leave-one-track-out (LOTO) evaluation.

For each track in the dataset, trains on all other tracks and tests on the
held-out track. This measures how much the model degrades when it has never
seen a given circuit — the cross-track generalisation figure the spec requires.

The key output is per-track OOF MAE compared against the 5-fold in-distribution
MAE (0.605 s), quantifying how track-specific the model is.

Saves: data/loto_results.parquet
       figures/loto_mae_by_track.png
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path("data/laps_corrected.parquet")
FITS = Path("data/race_fits.parquet")
OUT_RESULTS = Path("data/loto_results.parquet")
OUT_FIG = Path("figures/loto_mae_by_track.png")

STREET_CIRCUITS = {
    "Azerbaijan Grand Prix",
    "Singapore Grand Prix",
    "Monaco Grand Prix",
    "Saudi Arabian Grand Prix",
    "Las Vegas Grand Prix",
    "Miami Grand Prix",
}

ALPHA = 0.20
SEED = 12

FEATURES_NUMERIC = [
    "tyre_life",
    "tyre_life_sq",
    "age_soft",
    "age_medium",
    "age_hard",
    "stint",
    "starting_tyre_life",
    "fresh_tyre",
    "is_street",
    "evo_swing",
    "air_temp",
    "track_temp",
    "humidity",
    "wind_speed",
]
FEATURES_CATEGORICAL = ["compound", "event", "team"]


def build_features(df: pd.DataFrame, fits: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["starting_tyre_life"] = df.groupby(["year", "round", "driver", "stint"])[
        "tyre_life"
    ].transform("min")
    df["tyre_life_sq"] = df["tyre_life"].astype(float) ** 2
    df["fresh_tyre"] = df["fresh_tyre"].astype(float)
    df["is_street"] = df["event"].isin(STREET_CIRCUITS).astype(float)
    df["age_soft"] = np.where(df["compound"] == "SOFT", df["tyre_life"], 0).astype(
        float
    )
    df["age_medium"] = np.where(df["compound"] == "MEDIUM", df["tyre_life"], 0).astype(
        float
    )
    df["age_hard"] = np.where(df["compound"] == "HARD", df["tyre_life"], 0).astype(
        float
    )
    evo = fits[["year", "round", "evo_swing"]].drop_duplicates()
    df = df.merge(evo, on=["year", "round"], how="left")
    return df


def _fit_model(X_fit, y_fit, X_calib, y_calib):
    model = lgb.LGBMRegressor(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=40,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        objective="regression_l1",
        verbose=-1,
        random_state=SEED,
    )
    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_calib, y_calib)],
        categorical_feature=FEATURES_CATEGORICAL,
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    return model


def main():
    df = pd.read_parquet(DATA)
    fits = pd.read_parquet(FITS)
    df = df[df["fit_ok"]].copy()
    df = build_features(df, fits)

    needed = FEATURES_NUMERIC + FEATURES_CATEGORICAL + ["deg_s"]
    df = df.dropna(subset=needed).reset_index(drop=True)

    tracks = sorted(df["event"].unique())
    print(f"LOTO evaluation across {len(tracks)} tracks, {len(df):,} laps total")
    print()

    rng = np.random.default_rng(SEED)
    rows = []

    for track in tracks:
        test_mask = df["event"] == track
        train_mask = ~test_mask

        train = df[train_mask].copy()
        test = df[test_mask].copy()

        # Re-encode categoricals so the held-out track is unknown to the model.
        # For the event column in the test set we fall back to the most common
        # training track (the Bayesian prior analogue for the gradient booster).
        fallback_track = train["event"].value_counts().idxmax()
        for c in FEATURES_CATEGORICAL:
            train[c] = train[c].astype("category")
        test_event_col = test["event"].copy()
        test["event"] = test["event"].map(
            lambda e: e if e in train["event"].cat.categories else fallback_track
        )
        for c in FEATURES_CATEGORICAL:
            test[c] = test[c].astype(
                pd.CategoricalDtype(categories=train[c].cat.categories)
            )

        # Hold out the last 20% of training races as conformal calibration set
        train["race_id"] = (
            train["year"].astype(str) + "_" + train["round"].astype(str).str.zfill(2)
        )
        sorted_races = sorted(train["race_id"].unique())
        n_calib = max(2, int(round(0.20 * len(sorted_races))))
        calib_ids = set(sorted_races[-n_calib:])
        fit_mask = ~train["race_id"].isin(calib_ids)
        calib_mask_train = train["race_id"].isin(calib_ids)

        X_fit = train.loc[fit_mask, FEATURES_NUMERIC + FEATURES_CATEGORICAL]
        y_fit = train.loc[fit_mask, "deg_s"].astype(float).values
        X_calib = train.loc[calib_mask_train, FEATURES_NUMERIC + FEATURES_CATEGORICAL]
        y_calib = train.loc[calib_mask_train, "deg_s"].astype(float).values
        X_test = test[FEATURES_NUMERIC + FEATURES_CATEGORICAL]
        y_test = test["deg_s"].astype(float).values

        model = _fit_model(X_fit, y_fit, X_calib, y_calib)

        calib_pred = model.predict(X_calib)
        calib_resid = np.abs(y_calib - calib_pred)
        n_c = len(calib_resid)
        q_alpha = float(
            np.quantile(calib_resid, np.ceil((n_c + 1) * (1 - ALPHA)) / n_c)
        )

        test_pred = model.predict(X_test)
        mae = float(np.mean(np.abs(y_test - test_pred)))
        cov = float(np.mean(np.abs(y_test - test_pred) <= q_alpha))
        baseline = float(np.mean(np.abs(y_test)))
        is_street = track in STREET_CIRCUITS

        rows.append(
            {
                "track": track,
                "n": len(test),
                "mae": mae,
                "baseline_mae": baseline,
                "coverage": cov,
                "q_alpha": q_alpha,
                "is_street": is_street,
            }
        )
        print(
            f"  {track:<40s}  n={len(test):>5d}  "
            f"MAE={mae:.3f}s  cov={cov:.1%}  baseline={baseline:.3f}s"
            + (" [street]" if is_street else "")
        )

    results = pd.DataFrame(rows).sort_values("mae")
    results.to_parquet(OUT_RESULTS, index=False)

    in_dist_mae = 0.605  # 5-fold CV headline from train_lgbm.py
    overall_loto_mae = results["mae"].mean()
    overall_loto_cov = results["coverage"].mean()

    print()
    print("=" * 60)
    print("LEAVE-ONE-TRACK-OUT SUMMARY")
    print("=" * 60)
    print(f"  In-distribution MAE (5-fold CV):  {in_dist_mae:.3f} s")
    print(f"  LOTO mean MAE (all tracks):        {overall_loto_mae:.3f} s")
    print(
        f"  MAE degradation:                  +{overall_loto_mae - in_dist_mae:.3f} s"
    )
    print(
        f"  LOTO mean MAE (non-street):        "
        f"{results[~results['is_street']]['mae'].mean():.3f} s"
    )
    print(
        f"  LOTO mean MAE (street circuits):   "
        f"{results[results['is_street']]['mae'].mean():.3f} s"
    )
    print(f"  Mean 80% coverage:                 {overall_loto_cov:.1%}")
    print()
    print("Best tracks (lowest LOTO MAE):")
    for _, row in results.head(5).iterrows():
        print(f"  {row['track']:<40s}  {row['mae']:.3f}s")
    print("Worst tracks (highest LOTO MAE):")
    for _, row in results.tail(5).sort_values("mae", ascending=False).iterrows():
        print(f"  {row['track']:<40s}  {row['mae']:.3f}s")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 7))
    results_plot = results.sort_values("mae", ascending=False)
    colors = ["#d62728" if s else "#1f77b4" for s in results_plot["is_street"]]
    bars = ax.barh(results_plot["track"], results_plot["mae"], color=colors, alpha=0.85)
    ax.axvline(
        in_dist_mae,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"In-distribution MAE ({in_dist_mae:.3f} s)",
    )
    ax.axvline(
        overall_loto_mae,
        color="grey",
        linestyle=":",
        linewidth=1.2,
        label=f"LOTO mean MAE ({overall_loto_mae:.3f} s)",
    )

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#1f77b4", alpha=0.85, label="Permanent circuit"),
        Patch(facecolor="#d62728", alpha=0.85, label="Street circuit"),
        plt.Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"In-distribution CV MAE ({in_dist_mae:.3f} s)",
        ),
        plt.Line2D(
            [0],
            [0],
            color="grey",
            linestyle=":",
            linewidth=1.2,
            label=f"LOTO mean MAE ({overall_loto_mae:.3f} s)",
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    ax.set_xlabel("MAE (seconds)", fontsize=11)
    ax.set_title("Leave-One-Track-Out MAE — cross-track generalisation", fontsize=12)
    ax.set_xlim(0, results["mae"].max() * 1.15)
    plt.tight_layout()
    OUT_FIG.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150)
    plt.close(fig)
    print(f"Saved figure to {OUT_FIG}")
    print(f"Saved results to {OUT_RESULTS}")


if __name__ == "__main__":
    main()
