"""Temporal holdout eval: train on 2023-2024, test on 2025.

Tests whether the model generalises to a new season — the question any real
F1 team would ask. Uses the same features and LightGBM config as train_lgbm.py
but a single temporal split instead of 5-fold CV.

Saves: data/temporal_holdout.parquet
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

DATA = Path("data/laps_corrected.parquet")
FITS = Path("data/race_fits.parquet")
OUT = Path("data/temporal_holdout.parquet")

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


def main():
    df = pd.read_parquet(DATA)
    fits = pd.read_parquet(FITS)
    df = df[df["fit_ok"]].copy()
    df = build_features(df, fits)

    needed = FEATURES_NUMERIC + FEATURES_CATEGORICAL + ["deg_s"]
    df = df.dropna(subset=needed).reset_index(drop=True)

    train = df[df["year"] < 2025].copy()
    test = df[df["year"] == 2025].copy()
    print(
        f"Train: {len(train):,} laps, {train['year'].nunique()} seasons "
        f"({train.groupby(['year','round']).ngroups} races)"
    )
    print(
        f"Test:  {len(test):,} laps, 2025 "
        f"({test.groupby(['year','round']).ngroups} races)"
    )

    for c in FEATURES_CATEGORICAL:
        train[c] = train[c].astype("category")
        test[c] = test[c].astype("category")

    X_train = train[FEATURES_NUMERIC + FEATURES_CATEGORICAL]
    y_train = train["deg_s"].astype(float).values
    X_test = test[FEATURES_NUMERIC + FEATURES_CATEGORICAL]
    y_test = test["deg_s"].astype(float).values

    # Hold out last 20% of training races (by round order) for conformal calibration
    train_races = train.groupby(["year", "round"]).ngroups
    train["race_id"] = (
        train["year"].astype(str) + "_" + train["round"].astype(str).str.zfill(2)
    )
    sorted_races = sorted(train["race_id"].unique())
    n_calib = max(2, int(round(0.20 * len(sorted_races))))
    calib_race_ids = set(sorted_races[-n_calib:])
    fit_mask = ~train["race_id"].isin(calib_race_ids)
    calib_mask = train["race_id"].isin(calib_race_ids)

    X_fit = X_train.loc[fit_mask]
    y_fit = y_train[fit_mask]
    X_calib = X_train.loc[calib_mask]
    y_calib = y_train[calib_mask]

    print(
        f"\nFit races: {fit_mask.sum():,} laps  "
        f"Calib races: {calib_mask.sum():,} laps"
    )

    model = lgb.LGBMRegressor(
        n_estimators=800,
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

    calib_pred = model.predict(X_calib)
    calib_resid = np.abs(y_calib - calib_pred)
    n_c = len(calib_resid)
    q_alpha = float(np.quantile(calib_resid, np.ceil((n_c + 1) * (1 - ALPHA)) / n_c))

    test_pred = model.predict(X_test)
    test_lower = test_pred - q_alpha
    test_upper = test_pred + q_alpha

    mae = float(np.mean(np.abs(y_test - test_pred)))
    cov = float(np.mean((y_test >= test_lower) & (y_test <= test_upper)))
    width = float(np.mean(test_upper - test_lower))
    baseline_mae = float(np.mean(np.abs(y_test)))

    print()
    print("=" * 60)
    print("TEMPORAL HOLDOUT — train 2023-2024 → test 2025")
    print("=" * 60)
    print(f"  MAE:                   {mae:.3f} s")
    print(f"  Baseline (predict 0):  {baseline_mae:.3f} s")
    print(
        f"  Improvement:           {baseline_mae - mae:+.3f} s "
        f"({(baseline_mae - mae) / baseline_mae * 100:+.1f}%)"
    )
    print(f"  80% coverage:          {cov:.1%} (target 80%)")
    print(f"  80% interval width:    {width:.3f} s")
    print(f"  q_alpha:               {q_alpha:.3f} s")

    print()
    print("By compound:")
    out = test[
        [
            "year",
            "round",
            "event",
            "driver",
            "team",
            "compound",
            "tyre_life",
            "stint",
            "fresh_tyre",
            "is_street",
            "evo_swing",
            "air_temp",
            "track_temp",
            "deg_s",
        ]
    ].copy()
    out["pred"] = test_pred
    out["pred_lower"] = test_lower
    out["pred_upper"] = test_upper
    out["abs_err"] = np.abs(y_test - test_pred)
    out["in_interval"] = (y_test >= test_lower) & (y_test <= test_upper)

    for compound in ["SOFT", "MEDIUM", "HARD"]:
        sub = out[out["compound"] == compound]
        print(
            f"  {compound:<7s}  n={len(sub):>5d}  "
            f"MAE={sub['abs_err'].mean():.3f}s  "
            f"cov={sub['in_interval'].mean():.1%}"
        )

    # Per-track breakdown — flag tracks not seen in training
    train_tracks = set(train["event"].unique())
    print()
    print("By track (sorted by MAE):")
    by_track = (
        out.groupby("event")
        .apply(
            lambda x: pd.Series(
                {
                    "n": len(x),
                    "MAE": float(x["abs_err"].mean()),
                    "cov": float(x["in_interval"].mean()),
                    "mean_deg": float(x["deg_s"].mean()),
                    "new_track": x["event"].iloc[0] not in train_tracks,
                }
            )
        )
        .sort_values("MAE")
    )
    print(f"  {'event':<35s}  {'n':>5}  {'MAE':>6}  {'cov':>6}  {'new?':>4}")
    for ev, row in by_track.iterrows():
        flag = " *" if row["new_track"] else ""
        print(
            f"  {ev:<35s}  {int(row['n']):>5d}  "
            f"{row['MAE']:>6.3f}s  {row['cov']:>6.1%}{flag}"
        )
    if by_track["new_track"].any():
        print("  (* = track not seen in 2023-2024 training data)")

    out.to_parquet(OUT, index=False)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
