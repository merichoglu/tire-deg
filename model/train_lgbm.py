"""Train a LightGBM regressor for tyre pace_loss with split-conformal intervals.

Pipeline per fold:
  1. Split races into 5 groups; one group is the test fold.
  2. From the training races, hold out 20% (by stint, not by lap) as the
     calibration set for conformal intervals.
  3. Train LightGBM point regressor on the remaining 80% of training races.
  4. Compute absolute residuals on the calibration set; take the (1 - alpha)
     quantile q_alpha. Prediction interval for any new x:
        [yhat - q_alpha,  yhat + q_alpha]
     This is symmetric split-conformal — coverage guarantee 1 - alpha on
     exchangeable data.

We save out-of-fold (oof) predictions per lap to data/oof_predictions.parquet,
plus per-fold q_alpha values to data/conformal_q.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

DATA = Path("data/laps_corrected.parquet")
OUT_OOF = Path("data/oof_predictions.parquet")
OUT_Q = Path("data/conformal_q.json")

N_FOLDS = 5
CALIB_FRAC = 0.20  # of training races, by stint
ALPHA = 0.20  # nominal coverage target: 1 - alpha = 80%
SEED = 12

FEATURES_NUMERIC = [
    "tyre_life",
    "tyre_life_sq",
    "age_soft",
    "age_medium",
    "age_hard",
    "stint",
    "starting_tyre_life",
    "air_temp",
    "track_temp",
    "humidity",
    "wind_speed",
]
FEATURES_CATEGORICAL = ["compound", "event", "team"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features. starting_tyre_life captures used-set effects;
    age_soft/medium/hard are compound-specific tyre_life splines so the trees can
    learn distinct deg curves per compound without relying on the categorical
    feature interacting via path-splits."""
    df = df.copy()
    df["starting_tyre_life"] = df.groupby(["year", "round", "driver", "stint"])[
        "tyre_life"
    ].transform("min")
    df["tyre_life_sq"] = df["tyre_life"].astype(float) ** 2
    df["age_soft"] = np.where(df["compound"] == "SOFT", df["tyre_life"], 0).astype(
        float
    )
    df["age_medium"] = np.where(df["compound"] == "MEDIUM", df["tyre_life"], 0).astype(
        float
    )
    df["age_hard"] = np.where(df["compound"] == "HARD", df["tyre_life"], 0).astype(
        float
    )
    return df


def main():
    df = pd.read_parquet(DATA)
    # Only train on races where the correction is reliable.
    df = df[df["fit_ok"]].copy()
    df = build_features(df)

    # Drop any rows missing required features (deg_s is NaN when the stint had no
    # lap in age 2-4 to form a fresh-pace reference)
    needed = FEATURES_NUMERIC + FEATURES_CATEGORICAL + ["deg_s"]
    df = df.dropna(subset=needed).reset_index(drop=True)
    print(
        f"Training on {len(df):,} laps across {df.groupby(['year','round']).ngroups} races"
    )

    # Convert categoricals to pandas category dtype — LightGBM uses these natively
    for c in FEATURES_CATEGORICAL:
        df[c] = df[c].astype("category")

    X = df[FEATURES_NUMERIC + FEATURES_CATEGORICAL]
    y = df["deg_s"].astype(float).values

    # Race-level grouping for K-fold split
    df["race_id"] = df["year"].astype(str) + "_" + df["round"].astype(str).str.zfill(2)
    groups = df["race_id"].values

    rng = np.random.default_rng(SEED)
    oof_pred = np.full(len(df), np.nan)
    oof_lower = np.full(len(df), np.nan)
    oof_upper = np.full(len(df), np.nan)
    fold_assignment = np.full(len(df), -1, dtype=int)
    q_alphas = []
    importance_rows = []  # gain importance per fold

    gkf = GroupKFold(n_splits=N_FOLDS)
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        train_races = np.unique(groups[train_idx])
        test_races = np.unique(groups[test_idx])
        # Of train races, peel off CALIB_FRAC as the conformal calibration set.
        rng.shuffle(train_races)
        n_calib = max(2, int(round(CALIB_FRAC * len(train_races))))
        calib_races = set(train_races[:n_calib])
        fit_races = set(train_races[n_calib:])
        calib_mask = np.isin(groups, list(calib_races))
        fit_mask = np.isin(groups, list(fit_races))

        X_fit = X.loc[fit_mask]
        y_fit = y[fit_mask]
        X_calib = X.loc[calib_mask]
        y_calib = y[calib_mask]
        X_test = X.loc[test_idx]
        y_test = y[test_idx]

        model = lgb.LGBMRegressor(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=63,
            min_data_in_leaf=40,
            feature_fraction=0.9,
            bagging_fraction=0.9,
            bagging_freq=5,
            objective="regression_l1",  # MAE - robust to deg-curve outliers
            verbose=-1,
            random_state=SEED + fold,
        )
        model.fit(
            X_fit,
            y_fit,
            eval_set=[(X_calib, y_calib)],
            categorical_feature=FEATURES_CATEGORICAL,
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )

        # Conformal calibration
        calib_pred = model.predict(X_calib)
        calib_resid = np.abs(y_calib - calib_pred)
        # Use the finite-sample-corrected quantile (Lei et al. 2018)
        n_c = len(calib_resid)
        q_alpha = float(
            np.quantile(calib_resid, np.ceil((n_c + 1) * (1 - ALPHA)) / n_c)
        )
        q_alphas.append(q_alpha)

        test_pred = model.predict(X_test)
        test_mae = float(np.mean(np.abs(y_test - test_pred)))
        coverage = float(np.mean(np.abs(y_test - test_pred) <= q_alpha))

        print(
            f"  fold {fold}: n_train_races={len(fit_races)}, "
            f"n_calib_races={len(calib_races)}, n_test_races={len(test_races)}  "
            f"q_alpha={q_alpha:.3f}s  test_MAE={test_mae:.3f}s  "
            f"coverage={coverage:.1%}"
        )

        # Record gain importance for this fold
        for feat, gain in zip(
            model.feature_name_,
            model.booster_.feature_importance(importance_type="gain"),
        ):
            importance_rows.append({"fold": fold, "feature": feat, "gain": float(gain)})

        oof_pred[test_idx] = test_pred
        oof_lower[test_idx] = test_pred - q_alpha
        oof_upper[test_idx] = test_pred + q_alpha
        fold_assignment[test_idx] = fold

    # Overall metrics
    overall_mae = float(np.mean(np.abs(y - oof_pred)))
    overall_cov = float(np.mean((y >= oof_lower) & (y <= oof_upper)))
    overall_width = float(np.mean(oof_upper - oof_lower))

    # Baseline: predict 0 (fresh tyre always)
    baseline_mae = float(np.mean(np.abs(y - 0.0)))

    print()
    print("=" * 50)
    print(f"Overall OOF MAE:        {overall_mae:.3f} s")
    print(f"Baseline (predict 0):   {baseline_mae:.3f} s")
    print(
        f"Improvement:            {baseline_mae - overall_mae:+.3f} s "
        f"({(baseline_mae - overall_mae) / baseline_mae * 100:+.1f}%)"
    )
    print(f"80% interval coverage:  {overall_cov:.1%} (target 80%)")
    print(f"80% interval width:     {overall_width:.3f} s")
    print()

    # Save oof predictions
    out = df[
        [
            "year",
            "round",
            "event",
            "driver",
            "team",
            "compound",
            "tyre_life",
            "stint",
            "starting_tyre_life",
            "air_temp",
            "track_temp",
            "humidity",
            "wind_speed",
            "deg_s",
        ]
    ].copy()
    out["pred"] = oof_pred
    out["pred_lower"] = oof_lower
    out["pred_upper"] = oof_upper
    out["fold"] = fold_assignment
    out["abs_err"] = np.abs(y - oof_pred)
    out["in_interval"] = (y >= oof_lower) & (y <= oof_upper)
    out.to_parquet(OUT_OOF, index=False)
    print(f"Saved oof predictions to {OUT_OOF}")

    with open(OUT_Q, "w") as f:
        json.dump({"alpha": ALPHA, "q_alphas": q_alphas}, f, indent=2)
    print(f"Saved conformal quantiles to {OUT_Q}")

    imp_df = pd.DataFrame(importance_rows)
    imp_df.to_parquet("data/feature_importance.parquet", index=False)
    print("Saved feature importance to data/feature_importance.parquet")


if __name__ == "__main__":
    main()
