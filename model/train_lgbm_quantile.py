"""v1b — Conformalized Quantile Regression (CQR) for heteroskedastic intervals.

Three models per fold: point (MAE), lower quantile (alpha/2), upper quantile (1-alpha/2).
A separate calibration split computes CQR nonconformity scores:
    s_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i))
and inflates both bounds by the (1-alpha) quantile of those scores.

This gives adaptive interval widths (narrower early in stint, wider at high age)
with a finite-sample coverage guarantee — unlike the flat conformal width in
train_lgbm.py and unlike uncalibrated quantile regression.

Saves: data/oof_predictions_quantile.parquet
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

DATA = Path("data/laps_corrected.parquet")
OUT_OOF = Path("data/oof_predictions_quantile.parquet")

N_FOLDS = 5
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
    "air_temp",
    "track_temp",
    "humidity",
    "wind_speed",
]
FEATURES_CATEGORICAL = ["compound", "event", "team"]

LGB_COMMON = dict(
    n_estimators=800,
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=40,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=5,
    verbose=-1,
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["starting_tyre_life"] = df.groupby(["year", "round", "driver", "stint"])[
        "tyre_life"
    ].transform("min")
    df["tyre_life_sq"] = df["tyre_life"].astype(float) ** 2
    df["fresh_tyre"] = df["fresh_tyre"].astype(float)
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


def fit_quantile(X_fit, y_fit, X_val, y_val, alpha, seed):
    model = lgb.LGBMRegressor(
        **LGB_COMMON,
        objective="quantile",
        alpha=alpha,
        random_state=seed,
    )
    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_val, y_val)],
        categorical_feature=FEATURES_CATEGORICAL,
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    return model


def main():
    df = pd.read_parquet(DATA)
    df = df[df["fit_ok"]].copy()
    df = build_features(df)

    needed = FEATURES_NUMERIC + FEATURES_CATEGORICAL + ["deg_s"]
    df = df.dropna(subset=needed).reset_index(drop=True)
    print(
        f"Training on {len(df):,} laps across {df.groupby(['year','round']).ngroups} races"
    )

    for c in FEATURES_CATEGORICAL:
        df[c] = df[c].astype("category")

    X = df[FEATURES_NUMERIC + FEATURES_CATEGORICAL]
    y = df["deg_s"].astype(float).values

    df["race_id"] = df["year"].astype(str) + "_" + df["round"].astype(str).str.zfill(2)
    groups = df["race_id"].values

    rng = np.random.default_rng(SEED)
    oof_pred = np.full(len(df), np.nan)
    oof_lower = np.full(len(df), np.nan)
    oof_upper = np.full(len(df), np.nan)
    fold_assignment = np.full(len(df), -1, dtype=int)

    gkf = GroupKFold(n_splits=N_FOLDS)
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        train_races = np.unique(groups[train_idx])
        test_races = np.unique(groups[test_idx])

        # Split training races: 70% fit, 10% val (early stopping), 20% calib (CQR)
        rng.shuffle(train_races)
        n_calib = max(2, int(round(0.20 * len(train_races))))
        n_val = max(2, int(round(0.10 * len(train_races))))
        calib_races = set(train_races[:n_calib])
        val_races = set(train_races[n_calib : n_calib + n_val])
        fit_races = set(train_races[n_calib + n_val :])
        calib_mask = np.isin(groups, list(calib_races))
        val_mask = np.isin(groups, list(val_races))
        fit_mask = np.isin(groups, list(fit_races))

        X_fit = X.loc[fit_mask]
        y_fit = y[fit_mask]
        X_val = X.loc[val_mask]
        y_val = y[val_mask]
        X_calib = X.loc[calib_mask]
        y_calib = y[calib_mask]
        X_test = X.loc[test_idx]
        y_test = y[test_idx]

        m_mid = lgb.LGBMRegressor(
            **LGB_COMMON, objective="regression_l1", random_state=SEED + fold
        )
        m_mid.fit(
            X_fit,
            y_fit,
            eval_set=[(X_val, y_val)],
            categorical_feature=FEATURES_CATEGORICAL,
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
        m_lo = fit_quantile(X_fit, y_fit, X_val, y_val, ALPHA / 2, SEED + fold + 100)
        m_hi = fit_quantile(
            X_fit, y_fit, X_val, y_val, 1 - ALPHA / 2, SEED + fold + 200
        )

        # CQR calibration: nonconformity score = max(q_lo - y, y - q_hi)
        calib_lo = m_lo.predict(X_calib)
        calib_hi = m_hi.predict(X_calib)
        scores = np.maximum(calib_lo - y_calib, y_calib - calib_hi)
        n_c = len(scores)
        q_cqr = float(np.quantile(scores, np.ceil((n_c + 1) * (1 - ALPHA)) / n_c))

        pred = m_mid.predict(X_test)
        lower = m_lo.predict(X_test) - q_cqr
        upper = m_hi.predict(X_test) + q_cqr

        mae = float(np.mean(np.abs(y_test - pred)))
        cov = float(np.mean((y_test >= lower) & (y_test <= upper)))
        width = float(np.mean(upper - lower))
        print(
            f"  fold {fold}: n_test_races={len(test_races)}  "
            f"q_cqr={q_cqr:.3f}s  MAE={mae:.3f}s  cov={cov:.1%}  width={width:.3f}s"
        )

        oof_pred[test_idx] = pred
        oof_lower[test_idx] = lower
        oof_upper[test_idx] = upper
        fold_assignment[test_idx] = fold

    overall_mae = float(np.mean(np.abs(y - oof_pred)))
    overall_cov = float(np.mean((y >= oof_lower) & (y <= oof_upper)))
    overall_width = float(np.mean(oof_upper - oof_lower))
    baseline_mae = float(np.mean(np.abs(y)))

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
            "fresh_tyre",
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
    print(f"Saved to {OUT_OOF}")


if __name__ == "__main__":
    main()
