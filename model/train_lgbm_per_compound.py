"""Train one LightGBM per compound and compare to the joint model in train_lgbm.py.

Same target (`deg_s`), same 5-fold race-grouped CV, same conformal interval
construction — only the model partitioning differs. If a per-compound model
beats the joint model, that demonstrates real compound-specific interactions
that the joint model was under-fitting.

Saves: data/oof_predictions_per_compound.parquet
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

DATA = Path("data/laps_corrected.parquet")
OUT_OOF = Path("data/oof_predictions_per_compound.parquet")

N_FOLDS = 5
CALIB_FRAC = 0.20
ALPHA = 0.20
SEED = 12

FEATURES_NUMERIC = [
    "tyre_life",
    "tyre_life_sq",
    "stint",
    "starting_tyre_life",
    "air_temp",
    "track_temp",
    "humidity",
    "wind_speed",
]
FEATURES_CATEGORICAL = [
    "event",
    "team",
]  # compound dropped — we condition on it via partitioning


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["starting_tyre_life"] = df.groupby(["year", "round", "driver", "stint"])[
        "tyre_life"
    ].transform("min")
    df["tyre_life_sq"] = df["tyre_life"].astype(float) ** 2
    return df


def fit_one_compound(df_c: pd.DataFrame, compound_label: str) -> pd.DataFrame:
    """Run 5-fold CV on data restricted to one compound. Returns OOF rows."""
    df_c = df_c.reset_index(drop=True).copy()
    for c in FEATURES_CATEGORICAL:
        df_c[c] = df_c[c].astype("category")

    X = df_c[FEATURES_NUMERIC + FEATURES_CATEGORICAL]
    y = df_c["deg_s"].astype(float).values
    df_c["race_id"] = (
        df_c["year"].astype(str) + "_" + df_c["round"].astype(str).str.zfill(2)
    )
    groups = df_c["race_id"].values

    rng = np.random.default_rng(SEED)
    oof_pred = np.full(len(df_c), np.nan)
    oof_lower = np.full(len(df_c), np.nan)
    oof_upper = np.full(len(df_c), np.nan)
    fold_assignment = np.full(len(df_c), -1, dtype=int)

    gkf = GroupKFold(n_splits=N_FOLDS)
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        train_races = np.unique(groups[train_idx])
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

        # Smaller model since SOFT has only ~3k rows
        is_small = len(X_fit) < 4000
        model = lgb.LGBMRegressor(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=15 if is_small else 63,
            min_data_in_leaf=20 if is_small else 40,
            feature_fraction=0.9,
            bagging_fraction=0.9,
            bagging_freq=5,
            objective="regression_l1",
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

        calib_pred = model.predict(X_calib)
        calib_resid = np.abs(y_calib - calib_pred)
        n_c = len(calib_resid)
        q_alpha = float(
            np.quantile(calib_resid, np.ceil((n_c + 1) * (1 - ALPHA)) / n_c)
        )

        test_pred = model.predict(X_test)
        test_mae = float(np.mean(np.abs(y_test - test_pred)))
        coverage = float(np.mean(np.abs(y_test - test_pred) <= q_alpha))

        print(
            f"    [{compound_label}] fold {fold}: n_fit={len(X_fit):>5d} "
            f"q_alpha={q_alpha:.3f}s  test_MAE={test_mae:.3f}s  coverage={coverage:.1%}"
        )

        oof_pred[test_idx] = test_pred
        oof_lower[test_idx] = test_pred - q_alpha
        oof_upper[test_idx] = test_pred + q_alpha
        fold_assignment[test_idx] = fold

    out = df_c[
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
    return out


def main():
    df = pd.read_parquet(DATA)
    df = df[df["fit_ok"]].copy()
    df = build_features(df)

    needed = FEATURES_NUMERIC + FEATURES_CATEGORICAL + ["deg_s", "compound"]
    df = df.dropna(subset=needed).reset_index(drop=True)
    print(f"Total laps: {len(df):,}")

    frames = []
    for compound in ["SOFT", "MEDIUM", "HARD"]:
        df_c = df[df["compound"] == compound]
        print(f"\n  {compound}: {len(df_c):,} laps")
        frames.append(fit_one_compound(df_c, compound))

    all_oof = pd.concat(frames, ignore_index=True)
    all_oof.to_parquet(OUT_OOF, index=False)

    print()
    print("=" * 60)
    print("PER-COMPOUND MODEL RESULTS")
    print("=" * 60)
    overall_mae = float(np.mean(all_oof["abs_err"]))
    overall_cov = float(np.mean(all_oof["in_interval"]))
    overall_width = float(np.mean(all_oof["pred_upper"] - all_oof["pred_lower"]))
    print(f"  Overall OOF MAE:        {overall_mae:.3f} s")
    print(f"  80% coverage:           {overall_cov:.1%}")
    print(f"  80% interval width:     {overall_width:.3f} s")
    print()
    for c in ["SOFT", "MEDIUM", "HARD"]:
        sub = all_oof[all_oof["compound"] == c]
        m = float(sub["abs_err"].mean())
        cov = float(sub["in_interval"].mean())
        w = float((sub["pred_upper"] - sub["pred_lower"]).mean())
        print(
            f"  {c:<7s}  n={len(sub):>5d}  MAE={m:.3f}s  cov={cov:.1%}  width={w:.3f}s"
        )

    print(f"\nSaved to {OUT_OOF}")


if __name__ == "__main__":
    main()
