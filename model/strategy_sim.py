"""Race strategy simulator.

Trains a final LightGBM model on all available data, then compares tyre
strategies by predicting cumulative pace loss (deg_s) per stint.

Usage:
    python model/strategy_sim.py

Or import and call simulate() directly:

    from model.strategy_sim import load_model, simulate, TRACK_DEFAULTS
    model, meta = load_model()
    result = simulate(
        model, meta,
        track="British Grand Prix",
        team="Mercedes",
        stints=[("SOFT", 20), ("HARD", 38)],
    )
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

DATA = Path("data/laps_corrected.parquet")
FITS = Path("data/race_fits.parquet")
MODEL_OUT = Path("data/strategy_model.txt")
META_OUT = Path("data/strategy_meta.parquet")

SEED = 12
PIT_LOSS_S = 22.0  # typical stationary + out-lap delta vs flying lap

STREET_CIRCUITS = {
    "Azerbaijan Grand Prix",
    "Singapore Grand Prix",
    "Monaco Grand Prix",
    "Saudi Arabian Grand Prix",
    "Las Vegas Grand Prix",
    "Miami Grand Prix",
}

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


def _build_features(df: pd.DataFrame, fits: pd.DataFrame) -> pd.DataFrame:
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


def _track_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """Median weather per track from historical data."""
    return (
        df.groupby("event")[["track_temp", "air_temp", "humidity", "wind_speed"]]
        .median()
        .round(1)
    )


def load_model(force_retrain: bool = False) -> tuple[lgb.LGBMRegressor, dict]:
    """Load or train the final model on all data."""
    if MODEL_OUT.exists() and META_OUT.exists() and not force_retrain:
        model = lgb.LGBMRegressor()
        model._Booster = lgb.Booster(model_file=str(MODEL_OUT))

        meta_df = pd.read_parquet(META_OUT)
        meta = {
            "tracks": meta_df["tracks"].dropna().tolist(),
            "teams": meta_df["teams"].dropna().tolist(),
            "track_defaults": _track_defaults(
                pd.read_parquet(DATA)[lambda d: d["fit_ok"]]
            ),
            "evo_swing_by_track": dict(
                zip(meta_df["evo_track"].dropna(), meta_df["evo_swing"].dropna())
            ),
        }
        return model, meta

    print("Training final model on all data...")
    df = pd.read_parquet(DATA)
    fits = pd.read_parquet(FITS)
    df = df[df["fit_ok"]].copy()
    df = _build_features(df, fits)

    needed = FEATURES_NUMERIC + FEATURES_CATEGORICAL + ["deg_s"]
    df = df.dropna(subset=needed).reset_index(drop=True)

    for c in FEATURES_CATEGORICAL:
        df[c] = df[c].astype("category")

    X = df[FEATURES_NUMERIC + FEATURES_CATEGORICAL]
    y = df["deg_s"].astype(float).values

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
    model.fit(X, y, categorical_feature=FEATURES_CATEGORICAL)
    model.booster_.save_model(str(MODEL_OUT))

    track_defaults = _track_defaults(df)

    # Median evo_swing per track
    evo_swing_by_track = fits.groupby("event")["evo_swing"].median().round(3).to_dict()

    # Save metadata
    tracks = sorted(df["event"].cat.categories.tolist())
    teams = sorted(df["team"].cat.categories.tolist())
    max_len = max(len(tracks), len(teams), len(evo_swing_by_track))
    meta_df = pd.DataFrame(
        {
            "tracks": pd.Series(tracks),
            "teams": pd.Series(teams),
            "evo_track": pd.Series(list(evo_swing_by_track.keys())),
            "evo_swing": pd.Series(list(evo_swing_by_track.values())),
        }
    )
    meta_df.to_parquet(META_OUT, index=False)

    meta = {
        "tracks": tracks,
        "teams": teams,
        "track_defaults": track_defaults,
        "evo_swing_by_track": evo_swing_by_track,
    }
    print(
        f"  trained on {len(df):,} laps across {df.groupby(['year','round']).ngroups} races"
    )
    return model, meta


def simulate(
    model: lgb.LGBMRegressor,
    meta: dict,
    track: str,
    team: str,
    stints: list[tuple[str, int]],
    track_temp: float | None = None,
    air_temp: float | None = None,
    humidity: float | None = None,
    wind_speed: float | None = None,
    fresh_tyre: bool = True,
) -> pd.DataFrame:
    """Predict lap-by-lap deg for a sequence of stints.

    Parameters
    ----------
    stints : list of (compound, n_laps), e.g. [("SOFT", 20), ("HARD", 38)]
    fresh_tyre : whether each stint starts on a new set (True) or used (False)

    Returns a DataFrame with one row per lap, plus a per-stint summary.
    """
    defaults = meta["track_defaults"]
    if track in defaults.index:
        row = defaults.loc[track]
        track_temp = track_temp if track_temp is not None else row["track_temp"]
        air_temp = air_temp if air_temp is not None else row["air_temp"]
        humidity = humidity if humidity is not None else row["humidity"]
        wind_speed = wind_speed if wind_speed is not None else row["wind_speed"]
    else:
        track_temp = track_temp or 35.0
        air_temp = air_temp or 25.0
        humidity = humidity or 50.0
        wind_speed = wind_speed or 1.5

    evo_swing = meta["evo_swing_by_track"].get(
        track, meta["track_defaults"]["track_temp"].median() * 0.02
    )

    is_street = float(track in STREET_CIRCUITS)
    is_known_track = track in meta["tracks"]
    is_known_team = team in meta["teams"]

    rows = []
    for stint_num, (compound, n_laps) in enumerate(stints, start=1):
        for lap in range(1, n_laps + 1):
            tyre_life = lap
            rows.append(
                {
                    "stint": stint_num,
                    "compound": compound,
                    "tyre_life": float(tyre_life),
                    "tyre_life_sq": float(tyre_life**2),
                    "age_soft": float(tyre_life) if compound == "SOFT" else 0.0,
                    "age_medium": float(tyre_life) if compound == "MEDIUM" else 0.0,
                    "age_hard": float(tyre_life) if compound == "HARD" else 0.0,
                    "starting_tyre_life": 1.0,
                    "fresh_tyre": float(fresh_tyre),
                    "is_street": is_street,
                    "evo_swing": evo_swing,
                    "air_temp": air_temp,
                    "track_temp": track_temp,
                    "humidity": humidity,
                    "wind_speed": wind_speed,
                    "event": track if is_known_track else meta["tracks"][0],
                    "team": team if is_known_team else meta["teams"][0],
                }
            )

    laps_df = pd.DataFrame(rows)
    for c in FEATURES_CATEGORICAL:
        laps_df[c] = laps_df[c].astype("category")

    X = laps_df[FEATURES_NUMERIC + FEATURES_CATEGORICAL]
    laps_df["pred_deg_s"] = model.booster_.predict(X)

    # Restore correct event/team labels for display
    laps_df["event"] = track
    laps_df["team"] = team

    return laps_df


def _summarise(laps: pd.DataFrame, stints: list[tuple[str, int]]) -> None:
    """Print a strategy comparison table."""
    print(
        f"  {'Stint':<6} {'Compound':<8} {'Laps':<6} {'Avg deg/lap':<14} {'Total deg':<12} {'Peak deg'}"
    )
    print("  " + "-" * 62)
    total = 0.0
    for i, (compound, n_laps) in enumerate(stints, start=1):
        sub = laps[laps["stint"] == i]
        avg = sub["pred_deg_s"].mean()
        total_deg = sub["pred_deg_s"].sum()
        peak = sub["pred_deg_s"].max()
        total += total_deg
        print(
            f"  {i:<6} {compound:<8} {n_laps:<6} {avg:>+.3f}s/lap      "
            f"{total_deg:>+.2f}s       {peak:>+.3f}s"
        )
    print("  " + "-" * 62)
    print(
        f"  {'TOTAL':<6} {'':<8} {sum(n for _, n in stints):<6} {'':14} {total:>+.2f}s"
    )


def main():
    model, meta = load_model()

    print()
    print("Available tracks:")
    for i, t in enumerate(meta["tracks"], 1):
        print(f"  {i:>2}. {t}")

    print()
    print("=" * 70)
    print("STRATEGY COMPARISON — British Grand Prix, Mercedes")
    print("  Conditions: historical medians")
    print("=" * 70)

    strategies = {
        "1-stop: MEDIUM → HARD": [("MEDIUM", 28), ("HARD", 30)],
        "1-stop: HARD → MEDIUM": [("HARD", 28), ("MEDIUM", 30)],
        "2-stop: SOFT → MEDIUM → HARD": [("SOFT", 15), ("MEDIUM", 20), ("HARD", 23)],
        "2-stop: MEDIUM → HARD → SOFT": [("MEDIUM", 20), ("HARD", 20), ("SOFT", 18)],
    }

    results = []
    for name, stints in strategies.items():
        laps = simulate(model, meta, "British Grand Prix", "Mercedes", stints)
        total_deg = laps["pred_deg_s"].sum()
        n_pits = len(stints) - 1
        pit_penalty = n_pits * PIT_LOSS_S
        print(f"\n{name}")
        _summarise(laps, stints)
        print(f"  Pit stops: {n_pits}  Pit time loss: +{pit_penalty:.0f}s")
        print(f"  Total tyre deg + pit loss: {total_deg + pit_penalty:+.2f}s")
        results.append((name, total_deg, pit_penalty, total_deg + pit_penalty))

    print()
    print("=" * 70)
    print("RANKING (lower = less total time lost)")
    print("=" * 70)
    results.sort(key=lambda x: x[3])
    for rank, (name, deg, pit, total) in enumerate(results, 1):
        print(f"  {rank}. {name}")
        print(f"     deg={deg:+.2f}s  pits={pit:+.0f}s  total={total:+.2f}s")

    print()
    print("=" * 70)
    print("STRATEGY COMPARISON — Singapore Grand Prix, Ferrari")
    print("  (street circuit — wider uncertainty)")
    print("=" * 70)

    sg_strategies = {
        "1-stop: MEDIUM → HARD": [("MEDIUM", 30), ("HARD", 32)],
        "2-stop: SOFT → MEDIUM → HARD": [("SOFT", 18), ("MEDIUM", 20), ("HARD", 24)],
    }
    sg_results = []
    for name, stints in sg_strategies.items():
        laps = simulate(model, meta, "Singapore Grand Prix", "Ferrari", stints)
        total_deg = laps["pred_deg_s"].sum()
        n_pits = len(stints) - 1
        pit_penalty = n_pits * PIT_LOSS_S
        print(f"\n{name}")
        _summarise(laps, stints)
        print(f"  Pit stops: {n_pits}  Pit time loss: +{pit_penalty:.0f}s")
        print(f"  Total tyre deg + pit loss: {total_deg + pit_penalty:+.2f}s")
        sg_results.append((name, total_deg, pit_penalty, total_deg + pit_penalty))

    print()
    print("Ranking:")
    sg_results.sort(key=lambda x: x[3])
    for rank, (name, deg, pit, total) in enumerate(sg_results, 1):
        print(
            f"  {rank}. {name}  (deg={deg:+.2f}s  pits={pit:+.0f}s  total={total:+.2f}s)"
        )


if __name__ == "__main__":
    main()
