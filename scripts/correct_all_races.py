"""Fit the fuel + track-evolution + linear-deg regression on every race and
save a per-lap parquet with the corrected pace-loss labels.

For each race:
  1. Subtract a fixed fuel correction (0.056 s/lap of fuel left to burn)
  2. Fit  laptime ~ C(driver) + lap_n + lap_n2 + age:C(compound)
  3. Extract the fitted track-evolution component (lap_n + lap_n2 part only),
     centred so it's zero at the race's first kept lap.
  4. pace_loss_s = lap_time_fuel_corrected − track_evolution_s

Also save per-race diagnostics: R², residual RMSE, fitted track-evo shape,
fitted deg slopes per compound.

Output:
  data/laps_corrected.parquet   — per-lap, adds pace_loss_s + components
  data/race_fits.parquet        — one row per race, summary stats
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

DATA_PATH = Path("data/laps_clean.parquet")
OUT_LAPS = Path("data/laps_corrected.parquet")
OUT_FITS = Path("data/race_fits.parquet")
FUEL_SLOPE_S_PER_LAP = 0.056  # 0.035 s/kg * ~1.6 kg/lap, literature value


def fit_one_race(group: pd.DataFrame) -> tuple[pd.DataFrame, dict] | None:
    """Fit the model on one race. Return (per-lap rows with extra cols, race summary).
    Returns None if the race has too few laps / compounds to fit."""
    race_id = (
        int(group["year"].iloc[0]),
        int(group["round"].iloc[0]),
        str(group["event"].iloc[0]),
    )
    if len(group) < 100:
        return None
    if group["compound"].nunique() < 2:
        return None
    if group["driver"].nunique() < 5:
        return None

    g = group.copy().reset_index(drop=True)
    total_laps = int(g["lap_number"].max())
    g["fuel_corr_s"] = FUEL_SLOPE_S_PER_LAP * (total_laps - g["lap_number"])
    g["lap_time_fuel_corr"] = g["lap_time_s"] - g["fuel_corr_s"]
    g["age"] = g["tyre_life"].astype(float)
    g["lap_n"] = g["lap_number"].astype(float)
    g["lap_n2"] = g["lap_n"] ** 2

    try:
        model = smf.ols(
            "lap_time_fuel_corr ~ C(driver) + lap_n + lap_n2 + age:C(compound)",
            data=g,
        ).fit()
    except Exception as e:
        print(f"  ! {race_id} fit failed: {e}")
        return None

    beta_lap_n = model.params.get("lap_n", 0.0)
    beta_lap_n2 = model.params.get("lap_n2", 0.0)
    first_lap_val = g["lap_n"].min()
    evo_at_start = beta_lap_n * first_lap_val + beta_lap_n2 * (first_lap_val**2)
    g["track_evo_s"] = (
        beta_lap_n * g["lap_n"] + beta_lap_n2 * g["lap_n2"] - evo_at_start
    )
    # pace_loss = raw - fuel - track_evo
    g["pace_loss_s"] = g["lap_time_fuel_corr"] - g["track_evo_s"]
    g["fitted_lap_s"] = model.fittedvalues
    g["resid_s"] = g["lap_time_fuel_corr"] - g["fitted_lap_s"]

    # Race-level diagnostics
    deg_slopes = {}
    for c in ["SOFT", "MEDIUM", "HARD"]:
        key = f"age:C(compound)[{c}]"
        deg_slopes[c] = float(model.params.get(key, np.nan))

    # Track-evo curve summary: max swing and where it occurs
    lap_grid = np.arange(int(first_lap_val), total_laps + 1)
    evo_grid = beta_lap_n * lap_grid + beta_lap_n2 * lap_grid**2 - evo_at_start
    evo_min = float(evo_grid.min())
    evo_max = float(evo_grid.max())
    evo_end = float(evo_grid[-1])

    r2 = float(model.rsquared)
    resid_rmse = float(g["resid_s"].std())
    # Quality gate: R² >= 0.5 AND residual RMSE <= 1.0s. Bad-fit races usually had
    # red flags / SC chaos that the model can't account for.
    fit_ok = bool((r2 >= 0.5) and (resid_rmse <= 1.0))
    g["fit_ok"] = fit_ok

    summary = {
        "year": race_id[0],
        "round": race_id[1],
        "event": race_id[2],
        "n_laps": int(len(g)),
        "n_drivers": int(g["driver"].nunique()),
        "n_compounds": int(g["compound"].nunique()),
        "race_length": int(total_laps),
        "r2": r2,
        "resid_rmse": resid_rmse,
        "fit_ok": fit_ok,
        "beta_lap_n": float(beta_lap_n),
        "beta_lap_n2": float(beta_lap_n2),
        "evo_min": evo_min,
        "evo_max": evo_max,
        "evo_end": evo_end,
        "evo_swing": evo_max - evo_min,
        "deg_soft": deg_slopes["SOFT"],
        "deg_medium": deg_slopes["MEDIUM"],
        "deg_hard": deg_slopes["HARD"],
    }
    return g, summary


def main():
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df):,} laps")

    all_lap_frames = []
    all_summaries = []
    skipped = []
    for (year, round_no, event), group in df.groupby(["year", "round", "event"]):
        result = fit_one_race(group)
        if result is None:
            skipped.append((year, round_no, event))
            continue
        rows, summary = result
        all_lap_frames.append(rows)
        all_summaries.append(summary)

    laps_out = pd.concat(all_lap_frames, ignore_index=True)
    fits_out = pd.DataFrame(all_summaries)

    # Compute deg_s: per-stint pace loss relative to that stint's own fresh
    # reference (mean of pace_loss_s on laps 2-4 of the stint). This is the
    # actual modelling target — the tyre-degradation component, with car/driver/
    # race base pace removed.
    ref = (
        laps_out[laps_out["age"].between(2, 4)]
        .groupby(["year", "round", "driver", "stint"])["pace_loss_s"]
        .mean()
        .rename("stint_ref_pace")
        .reset_index()
    )
    laps_out = laps_out.merge(ref, how="left", on=["year", "round", "driver", "stint"])
    laps_out["deg_s"] = laps_out["pace_loss_s"] - laps_out["stint_ref_pace"]

    laps_out.to_parquet(OUT_LAPS, index=False)
    fits_out.to_parquet(OUT_FITS, index=False)

    print()
    print(f"Saved {len(laps_out):,} lap rows to {OUT_LAPS}")
    print(f"Saved {len(fits_out):,} race summaries to {OUT_FITS}")
    n_ok = int(fits_out["fit_ok"].sum())
    laps_ok = int(laps_out["fit_ok"].sum())
    print(
        f"fit_ok races:  {n_ok} / {len(fits_out)}  "
        f"({laps_ok:,} / {len(laps_out):,} laps)"
    )
    print("\nRaces flagged fit_ok=False:")
    print(
        fits_out[~fits_out["fit_ok"]]
        .sort_values("resid_rmse", ascending=False)[
            ["year", "event", "r2", "resid_rmse"]
        ]
        .to_string(index=False)
    )
    if skipped:
        print(f"Skipped {len(skipped)} races: {skipped}")
    print()
    print("Per-race fit quality summary:")
    print(
        fits_out[
            ["r2", "resid_rmse", "evo_swing", "deg_soft", "deg_medium", "deg_hard"]
        ]
        .describe()
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
