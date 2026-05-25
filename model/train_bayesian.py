"""v2 — hierarchical Bayesian deg model in PyMC, with race-grouped 5-fold CV.

Model:
  beta[track, compound]  ~ HalfNormal(mu_beta[compound], sigma_beta[compound])
  alpha[track, compound] ~ Normal(0, sigma_alpha)
  delta[team]            ~ Normal(0, sigma_team)              # team age modifier
  gamma[compound]        ~ Normal(0, sigma_gamma)             # per-compound temp effect

  mu = alpha[t,c] + (beta[t,c] + delta[team]) * age + gamma[c] * (track_temp - 30)
  y  ~ StudentT(nu, mu, sigma_obs)                            # heavy-tailed

Partial pooling handles unseen tracks (they inherit compound-level priors).
5-fold CV mirrors v1's split-by-race protocol; posterior predictive draws give
empirical credible intervals at the 10/90 levels for an honest v2 vs v1 comparison.

Saves:
  data/oof_predictions_bayesian.parquet    — per-lap OOF posterior summaries
  data/bayesian_summary.parquet            — per-fold sampling diagnostics
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
from sklearn.model_selection import GroupKFold

DATA = Path("data/laps_corrected.parquet")
OUT_OOF = Path("data/oof_predictions_bayesian.parquet")
OUT_SUMMARY = Path("data/bayesian_summary.parquet")

N_FOLDS = 5
ALPHA = 0.20  # for 80% credible intervals
SEED = 12

# Reduce dataset to speed up Bayesian fits — subsample if needed
MAX_LAPS_PER_FOLD_FIT = 20_000  # subsample to this if a fold has more

DRAWS = 800
TUNE = 800
CHAINS = 2


def prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["fit_ok"]].copy()
    df = df.dropna(
        subset=["deg_s", "tyre_life", "track_temp", "compound", "event", "team"]
    )
    df["race_id"] = df["year"].astype(str) + "_" + df["round"].astype(str).str.zfill(2)
    return df.reset_index(drop=True)


def build_and_fit(df_fit: pd.DataFrame, seed: int):
    """Build PyMC model on df_fit and sample. Returns (model, idata, codes)."""
    # Codify categoricals
    tracks = sorted(df_fit["event"].unique())
    compounds = ["SOFT", "MEDIUM", "HARD"]
    teams = sorted(df_fit["team"].unique())
    track_idx = pd.Categorical(df_fit["event"], categories=tracks).codes
    compound_idx = pd.Categorical(df_fit["compound"], categories=compounds).codes
    team_idx = pd.Categorical(df_fit["team"], categories=teams).codes

    age = df_fit["tyre_life"].astype(float).values
    temp = df_fit["track_temp"].astype(float).values - 30.0  # centred
    y = df_fit["deg_s"].astype(float).values

    coords = {
        "track": tracks,
        "compound": compounds,
        "team": teams,
        "obs": np.arange(len(df_fit)),
    }
    with pm.Model(coords=coords) as model:
        # Compound-level priors
        mu_beta = pm.HalfNormal("mu_beta", sigma=0.10, dims="compound")
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=0.05, dims="compound")
        # Track x compound deg slope (non-centred parametrisation)
        beta_raw = pm.Normal("beta_raw", 0.0, 1.0, dims=("track", "compound"))
        beta = pm.Deterministic(
            "beta",
            pm.math.maximum(mu_beta + sigma_beta * beta_raw, 0.0),
            dims=("track", "compound"),
        )

        # Track x compound intercept (small — pace_loss is already stint-referenced)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=0.2)
        alpha = pm.Normal("alpha", 0.0, sigma_alpha, dims=("track", "compound"))

        # Team age modifier
        sigma_team = pm.HalfNormal("sigma_team", sigma=0.02)
        delta = pm.Normal("delta", 0.0, sigma_team, dims="team")

        # Per-compound temperature effect (s per degree-C deviation from 30C, scaled by age)
        gamma = pm.Normal("gamma", 0.0, 0.02, dims="compound")

        # Likelihood
        mu = (
            alpha[track_idx, compound_idx]
            + (beta[track_idx, compound_idx] + delta[team_idx]) * age
            + gamma[compound_idx] * temp * age / 30.0
        )
        nu = pm.Exponential("nu", 1 / 10.0)
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=1.0)
        pm.StudentT("y", nu=nu, mu=mu, sigma=sigma_obs, observed=y)

        idata = pm.sample(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            target_accept=0.9,
            random_seed=seed,
            progressbar=True,
        )

    return model, idata, dict(tracks=tracks, compounds=compounds, teams=teams)


def predict_posterior(idata, df_test, codes, n_samples=400):
    """Compute posterior mean and 10/90 quantiles for df_test using stored idata."""
    tracks = codes["tracks"]
    compounds = codes["compounds"]
    teams = codes["teams"]

    # Map test rows to indices — unseen levels get the prior (we substitute with mean)
    track_idx = pd.Categorical(df_test["event"], categories=tracks).codes
    compound_idx = pd.Categorical(df_test["compound"], categories=compounds).codes
    team_idx = pd.Categorical(df_test["team"], categories=teams).codes
    age = df_test["tyre_life"].astype(float).values
    temp = df_test["track_temp"].astype(float).values - 30.0

    # Stack posterior draws (chain x draw x ...)
    post = idata.posterior
    beta = post["beta"].stack(sample=("chain", "draw")).values  # (T, C, S)
    alpha = post["alpha"].stack(sample=("chain", "draw")).values
    delta = post["delta"].stack(sample=("chain", "draw")).values  # (Teams, S)
    gamma = post["gamma"].stack(sample=("chain", "draw")).values  # (C, S)
    sigma_obs = post["sigma_obs"].stack(sample=("chain", "draw")).values  # (S,)
    nu = post["nu"].stack(sample=("chain", "draw")).values

    n_total = beta.shape[-1]
    rng = np.random.default_rng(SEED + 11)
    idx = rng.choice(n_total, size=min(n_samples, n_total), replace=False)

    n_test = len(df_test)
    preds = np.zeros((len(idx), n_test))

    # Mean betas/alphas to substitute for unseen track levels
    mean_beta_compound = beta.mean(axis=0)  # average over tracks: shape (C, S)
    for s_i, s in enumerate(idx):
        # Pick per-track betas where seen, mean per compound otherwise
        if (track_idx == -1).any():
            beta_t = np.where(
                track_idx == -1,
                mean_beta_compound[compound_idx, s],
                beta[np.where(track_idx == -1, 0, track_idx), compound_idx, s],
            )
            alpha_t = np.where(
                track_idx == -1,
                0.0,
                alpha[np.where(track_idx == -1, 0, track_idx), compound_idx, s],
            )
        else:
            beta_t = beta[track_idx, compound_idx, s]
            alpha_t = alpha[track_idx, compound_idx, s]
        delta_t = np.where(
            team_idx == -1, 0.0, delta[np.where(team_idx == -1, 0, team_idx), s]
        )
        gamma_c = gamma[compound_idx, s]

        mu_s = alpha_t + (beta_t + delta_t) * age + gamma_c * temp * age / 30.0
        # Sample StudentT noise for predictive interval
        noise = rng.standard_t(nu[s], size=n_test) * sigma_obs[s]
        preds[s_i] = np.squeeze(mu_s) + np.squeeze(noise)

    pred_mean = preds.mean(axis=0)
    pred_lower = np.quantile(preds, ALPHA / 2, axis=0)
    pred_upper = np.quantile(preds, 1 - ALPHA / 2, axis=0)
    return pred_mean, pred_lower, pred_upper


def main():
    df = pd.read_parquet(DATA)
    df = prep(df)
    print(f"Training on {len(df):,} laps across {df['race_id'].nunique()} races")

    rng = np.random.default_rng(SEED)
    groups = df["race_id"].values

    oof_pred = np.full(len(df), np.nan)
    oof_lower = np.full(len(df), np.nan)
    oof_upper = np.full(len(df), np.nan)
    fold_assignment = np.full(len(df), -1, dtype=int)
    fold_summaries = []

    gkf = GroupKFold(n_splits=N_FOLDS)
    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(df, df["deg_s"], groups=groups)
    ):
        print(f"\n=== fold {fold} ===")
        df_train = df.iloc[train_idx]
        df_test = df.iloc[test_idx]
        # Subsample training data per-fold if huge — keeps fit times tractable
        if len(df_train) > MAX_LAPS_PER_FOLD_FIT:
            df_train = df_train.sample(MAX_LAPS_PER_FOLD_FIT, random_state=SEED + fold)
        print(f"  fit on {len(df_train):,} laps, test on {len(df_test):,} laps")

        t0 = time.time()
        _, idata, codes = build_and_fit(df_train, seed=SEED + fold)
        fit_time = time.time() - t0
        print(f"  sampling done in {fit_time:.0f}s")

        # Sampler diagnostics
        try:
            import arviz as az

            summary = az.summary(
                idata,
                var_names=[
                    "mu_beta",
                    "sigma_beta",
                    "sigma_alpha",
                    "sigma_team",
                    "sigma_obs",
                    "nu",
                ],
            )
            max_rhat = float(summary["r_hat"].max())
            min_ess = float(summary["ess_bulk"].min())
        except Exception:
            max_rhat = np.nan
            min_ess = np.nan

        pred_mean, pred_lower, pred_upper = predict_posterior(idata, df_test, codes)
        mae = float(np.mean(np.abs(df_test["deg_s"].values - pred_mean)))
        cov = float(
            np.mean(
                (df_test["deg_s"].values >= pred_lower)
                & (df_test["deg_s"].values <= pred_upper)
            )
        )
        width = float(np.mean(pred_upper - pred_lower))
        print(
            f"  test MAE={mae:.3f}s  cov={cov:.1%}  width={width:.3f}s  max_rhat={max_rhat:.3f}  min_ess={min_ess:.0f}"
        )

        oof_pred[test_idx] = pred_mean
        oof_lower[test_idx] = pred_lower
        oof_upper[test_idx] = pred_upper
        fold_assignment[test_idx] = fold

        fold_summaries.append(
            dict(
                fold=fold,
                n_train=len(df_train),
                n_test=len(df_test),
                fit_seconds=fit_time,
                test_mae=mae,
                test_coverage=cov,
                test_width=width,
                max_rhat=max_rhat,
                min_ess=min_ess,
            )
        )

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
            "air_temp",
            "track_temp",
            "deg_s",
        ]
    ].copy()
    out["pred"] = oof_pred
    out["pred_lower"] = oof_lower
    out["pred_upper"] = oof_upper
    out["fold"] = fold_assignment
    out["abs_err"] = np.abs(df["deg_s"].values - oof_pred)
    out["in_interval"] = (df["deg_s"].values >= oof_lower) & (
        df["deg_s"].values <= oof_upper
    )
    out.to_parquet(OUT_OOF, index=False)

    summary_df = pd.DataFrame(fold_summaries)
    summary_df.to_parquet(OUT_SUMMARY, index=False)

    print()
    print("=" * 60)
    print("v2 BAYESIAN — OVERALL")
    print("=" * 60)
    print(f"  Overall OOF MAE:        {float(np.mean(out['abs_err'])):.3f} s")
    print(f"  80% coverage:           {float(np.mean(out['in_interval'])):.1%}")
    print(
        f"  80% interval width:     {float(np.mean(out['pred_upper'] - out['pred_lower'])):.3f} s"
    )
    print(f"  Total fit time:         {summary_df['fit_seconds'].sum():.0f}s")
    print()
    print(f"Saved oof predictions to {OUT_OOF}")
    print(f"Saved fold summary to {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
