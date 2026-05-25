# F1 Tyre Degradation Model

End-to-end pipeline for modelling F1 tyre pace loss from raw timing data.
Covers data ingestion, confounder correction, two ML models, and a race strategy simulator.

---

## Pipeline overview

```
FastF1 API
    └─ scripts/build_dataset.py       raw lap filtering → data/laps_clean.parquet
        └─ scripts/correct_all_races.py   fuel + track-evo correction → data/laps_corrected.parquet
            ├─ model/train_lgbm.py         v1 LightGBM + conformal intervals
            ├─ model/train_lgbm_quantile.py  v1b CQR quantile model
            ├─ model/train_bayesian.py     v2 hierarchical Bayesian (PyMC)
            ├─ model/eval_temporal.py      temporal holdout: 2023-24 → 2025
            └─ model/strategy_sim.py       race strategy simulator
```

---

## Dataset

|                            |                    |
| -------------------------- | ------------------ |
| Seasons                    | 2023, 2024, 2025   |
| Raw laps                   | 58,970             |
| Races                      | 62                 |
| Races passing quality gate | 53 / 62            |
| Fit-ok laps                | 51,437             |
| Tracks                     | 23                 |
| Drivers                    | 28                 |
| Compounds                  | SOFT, MEDIUM, HARD |

### Quality gate (`fit_ok`)

Each race is fit with an OLS model `laptime ~ driver + lap_n + lap_n² + age:compound`. A race passes if:

- R² ≥ 0.50
- Residual RMSE ≤ 1.0 s
- `beta_lap_n > −0.10` (rules out races where the correction absorbed tyre deg into a physically implausible negative track-trend)

9 races fail, mostly Monaco, Singapore 2023, and Azerbaijan 2025 (SC-heavy with corrupted lap time distributions).

---

## Target: `deg_s`

Raw lap time → remove fuel load (0.056 s/lap) → fit quadratic track-evolution curve → subtract per-stint fresh reference (mean of laps 2–4):

```
deg_s = pace_loss_s − stint_ref_pace
```

`deg_s` is zero at fresh tyres and increases as the tyre ages. Mean ≈ 0.40 s, std ≈ 1.1 s.

![Pace loss by compound](figures/pace_loss_by_compound.png)

---

## Models

### v1 — LightGBM + split-conformal intervals

5-fold race-grouped CV (no lap leakage). 20% of training races held out as conformal calibration set. Symmetric intervals: `[ŷ − q_α, ŷ + q_α]`.

**Features**

| Feature            | Importance |
| ------------------ | ---------- |
| tyre_life          | 31.4%      |
| event (track)      | 23.5%      |
| team               | 8.9%       |
| stint number       | 7.2%       |
| air_temp           | 5.0%       |
| starting_tyre_life | 4.5%       |
| track_temp         | 3.9%       |
| evo_swing          | 3.7%       |
| humidity           | 3.2%       |
| tyre_life²         | 2.8%       |
| is_street          | 0.9%       |
| compound           | 0.2%       |

`compound` ranks low because `deg_s` is stint-referenced — the compound effect is in the _shape_ of the curve, captured by `age_soft/medium/hard` interaction features and the track embedding.

![Feature importance](figures/v1_feature_importance.png)

**Results (5-fold CV)**

| Metric               | Value   |
| -------------------- | ------- |
| OOF MAE              | 0.605 s |
| Baseline (predict 0) | 0.785 s |
| Improvement          | 22.9%   |
| 80% coverage         | 75.7%   |
| Interval width       | 1.713 s |

By compound:

| Compound | n      | MAE     | 80% Coverage |
| -------- | ------ | ------- | ------------ |
| SOFT     | 3,009  | 0.617 s | 76.2%        |
| MEDIUM   | 16,405 | 0.544 s | 79.3%        |
| HARD     | 23,834 | 0.645 s | 73.3%        |

![Predicted vs actual](figures/v1_pred_vs_actual.png)
![Calibration](figures/v1_calibration.png)
![MAE breakdown](figures/v1_mae_breakdown.png)

---

### v1b — Conformalized Quantile Regression (CQR)

Same features. Three LightGBM models per fold: MAE point estimate, lower quantile (α/2), upper quantile (1−α/2). CQR nonconformity score `max(q_lo − y, y − q_hi)` calibrates the interval width, giving adaptive (heteroskedastic) intervals with a coverage guarantee.

| Metric         | v1 conformal | v1b CQR     |
| -------------- | ------------ | ----------- |
| MAE            | 0.605 s      | **0.602 s** |
| 80% coverage   | 75.7%        | **79.3%**   |
| Interval width | 1.713 s      | 1.860 s     |

CQR matches MAE and recovers the 80% coverage target. Intervals are narrower early in a stint and wider at high tyre age.

---

### v2 — Hierarchical Bayesian (PyMC)

Non-centred parametrisation with partial pooling: per-(track, compound) deg slopes share compound-level priors. Handles unseen tracks via prior fallback. Student-T likelihood for robustness to outliers.

```
beta[track, compound] ~ HalfNormal(mu_beta[compound], sigma_beta[compound])
alpha[track, compound] ~ Normal(0, sigma_alpha)
delta[team]            ~ Normal(0, sigma_team)
gamma[compound]        ~ Normal(0, 0.02)

mu = alpha[t,c] + (beta[t,c] + delta[team]) * age + gamma[c] * (temp−30) * age/30
y  ~ StudentT(nu, mu, sigma_obs)
```

20k-lap subsampling per fold to keep fit times tractable (~4–6 min/fold, 2 chains).

| Metric         | v1 LightGBM | v2 Bayesian |
| -------------- | ----------- | ----------- |
| OOF MAE        | 0.605 s     | 0.677 s     |
| 80% coverage   | 75.7%       | 72.2%       |
| Interval width | 1.713 s     | 1.727 s     |
| Total fit time | —           | 1,389 s     |

LightGBM wins on MAE. Bayesian undercoverage is partly due to subsampling — the posterior underestimates uncertainty with fewer training laps.

---

## Temporal holdout: 2023–2024 → 2025

Train on 34 races (2023–24), test on 19 races (2025). This is the honest out-of-sample evaluation — the model has never seen the 2025 season.

| Metric       | CV (5-fold) | Temporal 2025 |
| ------------ | ----------- | ------------- |
| MAE          | 0.605 s     | 0.633 s       |
| Baseline     | 0.785 s     | 0.757 s       |
| Improvement  | 22.9%       | 16.4%         |
| 80% coverage | 75.7%       | 75.8%         |

8% MAE degradation season-to-season. Coverage is stable, suggesting the conformal calibration generalises.

**Per-track MAE (2025 test set)**

| Track             | n         | MAE         | Coverage  |
| ----------------- | --------- | ----------- | --------- |
| Emilia Romagna GP | 821       | 0.457 s     | 88.9%     |
| Abu Dhabi GP      | 1,048     | 0.466 s     | 85.9%     |
| Canadian GP       | 1,084     | 0.472 s     | 86.4%     |
| Qatar GP          | 922       | 0.487 s     | 82.8%     |
| USGP              | 707       | 0.533 s     | 84.0%     |
| …                 |           |             |           |
| Hungarian GP      | 1,289     | 0.708 s     | 73.9%     |
| Japanese GP       | 997       | 0.872 s     | 57.7%     |
| **Singapore GP**  | **1,069** | **1.093 s** | **46.9%** |

Singapore and Japan are persistent outliers. Root cause: at Singapore, `deg_s` has only r=0.49 correlation with tyre age (vs r=0.59 at Austria) — the correction model absorbs real degradation signal into track-evolution noise at street circuits with heavy safety car activity.

---

## Per-compound ablation

Three separate per-compound models vs one joint model:

| Model                           | OOF MAE     |
| ------------------------------- | ----------- |
| Joint LightGBM                  | **0.605 s** |
| Per-compound (SOFT/MEDIUM/HARD) | 0.633 s     |

The joint model wins by borrowing cross-compound track information. A track that degrades tyres hard in general affects all compounds — the joint model learns this; per-compound models can't.

---

## Strategy simulator

`model/strategy_sim.py` compares tyre strategies by predicting cumulative pace loss per stint.

**Example — British Grand Prix, Mercedes**

| Strategy                  | Tyre deg | Pit loss | Total       |
| ------------------------- | -------- | -------- | ----------- |
| 1-stop: HARD → MEDIUM     | +17.6 s  | +22 s    | **+39.6 s** |
| 1-stop: MEDIUM → HARD     | +20.5 s  | +22 s    | +42.5 s     |
| 2-stop: MED → HARD → SOFT | +16.5 s  | +44 s    | +60.5 s     |
| 2-stop: SOFT → MED → HARD | +16.7 s  | +44 s    | +60.7 s     |

1-stop strategies dominate once the 44 s double-pit penalty is included. HARD→MEDIUM beats MEDIUM→HARD because running the harder tyre fresh (lower initial deg rate) and switching to MEDIUM for the final stint is more efficient than the reverse.

```python
from model.strategy_sim import load_model, simulate

model, meta = load_model()
laps = simulate(
    model, meta,
    track="British Grand Prix",
    team="Mercedes",
    stints=[("HARD", 28), ("MEDIUM", 30)],
)
print(laps.groupby("stint")["pred_deg_s"].agg(["mean", "sum"]))
```

---

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install fastf1 lightgbm pymc arviz statsmodels scikit-learn pandas numpy matplotlib

# 1. Fetch data (requires ~500 FastF1 API calls, uses local cache after first run)
python scripts/build_dataset.py

# 2. Correct for fuel load and track evolution
python scripts/correct_all_races.py

# 3. Train models
python model/train_lgbm.py
python model/train_lgbm_quantile.py
python model/train_bayesian.py      # ~25 min

# 4. Evaluate
python model/eval_temporal.py

# 5. Strategy simulator
python model/strategy_sim.py
```

Data and model artefacts are gitignored (`data/`, `cache/`). Re-running `build_dataset.py` with a warm cache takes ~2 min.
