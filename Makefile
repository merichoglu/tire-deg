.PHONY: all data train eval plots clean

# Full pipeline in one shot
all: data train eval plots

# --- Data ---
data:
	python scripts/build_dataset.py
	python scripts/correct_all_races.py

# --- Models ---
train:
	python model/train_lgbm.py
	python model/train_lgbm_quantile.py
	python model/train_lgbm_per_compound.py

# Bayesian model is expensive (~25 min); run separately
train-bayesian:
	python model/train_bayesian.py

# --- Evaluation ---
eval:
	python model/evaluate_model.py
	python model/eval_temporal.py
	python model/eval_loto.py

# --- Plots ---
plots:
	python model/plot_importance.py
	python scripts/inspect_fits.py
	python scripts/visualize_stints.py
	python scripts/visualize_corrected.py
	python scripts/worked_example.py

# --- Strategy simulator ---
strategy:
	python model/strategy_sim.py

clean:
	rm -rf data/ cache/ __pycache__ model/__pycache__ scripts/__pycache__
