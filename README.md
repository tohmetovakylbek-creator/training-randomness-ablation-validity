# Training Randomness and Ablation Validity

Private reproducibility repository for:

> **Training Randomness Can Dominate Uncertainty in Architecture Ablations: A Preregistered Two-Domain Study of Gated Load Forecasting**

The study quantifies how stochastic training and finite evaluation samples affect architectural ablations in decomposition-based load forecasting. It includes a preregistered second-domain replication and a non-energy feature-identity ablation on 17 OpenML-CC18 classification tasks.

## Repository status

This repository is an author working package. The manuscript, Supplement, energy-domain scripts, provenance records, and archival data references will be added before public release.

## Current layout

- `configs/` - frozen analysis and run configurations.
- `data/manifests/` - dataset-selection manifests; no restricted raw data.
- `data/processed/` - compact derived inputs suitable for version control.
- `src/cc18/` - OpenML-CC18 selection, training, analysis, and diagnostic code.
- `src/meta_analysis/` - REML and modified Hartung-Knapp pooling.
- `src/simulation/` - simulation-based power and coverage analysis.
- `results/cc18/` - per-dataset CC18 summaries and flip diagnostics.
- `manuscript/` - manuscript and Supplement drafts.
- `docs/` - protocol classification, deviation log, and reproducibility notes.

## OpenML-CC18 analysis

The CC18 experiment uses a compact FT-Transformer and removes the per-feature identity bias from the tokenizer:

`T_j(x_j) = b_j + f_j(x_j)  ->  T_j(x_j) = f_j(x_j)`.

It is a **feature-identity ablation**, not a BatchNorm ablation. Balanced error is the primary metric and cross-entropy is a secondary sensitivity metric. Dataset effects are pooled with REML and modified Hartung-Knapp inference.

Large per-example NPZ artifacts are intentionally excluded from Git history. They will be archived separately or attached to a versioned release.

## Reproducibility

The CC18 pipeline is ordered as follows:

1. `src/cc18/01_select_datasets.py`
2. `src/cc18/02_run_experiment.py`
3. `src/cc18/03_analyze.py`
4. `src/cc18/04_diagnose_flip_sparsity.py`
5. `src/meta_analysis/05_pool_reml_hk.py`

Run configuration and dataset-selection provenance are recorded under `configs/` and `data/manifests/`.
