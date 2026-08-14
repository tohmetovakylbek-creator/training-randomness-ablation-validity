# Training Randomness and Ablation Validity

Reproducibility repository for:

> **Training Randomness in Architecture Ablations: A Crossed Seed–Window Analysis with Preregistered Replication**
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21935720.svg)](https://doi.org/10.5281/zenodo.21935720)
The study quantifies how stochastic training and finite evaluation samples affect architectural ablations in decomposition-based load forecasting. It includes a preregistered second-domain replication and a non-energy feature-identity ablation on 17 OpenML-CC18 classification tasks.

## Repository status

This repository is an author working package. It currently contains the manuscript, the complete OpenML-CC18 analysis package, compact result tables, and selected energy-domain analyses. Additional energy-domain provenance records and archival references will be added before submission.

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

The 17 per-task NPZ files used to reconstruct the CC18 summaries are bundled in `artifacts/cc18_per_example_artifacts.zip`. Its SHA-256 digest is recorded in `CHECKSUMS.sha256`. A versioned archival release with a persistent identifier will be created for the submitted article.

## Reproducibility

The CC18 pipeline is ordered as follows:

1. `src/cc18/00_preflight_check.py`
2. `src/cc18/01_select_datasets.py`
3. `src/cc18/02_run_experiment.py`
4. `src/cc18/03_analyze.py`
5. `src/cc18/04_diagnose_flip_sparsity.py`
6. `src/meta_analysis/05_pool_reml_hk.py`

Run configuration and dataset-selection provenance are recorded under `configs/` and `data/manifests/`.

## Licensing

- Source code is licensed under the MIT License; see `LICENSE`.
- Author-created configurations, manifests, documentation, compact derived results, and generated CC18 artifacts are licensed under CC BY 4.0; see `LICENSE-DATA`.
- The manuscript is not covered by those repository licenses pending the journal's publishing agreement.
- Source datasets remain subject to their original licenses and terms.

See `LICENSE-SCOPE.md` for the directory-level scope and attribution guidance.
