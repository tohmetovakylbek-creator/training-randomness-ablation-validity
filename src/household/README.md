# Household / ECL / generality-check code

Source code for the energy-domain analyses in the manuscript: the primary
household study (UK-DALE, REFIT, SHEERM), the preregistered ECL replication,
and the PatchTST/BiLSTM external-architecture generality checks. This
complements `src/cc18/`, which covers the non-energy OpenML-CC18 scope probe.

**Status.** This code was recovered from the author's original working
project (not previously under version control) and added to the repository
after the fact. Unlike `src/cc18/`, it has not yet been re-run end-to-end in
this environment against the original raw datasets (UK-DALE/REFIT/SHEERM/ECL
are large, license-gated downloads and training is GPU-bound), so treat it as
recovered source rather than independently re-verified. It does import
cleanly and the core model (`models.vmd_patchtst_aswa.VMDPatchTSTASWA`) and
the VMD decomposition (`models.vmd.vmd_batch`) have been smoke-tested: the
model instantiates, runs a forward/loss pass for both `aux_mode` settings,
and VMD reconstruction (`sum_k u_k`) matches the input to ~1e-7.

Scripts take input/output paths as CLI arguments rather than hardcoding them
(unlike the CC18 scripts), so no path-rewriting was needed to make this
importable from its new location — run everything from this directory
(`src/household/`), since imports are flat (e.g. `ecl_loader.py` imports
`from ukdale_loader import ...`) by the original project's own design.

## Layout

- `models/` — the studied architecture.
  - `vmd.py` — causal VMD decomposition per input window (K=6, alpha=2000);
    wraps the third-party `vmdpy` package (MIT, Carvalho, implementing
    Dragomiretskiy & Zosso 2014), listed in `requirements.txt`.
  - `patchtst.py` — channel-independent, weight-shared PatchTST encoder with
    learnable mode embeddings (§3, the identity mechanism).
  - `aswa.py` / `vmd_patchtst_aswa.py` — the adaptive per-horizon aggregation
    (ASWA), gated skip fusion (`GatedSkip`, Eq. 1), and the full hybrid model
    with the `aux_mode`/`aggregation` factorial-design flags (§4.3.1, Table 3).
  - `aswa_original.py` / `vmd_patchtst_aswa_original.py` — the pre-factorial
    versions, kept as `patch_model.py` (below) documents and produces them.
  - `patch_model.py` — **not a model** — a one-time source patcher that adds
    the `aux_mode`/`aggregation` parameters to `aswa.py` and
    `vmd_patchtst_aswa.py` in place, backing up the originals under
    `*_original.py`. Already applied; kept for provenance. Re-running it on
    the current (already-patched) files will refuse (patterns won't match).

- **Data loading**: `ukdale_loader.py`, `refit_loader.py`, `sheerm_loader.py`
  — all three share preprocessing (`to_hourly`, `Scaler`,
  `chronological_split`, `sigma_clip_fit`, `make_windows`) from
  `ukdale_loader.py`, so any UK-DALE/REFIT/SHEERM difference in results is a
  dataset effect, not a codebase effect (§4.1). `audit_protocol.py` verifies
  this identity from already-processed `.npz` files without retraining.

- **Training / evaluation**: `train.py`, `evaluate.py`,
  `train_vmd_patchtst_ukdale.py` (historical implementation that produced the
  original unpublished-thesis result, §5.4.4/§6.2.4 — kept for provenance,
  not the current pipeline).

- **Factorial design & variance decomposition (§4.3.1, §5.1, §5.2, §5.4)**:
  `factorial_aux_agg.py`, `variance_model.py`, `meta_analysis.py`,
  `meta_estimators.py`, `normalized_metrics.py`, `renormalize.py`,
  `coverage_simulation.py`, `assumption_diagnostics.py`,
  `block_sensitivity.py`, `compare_gate.py`, `compare_reweighting.py`,
  `path_diagnostics.py`, `plot_meta.py`, `loo_summary.py`, `stats.py`.

- **Data-quality audit (§4.2.1/S1.1)**: `plateau_audit.py`,
  `flat_vs_interp.py`, `gap_check.py`, `check_shapes.py`.

- **Mechanism / intervention (§5.3)**: `intervention.py`, `mean_embedding.py`,
  `probe_control.py`, `probe_merge.py`, `interv_summary.py`, `mechanistic.py`,
  `aggregation_check.py` (explains the §5.1.2 sum-vs-convex masking bug),
  `skip_attribution.py`. `intervention_legacy.py` repeats the intervention on
  the historical implementation, for comparison.

- **External-architecture generality checks (§5.2.6/S3.1–S3.2)**:
  `generality_models.py`, `generality_variance.py`, `run_generality.py`,
  `crosscheck_variance.py` / `crosscheck_variance2.py` (verify
  `generality_variance.py` agrees with `variance_model.py` on the same
  household), `find_per_seed_npz.py`.

- **ECL preregistered replication (§5.5)**: `ecl_loader.py`,
  `ecl_select_clients.py`, `ts_to_ecl_csv.py`, `factorial_ecl.py`,
  `sanity_etth1.py` (ETTh1 pipeline-transfer sanity check; not part of the
  variance decomposition or meta-analysis).

## Not yet included

- The `results/` tree (per-window/per-seed arrays and JSON summaries this
  code produces) has not yet been curated into the repository — it needs the
  same compact-vs-bulk triage `artifacts/cc18_per_example_artifacts.zip`
  already got for CC18.
- No file named `results_manifest.csv` — the machine-readable per-experiment
  index the manuscript's Reproducibility section describes — exists in the
  author's original project. It does not currently exist anywhere and needs
  to be either located or generated before the manuscript's claim about it is
  accurate.
- A handful of scripts present in the original project were intentionally
  left out as outside this paper's reported scope: `ablation.py`,
  `tune_identity_lr.py`, `posthoc_full_vs_lean.py`, `run_all.py`.
