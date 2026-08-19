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

## Results data

Compact per-unit JSON/CSV summaries live in `results/household/`, mirroring
`results/cc18/`'s layout: one subfolder per experiment
(`factorial/`, `factorial_2comp/`, `factorial_5seed/`, `variance/`,
`variance_17/`, `meta/`, `meta_14/`, `meta_16/`, `meta_17/`, `meta_2comp/`,
`intervention/`, `intervention_5seed/`, `intervention_legacy/`, `paths/`,
`agg/`, `diagnostics/`, `figures/`, `ecl/`, `ecl_fullW/`, `generality/`,
`norm/`, `skip_attribution/`), plus loose root-level files
(`results_master.json` — the single source of truth the manuscript's Section
5.4.2 numbers were checked against; `factorial_norm.json`;
`ecl_eligibility.csv`/`ecl_selection.json`; `meanemb_*.json`,
`probe_*.json`; `mae_const_households.json`).

Verified against the manuscript: `results_master.json`'s 17 units are
exactly the REFIT (2,4,6,9,10) + SHEERM (1,2,3,4,5,8,9,10,11,12,13) +
UK-DALE (1) households from Table 2, with median `share_seed` = 82.8% and
median per-unit effect -0.24% of household error — matching the manuscript's
"median 83%" seed share and the sign/magnitude of the -0.26% pooled effect.

Per-window/per-seed `.npz`/`.npy` arrays (gitignored, like
`artifacts/cc18_per_example_artifacts.zip`) are bundled in
`artifacts/household_per_window_artifacts.zip`, checksummed in
`CHECKSUMS.sha256`. Its internal layout mirrors `results/household/`'s
subfolder names (minus the `results_`/`results/` prefix), so a file dropped
back into the matching `results/household/<name>/` subfolder sits next to
its JSON summary.

**Deliberately excluded** from both the code and results additions, as
outside this paper's reported scope (matches the excluded scripts, above):
`results/ukdale_vmd_patchtst_aswa/` (43MB of model checkpoints — not needed
to verify any reported number, only to rerun `intervention.py` without
retraining), `results_full/` and `results_sheerm/results.json` /
`results_refit_unified/results.json` (products of the excluded
`posthoc_full_vs_lean.py`/`run_all.py`, a different "full vs lean" framing
than the current factorial design — note this means the `results_sheerm`
name is misleading: it is not the SHEERM household results used in the
paper, those are in `results/household/{factorial,variance,meta,...}`),
and `results_smoke/`, `results_timing*/`, `results_tuning/` (scratch runs).

## Not yet included

- No file named `results_manifest.csv` — the machine-readable per-experiment
  index (domain, units, seed count, evaluation count, component, outcome,
  inferential status, source artifact) that the manuscript's Reproducibility
  section and Supplementary Table S1 describe — exists anywhere in the
  recovered project. It needs to be either located or generated before the
  manuscript's claim about it is accurate; `results_master.json` is close for
  the household study alone but doesn't cover ECL, generality checks, or
  CC18, and isn't in the S1 registry's column format.
- A handful of scripts present in the original project were intentionally
  left out as outside this paper's reported scope: `ablation.py`,
  `tune_identity_lr.py`, `posthoc_full_vs_lean.py`, `run_all.py`.
