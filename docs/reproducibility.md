# Reproducibility notes

## OpenML-CC18

- Suite: OpenML-CC18, suite ID 99.
- Split: repeat 0, fold 0, sample 0.
- Seeds: 42, 7, 13, 99, 2025.
- Configurations: feature-identity bias on and off.
- Primary metric: balanced error in percentage points.
- Secondary metric: per-example cross-entropy.
- Per-dataset intervals: Student t with 4 degrees of freedom.
- Pooling: random-effects REML with modified Hartung-Knapp inference.

The 17 NPZ artifacts were independently checked for dimensions, finite losses, class ranges, unique test indices, seed identity, and task-manifest agreement. Re-running the final analysis reproduced the committed per-dataset summary without numerical differences above 1e-12.

## results/results_manifest.csv

The manuscript's Reproducibility section and Supplementary Table S1 ("Experiment registry") describe a `results/results_manifest.csv` recording, for every experiment, its domain, units, seed count, evaluation count, component, outcome, and inferential status. No such file existed anywhere in the author's original project (verified against the full recovered `uk_dale_project` source tree).

The current `results/results_manifest.csv` was **reconstructed** to match Table S1's five rows (Household primary, PatchTST, BiLSTM, ECL, OpenML-CC18) exactly, with every numeric field (seed/window counts, unit counts) recomputed directly from the result files now in this repository rather than copied from the manuscript text:

- Household primary: 17 units, W = 78-218, median seed share 82.8% — from `results/household/results_master.json`.
- PatchTST / BiLSTM: 17 units each, W = 78-218, median seed share 34.1% / 69.1%, median seed-to-evaluation ratio 2.6 / 11.2 — from `results/household/generality/summary.json`.
- ECL: 17 clients, W = 120 (primary) and 157 (sensitivity) — from `results/household/ecl/factorial_ecl.json` and `results/household/ecl_fullW/factorial_ecl.json`.
- OpenML-CC18: 17 tasks, n_test 146-6756 — from `results/cc18/cc18_per_dataset_summary.csv`.

All of the above independently matched the manuscript's reported figures (median 83% household seed share, 34%/69% for PatchTST/BiLSTM, the Table 2 household composition, and the exact CC18 n_test range), which is the basis for trusting the underlying result files, not just this index. Because this file was built rather than recovered, it should be reviewed against the source data before the manuscript's claim about it is treated as satisfied — in particular, decide whether the granularity (5 rows, matching Table S1) is what "for every experiment" in the main text is meant to cover, or whether a more granular per-unit or per-configuration manifest is expected instead.

## Data policy

Large generated artifacts are excluded from Git history. Before public release, archive them with checksums and record a permanent DOI or release URL here.
