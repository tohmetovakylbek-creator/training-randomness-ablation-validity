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

## Data policy

Large generated artifacts are excluded from Git history. Before public release, archive them with checksums and record a permanent DOI or release URL here.
