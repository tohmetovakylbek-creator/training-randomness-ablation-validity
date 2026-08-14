# Deviation log

## OpenML-CC18

- During Phase 1, test sizes were estimated using an 80/20 split. The experiment subsequently used the official OpenML repeat-0, fold-0, sample-0 split, which is approximately a 10% fold for most tasks.
- The frozen set of 17 tasks was retained. Sensitivity analyses restricted pooling to tasks with at least 150 and at least 250 test observations.
- An initial run on `sick` collapsed to majority-class predictions because an all-missing numeric feature propagated a non-finite imputation value and unweighted cross-entropy favored the majority class. The preprocessing fallback and class-weighted training objective were corrected, after which all 170 fits were rerun.
- The implemented comparison is a feature-identity tokenizer-bias ablation. It must not be described as BatchNorm on/off.
