"""
Diagnostic for the balanced-error vs cross-entropy seed-share divergence
seen in cc18_per_dataset_summary.csv (median 3.0% vs 50.4%). Tests the
"discretization" hypothesis: balanced-error effects are exactly zero for
any test example whose argmax prediction is unchanged between the full
and ablated configs, and jump discretely when it does change -- so its
per-example variance is dominated by WHICH borderline examples happen to
be in this particular finite test set (an evaluation-variance /
V_win-like source), not by training-seed randomness, whenever flips are
rare. Cross-entropy has no such floor -- it moves for every example,
seed-driven or not.

No training, no GPU. Reads the same artifacts/*.npz files 03_analyze.py
already reads.

Run on: laptop or desktop, CPU is enough.
"""
import glob
import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
MANIFEST_PATH = os.path.join(REPO_ROOT, "data", "manifests", "cc18_manifest.csv")
ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts")
SUMMARY_PATH = os.path.join(REPO_ROOT, "results", "cc18", "cc18_per_dataset_summary.csv")
OUT_PATH = os.path.join(REPO_ROOT, "results", "cc18", "cc18_flip_diagnostic.csv")


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    rows = []

    for path in glob.glob(os.path.join(ARTIFACT_DIR, "task_*_fold_0.npz")):
        z = np.load(path, allow_pickle=True)
        task_id = int(z["task_id"])
        pred_full, pred_abl = z["pred_full"], z["pred_ablated"]   # (S, N)
        n_test = pred_full.shape[1]

        # fraction of (seed, example) pairs where the argmax prediction differs
        # between configs -- the discretization hypothesis predicts this is small
        # exactly where balanced-error seed share came out small.
        flip_rate = float((pred_full != pred_abl).mean())
        # per-seed flip counts, to see if it's consistently low or just averages out
        per_seed_flips = (pred_full != pred_abl).mean(axis=1)

        rows.append(dict(
            task_id=task_id,
            n_test=n_test,
            flip_rate_pct=round(100 * flip_rate, 2),
            flip_rate_min_seed_pct=round(100 * per_seed_flips.min(), 2),
            flip_rate_max_seed_pct=round(100 * per_seed_flips.max(), 2),
            n_examples_ever_flipped=int((pred_full != pred_abl).any(axis=0).sum()),
        ))

    out = pd.DataFrame(rows).merge(manifest[["task_id", "name"]], on="task_id")

    summary_path = SUMMARY_PATH
    if os.path.exists(summary_path):
        summary = pd.read_csv(summary_path)[["task_id", "balanced_error_seed_share",
                                               "cross_entropy_seed_share"]]
        out = out.merge(summary, on="task_id")
        out = out.sort_values("balanced_error_seed_share")
        cols = ["name", "n_test", "flip_rate_pct", "n_examples_ever_flipped",
                "balanced_error_seed_share", "cross_entropy_seed_share"]
    else:
        out = out.sort_values("flip_rate_pct")
        cols = ["name", "n_test", "flip_rate_pct", "n_examples_ever_flipped",
                "flip_rate_min_seed_pct", "flip_rate_max_seed_pct"]
        print(f"({SUMMARY_PATH} not found -- showing flip rates alone; run "
              f"03_analyze.py first to see the correlation with seed share directly)")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(out[cols].to_string(index=False))

    if "balanced_error_seed_share" in out.columns:
        corr = out["flip_rate_pct"].corr(out["balanced_error_seed_share"])
        print(f"\nCorrelation between flip rate and balanced-error seed share: {corr:.2f}")
        print("Flip rate showed a moderate positive association with the balanced-error "
              "seed share. Given the small number of datasets, this diagnostic is interpreted "
              "as exploratory evidence that metric discretization contributes to the difference "
              "between balanced-error and cross-entropy variance shares.")


if __name__ == "__main__":
    main()
