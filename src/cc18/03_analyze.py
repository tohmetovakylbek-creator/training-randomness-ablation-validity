"""
Phase 3 (v2) -- department head's corrections applied:

  * Completeness gate (point 9): refuses to compute anything unless
    every task in manifest.csv has a corresponding artifacts/task_<id>_
    fold_0.npz. Prints missing tasks and stops rather than silently
    averaging over whatever happens to be present.

  * No artificial blocks (point 3): the crossed seed x window estimators
    in seed_window_simgrid.py (crossed_anova, block_bootstrap_vwin,
    f_interval_share) are reused UNCHANGED -- what changes is what gets
    fed into them. Each individual test example is its own "window"
    (W = n_test, not 10-40 synthetic blocks), and the bootstrap block
    length is 1 (ordinary iid resampling), because CC-18 test examples
    are exchangeable and have no real neighbourhood structure to
    preserve. This is mathematically block_bootstrap_vwin(..., block=1).

  * Two metrics, both derived here from the same per-example artifacts
    without any retraining (point 5):
      - PRIMARY: balanced-error contribution,
          L_i = 100 * (N_test / (n_classes * N_test_class[y_i])) * 1(pred_i != y_i)
        chosen so mean(L_i) over the test fold equals 100 x balanced
        error exactly (a finite-sample identity, not just in
        expectation) -- verified algebraically and reproduced in
        _check_balanced_error_identity() below.
      - SECONDARY (sensitivity): per-example cross-entropy, effect_i =
        loss_ablated_i - loss_full_i, already stored directly.

  * Output: cc18_per_dataset_summary.csv with both metrics' seed share,
    F-based interval, and t_{S-1} effect CI per dataset -- feed this
    into your existing REML + modified Hartung-Knapp pooling code
    (the same one used for households/ECL) rather than reimplementing
    pooling here.

Run on: laptop or desktop, CPU is enough.
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

from seed_window_simgrid import crossed_anova, block_bootstrap_vwin, f_interval_share

ALPHA = 0.05
BLOCK_LEN = 1           # ordinary bootstrap -- see module docstring, point 3
NBOOT = 1000
ARTIFACT_DIR = "artifacts"


def _check_balanced_error_identity():
    """Self-test: mean(L_i) must equal 100 * (1 - balanced_accuracy) exactly."""
    rng = np.random.default_rng(0)
    n_classes = 4
    y = rng.integers(0, n_classes, size=537)
    pred = y.copy()
    flip = rng.random(len(y)) < 0.3
    pred[flip] = rng.integers(0, n_classes, size=flip.sum())

    counts = np.bincount(y, minlength=n_classes)
    N = len(y)
    weights = N / (n_classes * counts[y])
    L = 100 * weights * (pred != y)

    accs = [(pred[y == k] == k).mean() for k in range(n_classes) if (y == k).any()]
    balanced_err_pct = 100 * (1 - np.mean(accs))
    assert abs(L.mean() - balanced_err_pct) < 1e-9, (L.mean(), balanced_err_pct)


def balanced_error_contributions(pred, y_test, n_classes):
    """L_i per example, per seed. pred: (S, N). y_test: (N,). Returns (S, N)."""
    counts = np.bincount(y_test, minlength=n_classes)
    N = len(y_test)
    weights = N / (n_classes * counts[y_test])          # (N,)
    return 100.0 * weights[None, :] * (pred != y_test[None, :])


def analyze_one_metric(d, rng):
    """d: (1, S, W) paired per-example effects for one dataset. Returns summary dict."""
    S, W = d.shape[1], d.shape[2]

    ms_seed, ms_int, sig2_seed, sig2_int, _ = crossed_anova(d)
    dbar = d.mean(axis=1)
    vwin = block_bootstrap_vwin(dbar, BLOCK_LEN, NBOOT, rng)
    ghat = d.mean(axis=(1, 2))

    ms_seed, ms_int = float(ms_seed[0]), float(ms_int[0])
    sig2_seed, sig2_int, vwin, ghat = (float(sig2_seed[0]), float(sig2_int[0]),
                                        float(vwin[0]), float(ghat[0]))

    denom = sig2_seed / S + vwin
    if denom <= 0:
        # Both components are exactly zero -- either every example got the identical
        # prediction under both configs and all seeds (a legitimate, if extreme, null
        # result), or upstream data was degenerate (e.g. an all-missing feature column
        # that silently NaN'd training into a constant majority-class predictor -- see
        # the "sick" incident in the deviation log). Flag it; do not crash and do not
        # guess which case it is.
        return dict(
            S=S, W=W, effect=ghat, effect_ci_lo=np.nan, effect_ci_hi=np.nan,
            significant_full_t=False,
            seed_share=np.nan, seed_share_ci_lo=np.nan, seed_share_ci_hi=np.nan,
            seed_share_gt_half=False, truncated_at_zero=True,
            degenerate_zero_variance=True,
        )

    share = (sig2_seed / S) / denom
    lo, hi = f_interval_share(ms_seed, ms_int, S, W, vwin, alpha=ALPHA)

    se_full = np.sqrt(denom)
    t_crit = stats.t.ppf(1 - ALPHA / 2, S - 1)
    ci_lo, ci_hi = ghat - t_crit * se_full, ghat + t_crit * se_full

    return dict(
        S=S, W=W, effect=ghat, effect_ci_lo=ci_lo, effect_ci_hi=ci_hi,
        significant_full_t=bool((ci_lo > 0) or (ci_hi < 0)),
        seed_share=round(100 * share, 1),
        seed_share_ci_lo=round(100 * float(lo), 1),
        seed_share_ci_hi=round(100 * float(hi), 1),
        seed_share_gt_half=bool(share > 0.5),
        truncated_at_zero=bool(sig2_seed <= 0),
        degenerate_zero_variance=False,
    )


def analyze_dataset(npz_path, rng):
    z = np.load(npz_path, allow_pickle=True)
    y_test = z["y_test"]
    n_classes = int(z["n_classes"])
    S = z["seeds"].shape[0]

    # Primary: balanced-error contribution
    L_full = balanced_error_contributions(z["pred_full"], y_test, n_classes)      # (S, N)
    L_abl = balanced_error_contributions(z["pred_ablated"], y_test, n_classes)
    effect_primary = (L_abl - L_full)[None, :, :]                                 # (1, S, N)

    # Secondary: cross-entropy sensitivity metric
    effect_secondary = (z["loss_ablated"] - z["loss_full"])[None, :, :]

    primary = analyze_one_metric(effect_primary, rng)
    secondary = analyze_one_metric(effect_secondary, rng)

    return dict(
        task_id=int(z["task_id"]), dataset_id=int(z["dataset_id"]),
        n_test=y_test.shape[0], n_classes=n_classes,
        **{f"balanced_error_{k}": v for k, v in primary.items()},
        **{f"cross_entropy_{k}": v for k, v in secondary.items()},
    )


def main():
    _check_balanced_error_identity()

    manifest = pd.read_csv("manifest.csv")
    expected_task_ids = set(manifest.task_id.astype(int))

    found = {}
    for path in glob.glob(os.path.join(ARTIFACT_DIR, "task_*_fold_0.npz")):
        tid = int(os.path.basename(path).split("_")[1])
        found[tid] = path

    missing = expected_task_ids - set(found)
    extra = set(found) - expected_task_ids
    if missing:
        names = manifest[manifest.task_id.isin(missing)].name.tolist()
        print(f"INCOMPLETE: {len(missing)} / {len(expected_task_ids)} task(s) missing an artifact: "
              f"{names}")
        print("Refusing to compute medians over a partial set. Either finish the missing runs "
              "or record the shortfall as an explicit, named deviation before analyzing "
              "whatever is left.")
        return
    if extra:
        print(f"NOTE: {len(extra)} artifact(s) present that are not in manifest.csv "
              f"(task_ids {sorted(extra)}) -- ignoring them, but check why they exist.")

    print(f"All {len(expected_task_ids)} tasks present. Proceeding.")

    rng = np.random.default_rng(20260813)
    rows = [analyze_dataset(found[tid], rng) for tid in sorted(expected_task_ids)]
    out = pd.DataFrame(rows).merge(manifest[["task_id", "name"]], on="task_id").sort_values("name")
    out.to_csv("cc18_per_dataset_summary.csv", index=False)

    degenerate = out[out.balanced_error_degenerate_zero_variance | out.cross_entropy_degenerate_zero_variance]
    if len(degenerate):
        print(f"\nWARNING: {len(degenerate)} dataset(s) show exactly zero variance in both "
              f"components for at least one metric: {degenerate.name.tolist()}. This can be a "
              f"genuine null result (identical predictions under both configs, all seeds) or a "
              f"sign of a degenerate upstream run (e.g. an all-missing feature column collapsing "
              f"training to a majority-class predictor -- check Phase 2's log for that dataset, "
              f"in particular any sklearn RuntimeWarning about 'invalid value encountered in "
              f"divide' right before it, and whether every seed's best val balanced accuracy "
              f"stopped at exactly the majority-class rate). Do not average these into the "
              f"medians below without resolving which case it is.\n")

    print(out[["name", "n_test", "balanced_error_effect", "balanced_error_seed_share",
               "balanced_error_significant_full_t"]].to_string(index=False))

    print(f"\n[balanced error, primary]  median seed share: "
          f"{out.balanced_error_seed_share.median():.1f}%, "
          f">50% in {out.balanced_error_seed_share_gt_half.sum()}/{len(out)}, "
          f"individually significant (t_{{S-1}}): {out.balanced_error_significant_full_t.sum()}/{len(out)}")
    print(f"[cross-entropy, secondary] median seed share: "
          f"{out.cross_entropy_seed_share.median():.1f}%, "
          f">50% in {out.cross_entropy_seed_share_gt_half.sum()}/{len(out)}, "
          f"individually significant (t_{{S-1}}): {out.cross_entropy_significant_full_t.sum()}/{len(out)}")
    print("\nFeed cc18_per_dataset_summary.csv (balanced_error_effect / its CI, per dataset) "
          "into the existing REML + modified Hartung-Knapp pooling code used for "
          "households/ECL -- pooling is not reimplemented here.")


if __name__ == "__main__":
    main()
