"""
Phase 0.5 -- preflight data-quality check. NO training, NO GPU. Runs in
seconds to low minutes for all 17 tasks combined, because it only fetches
data and runs the same (cheap, CPU-only) preprocessing fit that
02_run_experiment.py does -- it stops before ever building a model.

Exists because two separate multi-hour training runs were burned on
problems this script would have caught immediately:
  - "sick": an entirely-missing numeric column reaching StandardScaler
    (silently produces NaN features -- now also defended against directly
    in 02_run_experiment.py, but this script tells you about it up front
    instead of via a downstream crash or a degenerate 0.5 accuracy that
    looks like normal output).
  - "sick" again: ~6% positive class, enough to make an unweighted
    training objective (now fixed, but this script would have flagged
    the imbalance ratio before that fix existed too).

Run this BEFORE every 02_run_experiment.py run from now on, including
after any change to the dataset selection or the preprocessing code.

Checks per task, using ONLY the official training fold (no leakage):
  1. Any numeric column with >95% missing (median is undefined or
     near-degenerate) -- these get silently zero-filled by
     02_run_experiment.py now, but you should know they exist.
  2. Any numeric column with zero variance after imputation (constant
     column -- uninformative, and a red flag if it wasn't expected to be).
  3. Class imbalance: minority class fraction. Anything under 10% is
     flagged as "likely needs class-weighted loss to avoid majority-class
     collapse" (which 02_run_experiment.py now does for every task, but
     severe cases, like <2%, deserve a second look regardless).
  4. Any categorical column with a single category in the training fold
     (uninformative, same spirit as check 2).
  5. Train/val/test sizes actually available after the official split and
     the internal 90/10 validation carve-out, since Phase 1's n_test_est
     assumed an 80/20 split and the real official folds turned out
     smaller for several tasks.

Nothing here is a hard failure -- it's a report to read BEFORE spending
GPU time, not a gate that blocks the real run.
"""
import os

import numpy as np
import openml
import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
MANIFEST_PATH = os.path.join(REPO_ROOT, "data", "manifests", "cc18_manifest.csv")
REPORT_PATH = os.path.join(REPO_ROOT, "results", "cc18", "cc18_preflight_report.csv")

VAL_FRACTION = 0.10
SPLIT_SEED = 20260813
IMBALANCE_WARN_THRESHOLD = 0.10
SEVERE_IMBALANCE_THRESHOLD = 0.02
MISSING_WARN_THRESHOLD = 0.95


def check_one_task(row):
    issues = []
    task = openml.tasks.get_task(row["task_id"], download_data=True)
    ds = task.get_dataset()
    X, y, cat_mask, _ = ds.get_data(target=task.target_name)
    y_all = pd.Categorical(y).codes.astype("int64")
    n_classes = int(y_all.max() + 1)

    train_idx, test_idx = task.get_train_test_split_indices(repeat=0, fold=0, sample=0)
    train_sub_idx, val_idx = train_test_split(
        train_idx, test_size=VAL_FRACTION, random_state=SPLIT_SEED,
        stratify=y_all[train_idx],
    )

    num_cols = [c for c, is_cat in zip(X.columns, cat_mask) if not is_cat]
    cat_cols = [c for c, is_cat in zip(X.columns, cat_mask) if is_cat]

    # Checks 1-2: numeric columns
    if num_cols:
        X_num_train = X.iloc[train_sub_idx][num_cols].apply(pd.to_numeric, errors="coerce")
        missing_frac = X_num_train.isna().mean()
        for col, frac in missing_frac.items():
            if frac >= MISSING_WARN_THRESHOLD:
                issues.append(f"numeric column '{col}' is {frac:.0%} missing in the training "
                               f"fold (will be zero-filled)")
        filled = X_num_train.fillna(X_num_train.median().fillna(0.0))
        zero_var = filled.std() == 0
        for col in zero_var[zero_var].index:
            issues.append(f"numeric column '{col}' has zero variance after imputation "
                           f"(constant, uninformative)")

    # Check 4: categorical columns with a single category
    if cat_cols:
        X_cat_train = X.iloc[train_sub_idx][cat_cols].astype(str)
        for col in cat_cols:
            if X_cat_train[col].nunique() <= 1:
                issues.append(f"categorical column '{col}' has a single category in the "
                               f"training fold (uninformative)")

    # Check 3: class imbalance
    counts = np.bincount(y_all[train_sub_idx], minlength=n_classes)
    min_frac = counts.min() / counts.sum()
    if min_frac < SEVERE_IMBALANCE_THRESHOLD:
        issues.append(f"SEVERE class imbalance: smallest class is {min_frac:.1%} of training "
                       f"data (class-weighted loss is on, but this is worth a second look; "
                       f"consider whether balanced accuracy is even a stable target at n_test="
                       f"{len(test_idx)})")
    elif min_frac < IMBALANCE_WARN_THRESHOLD:
        issues.append(f"class imbalance: smallest class is {min_frac:.1%} of training data "
                       f"(class-weighted loss handles this, flagging for awareness)")

    return dict(
        name=row["name"], task_id=row["task_id"],
        n_train=len(train_sub_idx), n_val=len(val_idx), n_test=len(test_idx),
        n_classes=n_classes, min_class_frac=round(100 * min_frac, 1),
        n_issues=len(issues), issues="; ".join(issues) if issues else "",
    )


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    rows = []
    for _, row in manifest.iterrows():
        print(f"checking {row['name']}...")
        try:
            rows.append(check_one_task(row))
        except Exception as e:
            rows.append(dict(name=row["name"], task_id=row["task_id"], n_train=None,
                              n_val=None, n_test=None, n_classes=None, min_class_frac=None,
                              n_issues=-1, issues=f"COULD NOT CHECK: {e}"))

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    out.to_csv(REPORT_PATH, index=False)

    print("\n" + "=" * 100)
    flagged = out[out.n_issues != 0]
    if len(flagged) == 0:
        print("No issues found on any of the 17 tasks. Safe to run 02_run_experiment.py.")
    else:
        print(f"{len(flagged)} / {len(out)} task(s) have something worth reading before you "
              f"spend GPU time:\n")
        for _, r in flagged.iterrows():
            print(f"  {r['name']} (task {r['task_id']}): {r['issues']}")
    print("=" * 100)
    print(f"\nFull report written to {REPORT_PATH} "
          f"(includes n_train/n_val/n_test/n_classes/min_class_frac for all 17, "
          f"not just the flagged ones).")


if __name__ == "__main__":
    main()
