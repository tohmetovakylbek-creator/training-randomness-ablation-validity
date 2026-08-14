"""
Phase 2 (v2) -- incorporates the department head's corrections:
  1. Official OpenML split (task.get_train_test_split_indices), not an
     ad-hoc train_test_split. Without this, cite the sample as "datasets
     drawn from CC18", never "CC18 tasks".
  2. Preprocessing (categorical encoding, numeric imputation, scaling)
     fit on the official training fold ONLY, then applied to test --
     the original script leaked test-set categories/statistics into
     fit_transform on the full data.
  3. NO artificial "blocks". Official CC18 test examples are exchangeable
     (CC-18 excludes time series by construction); a random partition
     into 10-40 "windows" has no real neighbourhood structure, so a
     moving-block bootstrap over it is not a conservative correction --
     it is noise stacked on top of noise. The fix is an ORDINARY
     bootstrap over individual test examples, which is exactly
     block_bootstrap_vwin(..., block=1, ...) from seed_window_simgrid.py
     applied with W = n_test (each example is its own column) -- see
     03_analyze.py. This script therefore no longer constructs blocks at
     all; it saves per-example arrays.
  4. Per-example NPZ output (loss, prediction, true label), not
     pre-aggregated block means -- so metric, bootstrap, and block
     construction (or lack thereof) can all be revisited later without
     retraining.
  5. Cross-entropy is kept as a per-example "sensitivity" quantity; the
     primary effect is computed downstream (Phase 3) from pred_full /
     pred_ablated / y_test as a balanced-error contribution, so both are
     recoverable from this file's output without retraining.
  6. Early stopping on a held-out validation split carved out of the
     official training fold (same validation examples across all
     seeds/configs for a given dataset, so the comparison stays paired).
  7. SEEDS matched to the rest of the paper.
  8. Resumable: one NPZ per task, existing files are skipped on rerun.
  9. Completeness is NOT decided here -- this script just records
     failures; 03_analyze.py refuses to compute anything until exactly
     len(manifest) successful NPZ files are present.

Run on: desktop (GPU). Needs manifest.csv (from Phase 1) in the same
directory. Output: one artifacts/task_<task_id>_fold_0.npz per dataset.
"""
import json
import os

import numpy as np
import openml
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from ft_transformer import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 7, 13, 99, 2025]         # matched to the rest of the paper
SPLIT_SEED = 20260813                 # train/val split, fixed across seeds/configs
MAX_EPOCHS = 100
PATIENCE = 12
BATCH_SIZE = 256
LR = 1e-3
VAL_FRACTION = 0.10                   # carved out of the official training fold
OUT_DIR = "artifacts"

os.makedirs(OUT_DIR, exist_ok=True)


def load_task_official_split(task_id):
    """Fetch the dataset and the OFFICIAL OpenML train/test indices (fold 0)."""
    task = openml.tasks.get_task(task_id, download_data=True)
    ds = task.get_dataset()
    X, y, cat_mask, _ = ds.get_data(target=task.target_name)

    train_idx, test_idx = task.get_train_test_split_indices(repeat=0, fold=0, sample=0)

    num_cols = [c for c, is_cat in zip(X.columns, cat_mask) if not is_cat]
    cat_cols = [c for c, is_cat in zip(X.columns, cat_mask) if is_cat]

    y_all = pd.Categorical(y).codes.astype("int64")
    n_classes = int(y_all.max() + 1)

    return X, num_cols, cat_cols, y_all, n_classes, train_idx, test_idx, task.dataset_id


def fit_preprocessing(X, num_cols, cat_cols, train_idx):
    """Fit encoder/scaler/imputer on the TRAIN fold only -- returns transformer state."""
    state = {}

    if num_cols:
        X_num_train_raw = X.iloc[train_idx][num_cols].apply(pd.to_numeric, errors="coerce")
        medians = X_num_train_raw.median()
        # Guard against a column that is entirely missing in the training fold (e.g.
        # "sick"/TBG, which OpenML's real hold-out fold has 100% NaN for): median() of
        # an all-NaN column is itself NaN, so fillna(medians) would be a no-op and NaN
        # would silently reach StandardScaler.fit -- which sklearn does NOT raise on;
        # it emits "invalid value encountered in divide" and produces a scaler that
        # propagates NaN through the whole model at inference (a single NaN token
        # poisons every other token via attention). Fall back to 0.0 for any column
        # whose median is undefined; 0.0 post-scaling is the "no information" value
        # for a column that carries none anyway.
        medians = medians.fillna(0.0)
        state["num_medians"] = medians
        X_num_train = X_num_train_raw.fillna(medians).values.astype("float32")
        scaler = StandardScaler().fit(X_num_train)
        state["scaler"] = scaler
    else:
        state["num_medians"] = None
        state["scaler"] = None

    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        enc.fit(X.iloc[train_idx][cat_cols].astype(str))
        state["cat_encoder"] = enc
        state["cat_cardinalities"] = [len(c) + 1 for c in enc.categories_]  # +1 for unseen
    else:
        state["cat_encoder"] = None
        state["cat_cardinalities"] = []

    return state


def apply_preprocessing(X, num_cols, cat_cols, idx, state):
    """Transform a given index set using transformers already fit on train."""
    if num_cols:
        X_num_raw = X.iloc[idx][num_cols].apply(pd.to_numeric, errors="coerce")
        X_num_raw = X_num_raw.fillna(state["num_medians"])
        X_num = state["scaler"].transform(X_num_raw.values.astype("float32"))
    else:
        X_num = np.zeros((len(idx), 0), dtype="float32")

    if cat_cols:
        X_cat_raw = state["cat_encoder"].transform(X.iloc[idx][cat_cols].astype(str))
        cardinalities = state["cat_cardinalities"]
        for j, card in enumerate(cardinalities):
            col = X_cat_raw[:, j]
            col[col < 0] = card - 1                 # unseen category -> dedicated bucket
        X_cat = X_cat_raw.astype("int64")
    else:
        X_cat = np.zeros((len(idx), 0), dtype="int64")

    return X_num.astype("float32"), X_cat


def _assert_finite(X_num, name):
    if X_num.size and not np.isfinite(X_num).all():
        bad_cols = np.where(~np.isfinite(X_num).all(axis=0))[0]
        raise ValueError(
            f"Non-finite values in numeric features after preprocessing ({name}), "
            f"columns {bad_cols.tolist()}. Do not train on this silently -- find out "
            f"why (e.g. an all-missing column in this split) before proceeding."
        )


def train_one_run(X_num_tr, X_cat_tr, y_tr, X_num_val, X_cat_val, y_val,
                   cardinalities, n_classes, use_identity, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model(X_num_tr.shape[1], cardinalities, n_classes, use_identity, seed).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

    # Inverse-frequency class weights for the TRAINING objective only. Without this,
    # an imbalanced dataset (e.g. "sick", ~6% positive) has a trivial low-CE local
    # optimum -- always predict the majority class -- which early stopping can lock
    # in before the model ever separates the classes, identically for both configs
    # and every seed (same init sequence regardless of use_identity), producing
    # exactly-zero variance that has nothing to do with the ablated mechanism.
    # Evaluation (balanced accuracy for early stopping, balanced-error and raw CE
    # for the reported effect) is intentionally left unweighted/as-is elsewhere.
    class_counts = np.bincount(y_tr, minlength=n_classes).astype("float64")
    class_weights = class_counts.sum() / (n_classes * np.maximum(class_counts, 1))
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)

    Xn = torch.tensor(X_num_tr, device=DEVICE)
    Xc = torch.tensor(X_cat_tr, device=DEVICE)
    yt = torch.tensor(y_tr, device=DEVICE)
    n = Xn.shape[0]

    Xn_val = torch.tensor(X_num_val, device=DEVICE)
    Xc_val = torch.tensor(X_cat_val, device=DEVICE)
    y_val_np = y_val

    best_val_bacc = -np.inf
    best_state = None
    epochs_since_improve = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            logits = model(Xn[idx], Xc[idx])
            loss = F.cross_entropy(logits, yt[idx], weight=class_weights_t)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(Xn_val, Xc_val)
            val_pred = val_logits.argmax(dim=1).cpu().numpy()
        val_bacc = balanced_accuracy(y_val_np, val_pred, n_classes)

        if val_bacc > best_val_bacc:
            best_val_bacc = val_bacc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, best_val_bacc, epoch + 1


def balanced_accuracy(y_true, y_pred, n_classes):
    accs = []
    for k in range(n_classes):
        mask = y_true == k
        if mask.sum() == 0:
            continue
        accs.append((y_pred[mask] == k).mean())
    return float(np.mean(accs)) if accs else 0.0


@torch.no_grad()
def evaluate(model, X_num_te, X_cat_te, y_te):
    model.eval()
    Xn = torch.tensor(X_num_te, device=DEVICE)
    Xc = torch.tensor(X_cat_te, device=DEVICE)
    yt = torch.tensor(y_te, device=DEVICE)
    logits = model(Xn, Xc)
    per_ex_loss = F.cross_entropy(logits, yt, reduction="none").cpu().numpy()
    pred = logits.argmax(dim=1).cpu().numpy()
    return per_ex_loss, pred


def run_dataset(row):
    out_path = os.path.join(OUT_DIR, f"task_{row['task_id']}_fold_0.npz")
    if os.path.exists(out_path):
        print(f"  {row['name']}: artifact already exists, skipping (resumable)")
        return "skipped"

    print(f"\n=== {row['name']} (task {row['task_id']}) ===")
    (X, num_cols, cat_cols, y_all, n_classes,
     train_idx, test_idx, dataset_id) = load_task_official_split(row["task_id"])
    print(f"  official split: {len(train_idx)} train / {len(test_idx)} test "
          f"(estimated at selection time: {row['n_test_est']})")

    train_sub_idx, val_idx = train_test_split(
        train_idx, test_size=VAL_FRACTION, random_state=SPLIT_SEED,
        stratify=y_all[train_idx],
    )

    state = fit_preprocessing(X, num_cols, cat_cols, train_sub_idx)  # fit on train-sub only
    X_num_tr, X_cat_tr = apply_preprocessing(X, num_cols, cat_cols, train_sub_idx, state)
    X_num_val, X_cat_val = apply_preprocessing(X, num_cols, cat_cols, val_idx, state)
    X_num_te, X_cat_te = apply_preprocessing(X, num_cols, cat_cols, test_idx, state)
    for arr, split_name in [(X_num_tr, "train"), (X_num_val, "val"), (X_num_te, "test")]:
        _assert_finite(arr, f"{row['name']}/{split_name}")
    y_tr, y_val, y_te = y_all[train_sub_idx], y_all[val_idx], y_all[test_idx]
    cardinalities = state["cat_cardinalities"]

    n_test = len(test_idx)
    loss_full = np.zeros((len(SEEDS), n_test))
    loss_ablated = np.zeros((len(SEEDS), n_test))
    pred_full = np.zeros((len(SEEDS), n_test), dtype="int64")
    pred_ablated = np.zeros((len(SEEDS), n_test), dtype="int64")

    for si, seed in enumerate(SEEDS):
        for use_identity, loss_arr, pred_arr in [
            (True, loss_full, pred_full), (False, loss_ablated, pred_ablated)
        ]:
            model, best_val_bacc, n_epochs = train_one_run(
                X_num_tr, X_cat_tr, y_tr, X_num_val, X_cat_val, y_val,
                cardinalities, n_classes, use_identity, seed,
            )
            per_ex_loss, pred = evaluate(model, X_num_te, X_cat_te, y_te)
            loss_arr[si] = per_ex_loss
            pred_arr[si] = pred
            print(f"  seed {seed}, identity={use_identity}: "
                  f"stopped at epoch {n_epochs}, best val balanced acc = {best_val_bacc:.4f}")

    np.savez(
        out_path,
        loss_full=loss_full, loss_ablated=loss_ablated,
        pred_full=pred_full, pred_ablated=pred_ablated,
        y_test=y_te, test_indices=np.asarray(test_idx),
        seeds=np.asarray(SEEDS), task_id=row["task_id"], dataset_id=dataset_id,
        n_classes=n_classes,
    )
    print(f"  wrote {out_path}")
    return "ok"


def main():
    manifest = pd.read_csv("manifest.csv")

    # Sanity check flagged separately from the department head's memo: n_test_est
    # in the manifest assumed an 80/20 holdout; the OFFICIAL OpenML split is
    # usually a 10-fold CV fold (~10%), not 20%. Print actual official test-fold
    # sizes for all 17 up front, before spending any GPU time, and flag anything
    # that looks too small for a stable balanced-error / bootstrap estimate.
    print("Checking official split sizes for all tasks before training...")
    small = []
    for _, row in manifest.iterrows():
        try:
            task = openml.tasks.get_task(row["task_id"], download_data=False)
            tr, te = task.get_train_test_split_indices(repeat=0, fold=0, sample=0)
            print(f"  {row['name']:38s} official test fold: {len(te)} "
                  f"(Phase-1 estimate was {row['n_test_est']})")
            if len(te) < 150:
                small.append((row["name"], len(te)))
        except Exception as e:
            print(f"  {row['name']}: could not check split ({e})")
    if small:
        print(f"\nWARNING: {len(small)} task(s) have official test folds under 150 rows: {small}")
        print("Consider swapping these for another already-vetted candidate before proceeding.\n")

    results = {}
    for _, row in manifest.iterrows():
        try:
            results[row["name"]] = run_dataset(row)
        except Exception as e:
            print(f"  FAILED on {row['name']}: {e} -- log this in the deviation log, do not silently drop")
            results[row["name"]] = f"FAILED: {e}"

    ok = sum(1 for v in results.values() if v in ("ok", "skipped"))
    print(f"\n{ok} / {len(manifest)} tasks have an artifact on disk.")
    failures = {k: v for k, v in results.items() if v not in ("ok", "skipped")}
    if failures:
        print(f"FAILURES ({len(failures)}): {json.dumps(failures, indent=2)}")
        print("Do NOT proceed to Phase 3 until every task has a successful artifact "
              "or the shortfall is recorded as an explicit deviation.")


if __name__ == "__main__":
    main()
