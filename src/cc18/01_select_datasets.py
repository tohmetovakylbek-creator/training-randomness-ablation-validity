"""
Phase 1 — reproducible selection of 17 datasets from OpenML-CC18.
Run on: laptop (no GPU needed, just network access to openml.org).

This script is the *entire* preregistered selection rule. Run it once,
commit the two output files (data/manifests/cc18_manifest.json,
data/manifests/cc18_manifest.csv) to the repo BEFORE any model is
trained, and never edit them by hand afterwards.
If the rule needs to change, that is a deviation and goes in the
deviation log, not a silent edit here.

Selection rule (state this verbatim in the registration document):
  1. Universe: OpenML-CC18 (suite id 99), all 72 tasks.
  2. Keep tasks whose default 80/20 stratified holdout leaves a test
     fold of at least 250 rows, so that the adaptive block count
     W = min(40, max(10, floor(n_test / 25))) gives at least 10 blocks
     with a floor of 25 rows/block.
  3. Keep tasks with <= 20 classes (matches the CC-18/TabZilla
     convention of excluding very-high-cardinality targets, e.g.
     letter, isolet).
  4. From the surviving pool, draw 17 tasks by simple random sample
     with SEED = 20260813, no replacement.
  5. Report, but do not use for selection, whether each drawn task is
     also in the TabZilla-hard 36 (McElfresh et al., 2023) -- treat
     that overlap as a robustness note, not a selection criterion,
     because the TabZilla-hard list used here was reconstructed from
     secondary sources and has not been verified byte-for-byte against
     https://github.com/naszilla/tabzilla. Re-verify before citing the
     overlap count in the paper.
  6. Exclude, by name, any dataset with topical overlap with the paper's
     own domain (energy/time-series), features with no semantic identity
     (raw pixels, pixel-derived transforms, digitizer stroke
     coordinates), windowed sensor/signal data, or anonymized/unnamed
     features -- see EXCLUDE_NAMES / EXCLUDE_NAME_PATTERNS below and
     their deviation-log rationale. Round 1 covered the first two
     categories; round 2 added the mfeat-* pattern generalization plus
     the sensor-signal and anonymized-feature categories. Before drawing
     a third time, dump the full remaining candidate pool
     (full_pool_for_review.csv, written below) and classify all of it in
     one pass rather than reacting name-by-name again.
"""
import json
import os
import random

import openml
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
MANIFEST_DIR = os.path.join(REPO_ROOT, "data", "manifests")
FULL_POOL_PATH = os.path.join(MANIFEST_DIR, "cc18_full_pool_for_review.csv")
MANIFEST_CSV_PATH = os.path.join(MANIFEST_DIR, "cc18_manifest.csv")
MANIFEST_JSON_PATH = os.path.join(MANIFEST_DIR, "cc18_manifest.json")

SEED = 20260813
N_DATASETS = 17
MIN_TEST_ROWS = 250
MAX_CLASSES = 20

# Added after reviewing the first draw (2026-08-13), BEFORE any model was
# trained -- this is a preregistered filter tightening, not a post-hoc
# result-driven exclusion, and it must be logged as a deviation with the
# reasoning below.
#
# 1. Topical overlap with the paper's own domain: "electricity" (OpenML
#    task 219 / dataset 151, the Australian NSW electricity-market
#    series) is fundamentally a time-ordered price/demand series exposed
#    by OpenML as an i.i.d. classification task. Including it in a
#    benchmark meant to demonstrate the result outside energy and outside
#    forecasting would undercut the entire point of Section 4.
# 2. Features without a semantic identity: several CC-18 datasets are
#    UCI image or signal datasets flattened into numeric feature vectors
#    (raw pixel intensities, pixel-block counts, PCA components of pixel
#    grids, or digitizer stroke coordinates). The ablated mechanism in
#    this study is "does the shared encoder benefit from being told which
#    named feature it is looking at" -- that question is well posed for
#    age/income/wavelet-skewness and not well posed for "pixel 347", so
#    these are excluded regardless of whether they would otherwise pass
#    the numeric filters. This is applied as a *pattern* on the mfeat-*
#    family (all six variants), not case-by-case, after round 2 pulled a
#    variant (mfeat-morphological) that a name-by-name list had missed.
# 3. Windowed sensor/signal data (added after round 2): datasets built by
#    computing handcrafted statistics over sliding time-windows of a
#    physical sensor signal (accelerometer/gyroscope, sonar range
#    readings, etc.) are structurally the same kind of object as this
#    paper's own IIoT/forecasting inputs, just from a different
#    application. Excluded for the same reason as "electricity", one
#    level more general.
# 4. Anonymized or unnamed features (added after round 2): datasets whose
#    columns are deliberately obfuscated (e.g. "feature1", "feature2"
#    with no disclosed meaning) make the identity-ablation question
#    ("does the encoder benefit from knowing this is income") impossible
#    to state, independent of any domain-overlap concern.
#
# Round 1 exclusions (topical overlap / raw pixel data):
#   electricity, Fashion-MNIST, mfeat-pixel, optdigits, satimage,
#   mfeat-karhunen, pendigits
# Round 2 exclusions (missed image variants + two new categories):
#   CIFAR_10, mnist_784, texture, mfeat-morphological, har,
#   wall-robot-navigation, numerai28.6
# Round 3 exclusions (full-pool review of all 72 CC-18 names in one pass,
# rather than reacting to each redraw -- see full_pool_for_review.csv):
#   GesturePhaseSegmentationProcessed, ozone-level-8hr (sensor time-window,
#   same class as har/wall-robot-navigation); madelon (synthetic,
#   anonymized probe features, same class as numerai28.6); phoneme,
#   segment, semeion (audio/image-derived, same class as the pixel
#   exclusions); splice (near-duplicate of "dna" -- both are the UCI
#   splice-junction gene-sequence dataset under two encodings, 3186 vs
#   3190 instances, same 3 classes; keeping both would double-count one
#   biological dataset as two independent replication units, which
#   violates the independence assumption behind the crossed ANOVA /
#   REML-HK pooling -- keep "dna", drop "splice"). Verify the dna/splice
#   identification against OpenML's dataset descriptions before citing
#   it as fact in the paper; treated here as confirmed based on the
#   matching class structure and near-identical instance counts.
# Round 4 exclusions (two names not seen in the round-3 full-pool pass
# because they only surfaced once drawn):
#   Bioresponse (Kaggle 2012 molecular-activity data; features are
#   deliberately anonymized descriptors D1..D1776, same "no semantic
#   identity" class as numerai28.6/madelon, different domain); wilt
#   (aerial/satellite image-object features -- mean/stdev of spectral
#   bands, GLCM texture -- same "image/remote-sensing-derived" class as
#   satimage).
EXCLUDE_NAMES = {
    "electricity", "Fashion-MNIST", "mfeat-pixel", "optdigits", "satimage",
    "mfeat-karhunen", "pendigits",
    "CIFAR_10", "mnist_784", "texture", "mfeat-morphological",
    "har", "wall-robot-navigation", "numerai28.6",
    "GesturePhaseSegmentationProcessed", "ozone-level-8hr", "madelon",
    "phoneme", "segment", "semeion", "splice",
    "Bioresponse", "wilt",
}
# Any remaining mfeat-* variant (fourier, zernike, factors, ...) is caught
# by the pattern check below even if not named explicitly above --
# applying reason 2 as a pattern, not a per-name list, so a third round
# does not repeat the same mistake on the next mfeat-* variant.
EXCLUDE_NAME_PATTERNS = ("mfeat-",)

# If a redraw pulls a name not seen before, manually check it against
# reasons 1-4 above before accepting it -- do not assume the numeric
# filters alone are sufficient. Known other CC-18 names to watch for:
# isolet, letter, USPS (image/signal), semeion (pixel), one-hundred-plants-*
# (leaf-image-derived).

# Reconstructed from secondary sources (McElfresh et al., 2023 and
# citing papers); NOT fetched from the canonical repo at run time.
# Treat purely as a robustness annotation -- see step 5 above.
TABZILLA_HARD_NAMES = {
    "Australian", "Bioresponse", "GesturePhaseSegmentationProcessed",
    "MiniBooNE", "SpeedDating", "ada_agnostic", "airlines", "albert",
    "artificial-characters", "audiology", "balance-scale", "cnae-9",
    "colic", "credit-approval", "credit-g", "electricity", "elevators",
    "guillermo", "heart-h", "higgs", "jasmine",
    "jungle_chess_2pcs_raw_endgame_complete", "kc1", "lymph",
    "mfeat-fourier", "mfeat-zernike", "monks-problems-2", "nomao",
    "one-hundred-plants-texture", "phoneme", "poker-hand", "profb",
    "socmob", "splice", "vehicle", "100-plants-texture",
}


def main():
    print("Fetching OpenML-CC18 (suite 99) ...")
    suite = openml.study.get_suite(99)
    task_ids = list(suite.tasks)
    print(f"  {len(task_ids)} tasks in the suite")

    rows = []
    for tid in task_ids:
        try:
            task = openml.tasks.get_task(tid, download_data=False)
            ds = openml.datasets.get_dataset(
                task.dataset_id, download_data=False, download_qualities=True
            )
            q = ds.qualities or {}
            n_instances = int(q.get("NumberOfInstances", 0) or 0)
            n_classes = int(q.get("NumberOfClasses", 0) or 0)
            n_test = round(n_instances * 0.2)
            rows.append(dict(
                task_id=tid,
                dataset_id=task.dataset_id,
                name=ds.name,
                n_instances=n_instances,
                n_classes=n_classes,
                n_test_est=n_test,
                in_tabzilla_hard_unverified=ds.name in TABZILLA_HARD_NAMES,
            ))
        except Exception as e:
            print(f"  skipping task {tid}: {e}")

    df = pd.DataFrame(rows)
    df["w_blocks"] = (df.n_test_est / 25).apply(lambda x: min(40, max(10, int(x))))

    # Dump the full pool (pre-filter) once, so a human can eyeball-classify
    # every remaining CC-18 name in one pass instead of discovering
    # problem datasets one redraw at a time.
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    df.sort_values("name").to_csv(FULL_POOL_PATH, index=False)
    print(f"  wrote {FULL_POOL_PATH} ({len(df)} datasets) -- "
          f"review names before the next redraw instead of reacting round-by-round")

    name_pattern_hit = df.name.str.lower().str.startswith(EXCLUDE_NAME_PATTERNS)
    pool = df[(df.n_test_est >= MIN_TEST_ROWS) & (df.n_classes <= MAX_CLASSES) & (df.n_classes >= 2)
              & (~df.name.isin(EXCLUDE_NAMES)) & (~name_pattern_hit)]
    print(f"  {len(pool)} / {len(df)} tasks pass the preregistered filters "
          f"(after excluding {len(EXCLUDE_NAMES)} named datasets and the "
          f"{name_pattern_hit.sum()} matching EXCLUDE_NAME_PATTERNS)")

    rng = random.Random(SEED)
    chosen_ids = rng.sample(list(pool.task_id), k=min(N_DATASETS, len(pool)))
    chosen = pool[pool.task_id.isin(chosen_ids)].sort_values("name").reset_index(drop=True)

    print(f"\nSelected {len(chosen)} datasets (seed={SEED}):")
    print(chosen[["task_id", "name", "n_instances", "n_classes", "w_blocks",
                   "in_tabzilla_hard_unverified"]].to_string(index=False))
    print(f"\nOverlap with TabZilla-hard (unverified list): "
          f"{chosen.in_tabzilla_hard_unverified.sum()} / {len(chosen)}")

    chosen.to_csv(MANIFEST_CSV_PATH, index=False)
    with open(MANIFEST_JSON_PATH, "w") as f:
        json.dump(dict(
            seed=SEED,
            selection_rule="CC-18 (suite 99), n_test>=250, n_classes in [2,20], "
                            "excluding named datasets in EXCLUDE_NAMES and any dataset matching "
                            "EXCLUDE_NAME_PATTERNS (topical overlap with the paper's own domain, "
                            "features without a semantic identity, windowed sensor/signal data, "
                            "or anonymized/unnamed features), simple random sample without replacement",
            excluded_names=sorted(EXCLUDE_NAMES),
            excluded_name_patterns=list(EXCLUDE_NAME_PATTERNS),
            n_datasets=len(chosen),
            tasks=chosen.to_dict(orient="records"),
        ), f, indent=2)
    print(f"\nWrote {MANIFEST_CSV_PATH} and {MANIFEST_JSON_PATH}. Commit both before Phase 2.")


if __name__ == "__main__":
    main()
