"""Integration tests on the bundled NSL-KDD data (marked ``slow``).

These exercise the real pipeline end-to-end and lock in the study's central
empirical finding as a regression test: the same model that scores ~99% on a
tutorial-style random split collapses on the official KDDTest+ set.

Skipped automatically when the data is absent; run just these with
``pytest -m slow`` or skip them with ``pytest -m "not slow"``.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data import FEATURE_COLUMNS, load_train_test
from src.features import build_preprocessor, feature_groups

pytestmark = pytest.mark.slow

RANDOM_STATE = 42
TRAIN_SUBSAMPLE = 20_000  # keep the fit fast while staying representative


def _rf_pipeline(train):
    pre = build_preprocessor(feature_groups(train))
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                 random_state=RANDOM_STATE)
    return Pipeline([("prep", pre), ("clf", clf)])


def test_real_data_loads_with_full_label_coverage(real_raw_dir):
    # load_train_test calls check_label_coverage internally; reaching here means
    # every fine-grained label in both partitions mapped to a known family.
    train, test = load_train_test(real_raw_dir)
    assert len(train) > 100_000 and len(test) > 20_000
    assert set(train["attack_category"].unique()) <= {
        "normal", "DoS", "Probe", "R2L", "U2R"}


def test_preprocessor_transforms_official_testset_cleanly(real_raw_dir):
    train, test = load_train_test(real_raw_dir)
    pre = build_preprocessor(feature_groups(train))
    pre.fit(train[FEATURE_COLUMNS])
    out_test = pre.transform(test[FEATURE_COLUMNS])
    # KDDTest+ has services unseen in training; output must stay finite & aligned.
    assert out_test.shape[1] == pre.transform(train[FEATURE_COLUMNS][:5]).shape[1]
    assert np.isfinite(out_test).all()


def test_accuracy_collapses_from_random_split_to_official_test(real_raw_dir):
    """The thesis, as a regression test: Protocol A >> Protocol B."""
    train, test = load_train_test(real_raw_dir)
    train = train.sample(TRAIN_SUBSAMPLE, random_state=RANDOM_STATE)

    X_train, y_train = train[FEATURE_COLUMNS], train["is_attack"]
    X_test, y_test = test[FEATURE_COLUMNS], test["is_attack"]

    # Protocol A: tutorial-style random 80/20 split of the training data.
    Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE,
        stratify=y_train)
    acc_a = _rf_pipeline(train).fit(Xa_tr, ya_tr).score(Xa_te, ya_te)

    # Protocol B: train on KDDTrain+, evaluate on the official KDDTest+.
    acc_b = _rf_pipeline(train).fit(X_train, y_train).score(X_test, y_test)

    assert acc_a > 0.95, f"random-split accuracy unexpectedly low: {acc_a:.3f}"
    assert acc_a - acc_b > 0.10, (
        f"expected a large A->B drop, got A={acc_a:.3f} B={acc_b:.3f}")
