"""Tests for src/features.py — the leakage-safe preprocessing contract.

The headline scientific claim depends on the preprocessor being honest: fit on
train only, and robust to KDDTest+ categories never seen in training. These
tests pin that behaviour down.
"""
from __future__ import annotations

import numpy as np

from src.data import CATEGORICAL_COLUMNS, FEATURE_COLUMNS
from src.features import (
    ENGINEERED_COLUMNS, add_engineered_features, build_preprocessor,
    feature_groups, redundant_drop_set,
)


def test_feature_groups_drops_constant(raw_frame):
    groups = feature_groups(raw_frame)
    # num_outbound_cmds is constant in the data -> dropped, never modelled.
    assert "num_outbound_cmds" in groups["dropped"]


def test_feature_groups_is_a_partition(raw_frame):
    groups = feature_groups(raw_frame)
    buckets = (groups["dropped"] + groups["categorical"]
               + groups["log_numeric"] + groups["plain_numeric"])
    # Every input feature lands in exactly one bucket (disjoint + complete).
    assert sorted(buckets) == sorted(FEATURE_COLUMNS)
    assert len(buckets) == len(set(buckets))
    # The nominal columns (that survive the drop) are routed to 'categorical'.
    for c in CATEGORICAL_COLUMNS:
        assert c in groups["categorical"]


def test_feature_groups_respects_subset(raw_frame):
    cols = ["src_bytes", "dst_bytes", "protocol_type", "same_srv_rate"]
    groups = feature_groups(raw_frame, feature_cols=cols)
    seen = (groups["dropped"] + groups["categorical"]
            + groups["log_numeric"] + groups["plain_numeric"])
    assert sorted(seen) == sorted(cols)


def test_preprocessor_handles_unseen_categories_without_leakage(raw_frame):
    """Fit on train; transform a test row with an unseen service -> finite output.

    A label/ordinal encoder would crash or invent a code for the unseen value;
    OneHotEncoder(handle_unknown='ignore') must encode it as all-zeros instead.
    """
    train = raw_frame.iloc[:40].copy()
    test = raw_frame.iloc[40:].copy()
    test.loc[test.index[0], "service"] = "totally_unseen_service"

    groups = feature_groups(train)
    pre = build_preprocessor(groups)
    pre.fit(train[FEATURE_COLUMNS])

    out_train = pre.transform(train[FEATURE_COLUMNS])
    out_test = pre.transform(test[FEATURE_COLUMNS])

    assert out_train.shape[1] == out_test.shape[1]  # stable feature space
    assert np.isfinite(out_test).all()              # no NaN/inf from unseen value


def test_redundant_drop_set_never_drops_both_of_a_pair():
    pairs = [
        {"feat_a": "num_root", "feat_b": "num_compromised"},
        {"feat_a": "srv_serror_rate", "feat_b": "serror_rate"},
        # Overlapping pair: num_root already kept, so it must not be dropped.
        {"feat_a": "num_root", "feat_b": "dst_host_count"},
    ]
    drop = redundant_drop_set(pairs)
    # Survivors (feat_a side) are never in the drop set.
    assert "num_root" not in drop
    assert "srv_serror_rate" not in drop
    # The dropped members are present, and nothing is dropped twice.
    assert set(drop) == {"num_compromised", "serror_rate", "dst_host_count"}
    assert len(drop) == len(set(drop))


def test_add_engineered_features(raw_frame):
    out = add_engineered_features(raw_frame)
    for col in ENGINEERED_COLUMNS:
        assert col in out.columns
    # Original columns are preserved (non-destructive).
    assert set(FEATURE_COLUMNS).issubset(out.columns)
    # bytes_ratio uses +1 smoothing -> always finite, even when dst_bytes == 0.
    z = raw_frame.copy()
    z["dst_bytes"] = 0.0
    assert np.isfinite(add_engineered_features(z)["bytes_ratio"]).all()
