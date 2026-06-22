"""Tests for src/unsw.py — the modern-dataset (UNSW-NB15) loader and grouping.

UNSW-NB15 is the external-validation dataset (§7 of the analysis). These tests
pin down its schema, the label normalisation that lets the NSL-KDD helpers be
reused verbatim, and the leakage-safe feature grouping.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import unsw
from src.features import build_preprocessor


def _make_unsw_frame(n: int = 40) -> pd.DataFrame:
    """A small in-memory UNSW-NB15 frame with the real column names + targets."""
    rng = np.arange(n)
    data = {c: (rng % 5).astype(float) for c in unsw.FEATURE_COLUMNS}
    data["proto"] = np.take(["tcp", "udp", "arp"], rng % 3)
    data["service"] = np.take(["-", "http", "dns"], rng % 3)
    data["state"] = np.take(["FIN", "INT", "CON"], rng % 3)
    data["sbytes"] = (rng % 5 * 1000.0)        # heavy-tailed: max > 100 -> log
    data["tcprtt"] = (rng % 5) / 100.0          # small rate: max < 1 -> plain
    data["is_ftp_login"] = (rng % 2)            # 0/1 flag -> plain
    attack = np.take(["Normal", "Generic", "Exploits", "Worms"], rng % 4)
    data["attack_cat"] = attack
    data["label"] = (attack != "Normal").astype(int)
    data["id"] = rng
    return pd.DataFrame(data)


@pytest.fixture
def unsw_frame() -> pd.DataFrame:
    return _make_unsw_frame()


@pytest.fixture
def unsw_file(tmp_path, unsw_frame):
    path = tmp_path / "unsw.csv"
    unsw_frame.to_csv(path, index=False)  # UNSW CSVs ship WITH a header
    return path


def test_schema_shapes():
    assert len(unsw.FEATURE_COLUMNS) == 42
    assert set(unsw.CATEGORICAL_COLUMNS) == {"proto", "service", "state"}
    assert unsw.CLASS_ORDER[0] == "normal"        # normal-class normalised to lower-case
    assert len(unsw.CLASS_ORDER) == 10            # normal + 9 attack families
    assert set(unsw.RARE_ATTACKS) <= set(unsw.CLASS_ORDER)
    # 'id'/'attack_cat'/'label' are targets/index, never modelled.
    for leak in ("id", "attack_cat", "label"):
        assert leak not in unsw.FEATURE_COLUMNS


def test_load_unsw_normalises_normal_and_derives_targets(unsw_file):
    df = unsw.load_unsw(unsw_file)
    # 'Normal' -> 'normal' so the NSL-KDD 'normal'-row conventions carry over.
    assert "Normal" not in set(df["attack_category"])
    assert "normal" in set(df["attack_category"])
    # is_attack mirrors the binary label, binary_label mirrors is_attack.
    assert (df["is_attack"] == df["label"]).all()
    assert (df.loc[df["is_attack"] == 0, "binary_label"] == "normal").all()
    assert (df.loc[df["is_attack"] == 1, "binary_label"] == "attack").all()


def test_feature_matrix_excludes_targets(unsw_file):
    df = unsw.load_unsw(unsw_file)
    fm = unsw.feature_matrix(df)
    assert list(fm.columns) == unsw.FEATURE_COLUMNS
    for leak in ("id", "attack_cat", "label", "is_attack"):
        assert leak not in fm.columns


def test_feature_groups_partition_and_log_heuristic(unsw_frame):
    groups = unsw.feature_groups(unsw_frame)
    buckets = (groups["dropped"] + groups["categorical"]
               + groups["log_numeric"] + groups["plain_numeric"])
    assert sorted(buckets) == sorted(unsw.FEATURE_COLUMNS)   # partition
    assert len(buckets) == len(set(buckets))                  # disjoint
    for c in unsw.CATEGORICAL_COLUMNS:
        assert c in groups["categorical"]
    # heavy-tailed numeric -> log; bounded rate / 0-1 flag -> plain.
    assert "sbytes" in groups["log_numeric"]
    assert "tcprtt" in groups["plain_numeric"]
    assert "is_ftp_login" in groups["plain_numeric"]


def test_preprocessor_builds_from_unsw_groups(unsw_frame):
    # The generic build_preprocessor consumes UNSW groups unchanged.
    pre = build_preprocessor(unsw.feature_groups(unsw_frame))
    out = pre.fit_transform(unsw.feature_matrix(unsw_frame))
    assert out.shape[0] == len(unsw_frame)
    assert np.isfinite(out).all()


# --- integration on the bundled data -----------------------------------------
@pytest.mark.slow
def test_real_unsw_loads_and_preprocesses(real_unsw_dir):
    train, test = unsw.load_train_test(real_unsw_dir)
    assert train.shape[0] == 175_341 and test.shape[0] == 82_332
    assert set(train["attack_category"].unique()) <= set(unsw.CLASS_ORDER)
    # Test set has states unseen in training (ACC/CLO); preprocessing must cope.
    pre = build_preprocessor(unsw.feature_groups(train))
    pre.fit(unsw.feature_matrix(train))
    out_test = pre.transform(unsw.feature_matrix(test))
    assert np.isfinite(out_test).all()
