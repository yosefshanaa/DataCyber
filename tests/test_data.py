"""Tests for src/data.py — schema integrity and the attack-label taxonomy.

The label mapping is load-bearing for the whole study: the per-class (R2L/U2R)
analysis is only trustworthy if every fine-grained label is mapped exactly once
into the right family, and if an unmapped label fails loudly instead of silently
distorting the metrics.
"""
from __future__ import annotations

import pytest

from src.data import (
    ALL_COLUMNS, CATEGORICAL_COLUMNS, CLASS_ORDER, FEATURE_COLUMNS,
    check_label_coverage, feature_matrix, load_nsl_kdd, map_attack_category,
)


def test_schema_shapes():
    assert len(FEATURE_COLUMNS) == 41
    assert ALL_COLUMNS == FEATURE_COLUMNS + ["label", "difficulty"]
    assert CLASS_ORDER == ["normal", "DoS", "Probe", "R2L", "U2R"]
    assert set(CATEGORICAL_COLUMNS) == {"protocol_type", "service", "flag"}


@pytest.mark.parametrize("label,expected", [
    ("normal", "normal"),
    ("neptune", "DoS"),
    ("smurf", "DoS"),
    ("satan", "Probe"),
    ("nmap", "Probe"),
    ("guess_passwd", "R2L"),
    ("warezmaster", "R2L"),
    ("buffer_overflow", "U2R"),
    ("rootkit", "U2R"),
])
def test_map_attack_category_known(label, expected):
    assert map_attack_category(label) == expected


def test_map_attack_category_unknown_is_sentinel():
    # Unknown labels must not be guessed into a real class.
    assert map_attack_category("a_brand_new_attack") == "UNKNOWN"


def test_check_label_coverage_passes_on_clean(raw_frame):
    df = load_nsl_kdd_from_frame(raw_frame)
    # Should not raise — every synthetic label is in the taxonomy.
    check_label_coverage(df)


def test_check_label_coverage_raises_on_unmapped(raw_frame):
    df = load_nsl_kdd_from_frame(raw_frame)
    df.loc[0, "attack_category"] = "UNKNOWN"
    df.loc[0, "label"] = "mystery"
    with pytest.raises(ValueError, match="Unmapped NSL-KDD attack labels"):
        check_label_coverage(df)


def test_load_nsl_kdd_derives_targets(synthetic_file):
    df = load_nsl_kdd(synthetic_file)
    # Derived columns exist and are internally consistent.
    for col in ("attack_category", "is_attack", "binary_label"):
        assert col in df.columns
    assert set(df["attack_category"]).issubset(set(CLASS_ORDER))
    # is_attack is the complement of the 'normal' label.
    assert (df["is_attack"] == (df["label"] != "normal").astype(int)).all()
    # binary_label mirrors is_attack.
    assert (df.loc[df["is_attack"] == 0, "binary_label"] == "normal").all()
    assert (df.loc[df["is_attack"] == 1, "binary_label"] == "attack").all()


def test_feature_matrix_default_and_subset(synthetic_file):
    df = load_nsl_kdd(synthetic_file)
    full = feature_matrix(df)
    assert list(full.columns) == FEATURE_COLUMNS  # never leaks label/difficulty
    subset = feature_matrix(df, cols=["src_bytes", "dst_bytes"])
    assert list(subset.columns) == ["src_bytes", "dst_bytes"]
    # Returns a copy — mutating it must not touch the source frame.
    subset.iloc[0, 0] = -999
    assert df["src_bytes"].iloc[0] != -999


# --- helper -----------------------------------------------------------------
def load_nsl_kdd_from_frame(raw_frame):
    """Apply the labelling that load_nsl_kdd does, without round-tripping a file."""
    df = raw_frame.copy()
    df["attack_category"] = df["label"].map(map_attack_category)
    return df
