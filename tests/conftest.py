"""Shared fixtures for the test suite.

Most tests run on a small, deterministic *synthetic* NSL-KDD frame so they are
fast and do not depend on the bundled data being present. The few integration
tests that need the real partitions use ``real_raw_dir`` and skip cleanly when
the data is absent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import ALL_COLUMNS, FEATURE_COLUMNS

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"

# One representative label from each of the five NSL-KDD classes.
_LABEL_CYCLE = ["normal", "neptune", "satan", "guess_passwd", "buffer_overflow"]


def _make_raw_frame(n: int = 50, seed: int = 0) -> pd.DataFrame:
    """Build a deterministic raw NSL-KDD frame (41 features + label + difficulty).

    Construction is fully controlled (not random) so column-level invariants are
    stable: ``num_outbound_cmds`` is held constant (as in the real data) so the
    zero-variance drop is exercised, the 3 nominal columns cycle through valid
    values, and all count/byte features stay non-negative (log1p-safe).
    """
    rng = np.arange(n)
    data: dict[str, object] = {}
    for col in FEATURE_COLUMNS:
        # Vary every numeric feature but keep it non-negative.
        data[col] = (rng % 7).astype(float)
    # Categorical columns cycle through valid NSL-KDD vocabulary.
    data["protocol_type"] = np.take(["tcp", "udp", "icmp"], rng % 3)
    data["service"] = np.take(["http", "private", "domain_u", "smtp"], rng % 4)
    data["flag"] = np.take(["SF", "S0", "REJ"], rng % 3)
    # A genuinely constant feature (matches real NSL-KDD) -> must be dropped.
    data["num_outbound_cmds"] = np.zeros(n)
    # Binary flags alternate so they are not accidentally constant.
    data["logged_in"] = (rng % 2)
    data["label"] = np.take(_LABEL_CYCLE, rng % len(_LABEL_CYCLE))
    data["difficulty"] = (rng % 21).astype(int)
    return pd.DataFrame(data, columns=ALL_COLUMNS)


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """A small raw NSL-KDD frame (pre-labelling)."""
    return _make_raw_frame()


@pytest.fixture
def synthetic_file(tmp_path, raw_frame) -> Path:
    """Write the raw frame to a headerless CSV, mimicking the on-disk format."""
    path = tmp_path / "synthetic_nsl.txt"
    raw_frame.to_csv(path, header=False, index=False)
    return path


@pytest.fixture
def real_raw_dir() -> Path:
    """Path to the bundled NSL-KDD data; skip the test if it is not present."""
    if not (RAW_DIR / "KDDTrain+.txt").exists():
        pytest.skip("bundled NSL-KDD data not found (run python data/get_data.py)")
    return RAW_DIR


@pytest.fixture
def real_unsw_dir() -> Path:
    """Path to the bundled UNSW-NB15 data; skip the test if it is not present."""
    if not (RAW_DIR / "UNSW_NB15_training-set.csv").exists():
        pytest.skip("bundled UNSW-NB15 data not found (run python data/get_data.py)")
    return RAW_DIR
