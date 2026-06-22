"""Data loading and preprocessing for the **UNSW-NB15** intrusion-detection
dataset — the modern (2015) external-validation counterpart to NSL-KDD.

UNSW-NB15 (Moustafa & Slay, 2015, UNSW Canberra) was built precisely because
KDD'99 / NSL-KDD are old and unrepresentative of modern traffic. We use the
authors' official partitioned CSVs (``UNSW_NB15_training-set.csv`` /
``UNSW_NB15_testing-set.csv``): 42 flow features, a 10-class ``attack_cat`` and a
binary ``label``.

The design mirrors :mod:`src.data` so the rest of the pipeline (the leakage-safe
preprocessor in :mod:`src.features`, the model zoo in :mod:`src.models` and the
metrics in :mod:`src.evaluate`) can be reused unchanged.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
# The 42 model-input features, in on-disk order. The CSV also carries a leading
# ``id`` index column and the two targets (``attack_cat``, ``label``) which are
# NOT features and are excluded here by construction.
FEATURE_COLUMNS: list[str] = [
    "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes",
    "rate", "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt",
    "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt",
    "synack", "ackdat", "smean", "dmean", "trans_depth", "response_body_len",
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd",
    "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
]

# The three nominal (categorical) features. ``proto`` is high-cardinality (~130
# values), which is exactly why OneHotEncoder(handle_unknown='ignore') matters.
CATEGORICAL_COLUMNS: list[str] = ["proto", "service", "state"]

# Class order: normal first, then attack families by descending frequency. The
# last four are the *rare* families — the UNSW-NB15 analogue of NSL-KDD's R2L/U2R.
CLASS_ORDER: list[str] = [
    "normal", "Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance",
    "Analysis", "Backdoor", "Shellcode", "Worms",
]
RARE_ATTACKS: list[str] = ["Analysis", "Backdoor", "Shellcode", "Worms"]


def load_unsw(path: str | Path) -> pd.DataFrame:
    """Load one UNSW-NB15 partition CSV into a typed, labelled DataFrame.

    The files already have a header row. We normalise the normal-class name to
    ``"normal"`` (UNSW writes ``"Normal"``) so the 5-class helpers and the
    ``normal``-row-as-false-positive-rate convention from the NSL-KDD analysis
    carry over verbatim. Adds:
      * ``attack_category`` — the 10-class target (see ``CLASS_ORDER``).
      * ``is_attack``       — binary target (1 = attack, 0 = normal), from ``label``.
      * ``binary_label``    — human-readable {"attack", "normal"}.
    """
    df = pd.read_csv(path)
    df["attack_category"] = df["attack_cat"].replace("Normal", "normal")
    df["is_attack"] = df["label"].astype(int)
    df["binary_label"] = df["is_attack"].map({0: "normal", 1: "attack"})
    return df


def load_train_test(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the official UNSW-NB15 training/testing partitions as (train, test)."""
    raw_dir = Path(raw_dir)
    train = load_unsw(raw_dir / "UNSW_NB15_training-set.csv")
    test = load_unsw(raw_dir / "UNSW_NB15_testing-set.csv")
    return train, test


def feature_matrix(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Return the model-input columns (default: the 42 canonical features)."""
    return df[FEATURE_COLUMNS if cols is None else cols].copy()


def feature_groups(train: pd.DataFrame,
                   feature_cols: list[str] | None = None) -> dict[str, list[str]]:
    """Partition UNSW-NB15 features into transform groups using *train* only.

    Same contract as :func:`src.features.feature_groups` but with UNSW's schema:
    nominal columns go to ``categorical``; zero-variance columns are dropped; and
    a numeric column is log-scaled when it is non-negative on train and
    heavy-tailed (``max > 100``) — i.e. bytes/loads/jitter/large counts — while
    bounded rates, ttls and 0/1 flags are merely standardised. The returned dict
    plugs straight into :func:`src.features.build_preprocessor`.
    """
    cols = list(FEATURE_COLUMNS if feature_cols is None else feature_cols)
    dropped = [c for c in cols if train[c].nunique(dropna=False) <= 1]
    categorical = [c for c in CATEGORICAL_COLUMNS if c in cols and c not in dropped]
    numeric = [c for c in cols if c not in categorical and c not in dropped]
    log_numeric = [
        c for c in numeric
        if (train[c].min() >= 0) and (train[c].max() > 100)
    ]
    plain_numeric = [c for c in numeric if c not in log_numeric]
    return {
        "dropped": dropped,
        "categorical": categorical,
        "log_numeric": log_numeric,
        "plain_numeric": plain_numeric,
    }
