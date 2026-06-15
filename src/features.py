"""Feature engineering for NSL-KDD — built as scikit-learn transformers so that
*every* fit happens on training data only (no leakage into the test set).

Design choices (justified in the report):
  * One-hot encode the 3 nominal features with ``handle_unknown='ignore'`` —
    KDDTest+ contains ``service`` values unseen in KDDTrain+; ignoring them is the
    honest behaviour (a label/ordinal encoder would silently invent an ordering
    and break on unseen categories).
  * ``log1p`` the heavy-tailed, non-negative count/byte features to tame extreme
    right-skew (src_bytes etc. span 0 .. 1.3e9) before standardisation.
  * Drop zero-variance (constant) features such as ``num_outbound_cmds`` — they
    carry no information and only dilute feature importance (redundancy).
  * StandardScaler so scale-sensitive models (Logistic Regression, SVM, KNN) are
    on equal footing; tree ensembles are invariant to it, so one shared
    preprocessor is safe and avoids duplicated pipelines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler,
)

from .data import CATEGORICAL_COLUMNS, FEATURE_COLUMNS

# Non-negative count/byte features whose distributions are heavy-tailed and
# benefit from a log transform. Rate features (already in [0, 1]) are excluded.
_LOG_CANDIDATES: list[str] = [
    "duration", "src_bytes", "dst_bytes", "hot", "num_failed_logins",
    "num_compromised", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "count", "srv_count", "dst_host_count",
    "dst_host_srv_count", "wrong_fragment", "urgent", "num_outbound_cmds",
]


def feature_groups(train: pd.DataFrame,
                   feature_cols: list[str] | None = None) -> dict[str, list[str]]:
    """Partition the input features into transform groups using *train* only.

    ``feature_cols`` defaults to the 41 canonical features but can be a subset
    (redundancy ablation) or a superset (engineered features); any column in
    ``_LOG_CANDIDATES`` is log-scaled, the rest are scaled, and zero-variance
    columns are dropped. Returns a dict with ``dropped`` (constant),
    ``categorical``, ``log_numeric`` and ``plain_numeric``.
    """
    cols = list(FEATURE_COLUMNS if feature_cols is None else feature_cols)
    dropped = [c for c in cols if train[c].nunique(dropna=False) <= 1]
    categorical = [c for c in CATEGORICAL_COLUMNS if c in cols and c not in dropped]
    numeric = [c for c in cols if c not in categorical and c not in dropped]
    log_numeric = [c for c in numeric if c in _LOG_CANDIDATES]
    plain_numeric = [c for c in numeric if c not in log_numeric]
    return {
        "dropped": dropped,
        "categorical": categorical,
        "log_numeric": log_numeric,
        "plain_numeric": plain_numeric,
    }


def build_preprocessor(groups: dict[str, list[str]]) -> ColumnTransformer:
    """Assemble the leakage-safe ColumnTransformer from ``feature_groups`` output."""
    log_pipe = Pipeline([
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer(
        transformers=[
            ("log_num", log_pipe, groups["log_numeric"]),
            ("num", StandardScaler(), groups["plain_numeric"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             groups["categorical"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ---------------------------------------------------------------------------
# Literal "tutorial-style" preprocessing (for the reproduction control)
# ---------------------------------------------------------------------------
# The reviewed repositories ordinal/label-encode the nominal columns, feed raw
# (un-scaled) values to tree/NN models, and — crucially — keep the NSL-KDD
# ``difficulty`` column as a feature. We reconstruct exactly that pipeline so the
# A/B experiment compares against what the tutorials actually do, leak included.
TUTORIAL_COLUMNS: list[str] = FEATURE_COLUMNS + ["difficulty"]


def build_tutorial_preprocessor() -> ColumnTransformer:
    """Reconstruct the tutorials' preprocessing: LabelEncoder-style ordinal coding
    of the 3 nominal columns, everything else (incl. the leaky ``difficulty``)
    passed through raw. ``unknown_value=-1`` keeps it runnable on KDDTest+."""
    return ColumnTransformer(
        transformers=[
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=-1), CATEGORICAL_COLUMNS),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def redundant_drop_set(redundant_pairs: list[dict]) -> list[str]:
    """Greedily pick one feature from each highly-correlated pair to drop.

    Keeps the first feature of a pair and drops the second unless it is already
    being kept as another pair's survivor — so we never drop both members of a
    pair and never orphan a feature. Used by the redundancy ablation.
    """
    keep: set[str] = set()
    drop: set[str] = set()
    for pair in redundant_pairs:
        a, b = pair["feat_a"], pair["feat_b"]
        if a in drop or b in drop:
            continue
        keep.add(a)
        drop.add(b)
    return sorted(drop)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Optional domain features proposed in the report (not used by default).

    * ``bytes_ratio``       — asymmetry between sent/received bytes (exfil signal).
    * ``total_bytes``       — overall volume.
    * ``is_well_known_port``-ish proxy via service grouping is left to encoding.
    These illustrate *feature creation*; we keep them separate so the baseline
    comparison stays clean.
    """
    out = df.copy()
    out["total_bytes"] = out["src_bytes"] + out["dst_bytes"]
    out["bytes_ratio"] = (out["src_bytes"] + 1) / (out["dst_bytes"] + 1)
    out["error_rate_mean"] = out[
        ["serror_rate", "rerror_rate", "srv_serror_rate", "srv_rerror_rate"]
    ].mean(axis=1)
    return out


# Names of the columns created by :func:`add_engineered_features` (so the
# engineered-feature experiment can request exactly this superset).
ENGINEERED_COLUMNS: list[str] = ["total_bytes", "bytes_ratio", "error_rate_mean"]
