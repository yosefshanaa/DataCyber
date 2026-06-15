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
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .data import CATEGORICAL_COLUMNS, FEATURE_COLUMNS

# Non-negative count/byte features whose distributions are heavy-tailed and
# benefit from a log transform. Rate features (already in [0, 1]) are excluded.
_LOG_CANDIDATES: list[str] = [
    "duration", "src_bytes", "dst_bytes", "hot", "num_failed_logins",
    "num_compromised", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "count", "srv_count", "dst_host_count",
    "dst_host_srv_count", "wrong_fragment", "urgent", "num_outbound_cmds",
]


def feature_groups(train: pd.DataFrame) -> dict[str, list[str]]:
    """Partition the 41 features into transform groups using *train* only.

    Returns dict with: ``dropped`` (constant), ``categorical``, ``log_numeric``
    (skewed counts to log+scale) and ``plain_numeric`` (everything else to scale).
    """
    dropped = [c for c in FEATURE_COLUMNS if train[c].nunique(dropna=False) <= 1]
    categorical = [c for c in CATEGORICAL_COLUMNS if c not in dropped]
    numeric = [c for c in FEATURE_COLUMNS
               if c not in categorical and c not in dropped]
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
