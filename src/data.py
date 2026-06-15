"""Data loading and labelling for the NSL-KDD intrusion-detection dataset.

The NSL-KDD files have no header row. Each record holds 41 features, a fine
grained attack ``label`` and a ``difficulty`` score (an artefact of how NSL-KDD
was sub-sampled from KDD'99). We attach the canonical column names, derive a
binary target (attack vs. normal) and a 5-class target (normal/DoS/Probe/R2L/U2R)
that we use throughout the analysis.

All functions are pure and side-effect free so the notebook and the report
pipeline can share them without duplication.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Canonical schema
# ---------------------------------------------------------------------------
# The 41 NSL-KDD features in their fixed on-disk order, followed by the two
# trailing columns (fine-grained label and the NSL-KDD "difficulty" score).
FEATURE_COLUMNS: list[str] = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
ALL_COLUMNS: list[str] = FEATURE_COLUMNS + ["label", "difficulty"]

# The three nominal (categorical) features.
CATEGORICAL_COLUMNS: list[str] = ["protocol_type", "service", "flag"]

# Features stored as 0/1 flags — numeric on disk but semantically binary.
BINARY_FLAG_COLUMNS: list[str] = [
    "land", "logged_in", "root_shell", "su_attempted", "is_host_login",
    "is_guest_login",
]

# ---------------------------------------------------------------------------
# Attack family mapping (the standard NSL-KDD 4-attack taxonomy)
# ---------------------------------------------------------------------------
_ATTACK_TO_CATEGORY: dict[str, str] = {
    # Denial of Service
    **{a: "DoS" for a in [
        "back", "land", "neptune", "pod", "smurf", "teardrop", "mailbomb",
        "apache2", "processtable", "udpstorm", "worm"]},
    # Probe / surveillance
    **{a: "Probe" for a in [
        "ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"]},
    # Remote-to-Local
    **{a: "R2L" for a in [
        "ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy",
        "warezclient", "warezmaster", "sendmail", "named", "snmpgetattack",
        "snmpguess", "xlock", "xsnoop", "httptunnel"]},
    # User-to-Root (privilege escalation)
    **{a: "U2R" for a in [
        "buffer_overflow", "loadmodule", "perl", "rootkit", "ps", "sqlattack",
        "xterm"]},
}

CLASS_ORDER: list[str] = ["normal", "DoS", "Probe", "R2L", "U2R"]


def map_attack_category(label: str) -> str:
    """Map a fine-grained NSL-KDD label to {normal, DoS, Probe, R2L, U2R}.

    Attack names are never guessed: anything outside the taxonomy above maps to
    the sentinel ``"UNKNOWN"`` so the caller can detect schema drift (see
    :func:`check_label_coverage`). In practice every label in KDDTrain+/KDDTest+
    is covered, which the loader asserts.
    """
    if label == "normal":
        return "normal"
    return _ATTACK_TO_CATEGORY.get(label, "UNKNOWN")


def check_label_coverage(df: pd.DataFrame) -> None:
    """Raise if any row's attack label fell through to ``"UNKNOWN"``.

    Turns a silent labelling gap (which would distort per-class metrics) into a
    loud, explicit failure — the integrity guard the docstring of
    :func:`map_attack_category` promises.
    """
    unknown = sorted(df.loc[df["attack_category"] == "UNKNOWN", "label"].unique())
    if unknown:
        raise ValueError(
            f"Unmapped NSL-KDD attack labels (schema drift): {unknown}. "
            "Extend _ATTACK_TO_CATEGORY in src/data.py.")


def load_nsl_kdd(path: str | Path) -> pd.DataFrame:
    """Load one NSL-KDD ``.txt`` file into a typed, labelled DataFrame.

    Adds three derived columns:
      * ``attack_category`` — the 5-class target (see ``CLASS_ORDER``).
      * ``is_attack``       — binary target (1 = attack, 0 = normal).
      * ``binary_label``    — human-readable {"attack", "normal"}.
    """
    df = pd.read_csv(path, header=None, names=ALL_COLUMNS)
    df["attack_category"] = df["label"].map(map_attack_category)
    check_label_coverage(df)  # fail loudly on any unmapped attack name
    df["is_attack"] = (df["label"] != "normal").astype(int)
    df["binary_label"] = df["is_attack"].map({0: "normal", 1: "attack"})
    return df


def load_train_test(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the official KDDTrain+ and KDDTest+ partitions as (train, test)."""
    raw_dir = Path(raw_dir)
    train = load_nsl_kdd(raw_dir / "KDDTrain+.txt")
    test = load_nsl_kdd(raw_dir / "KDDTest+.txt")
    return train, test


def feature_matrix(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Return the model-input columns (default: the 41 canonical features).

    ``cols`` lets callers request a custom feature set — e.g. the redundancy
    ablation (a subset) or the engineered-feature experiment (a superset) — while
    keeping label/difficulty/target columns out by construction.
    """
    return df[FEATURE_COLUMNS if cols is None else cols].copy()
