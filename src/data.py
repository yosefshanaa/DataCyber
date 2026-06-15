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

    Unknown attack names default to ``R2L``/``U2R`` are *not* guessed: anything
    unmapped is returned verbatim so the caller can detect schema drift. In
    practice every label in KDDTrain+/KDDTest+ is covered by the mapping above.
    """
    if label == "normal":
        return "normal"
    return _ATTACK_TO_CATEGORY.get(label, "UNKNOWN")


def load_nsl_kdd(path: str | Path) -> pd.DataFrame:
    """Load one NSL-KDD ``.txt`` file into a typed, labelled DataFrame.

    Adds three derived columns:
      * ``attack_category`` — the 5-class target (see ``CLASS_ORDER``).
      * ``is_attack``       — binary target (1 = attack, 0 = normal).
      * ``binary_label``    — human-readable {"attack", "normal"}.
    """
    df = pd.read_csv(path, header=None, names=ALL_COLUMNS)
    df["attack_category"] = df["label"].map(map_attack_category)
    df["is_attack"] = (df["label"] != "normal").astype(int)
    df["binary_label"] = df["is_attack"].map({0: "normal", 1: "attack"})
    return df


def load_train_test(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the official KDDTrain+ and KDDTest+ partitions as (train, test)."""
    raw_dir = Path(raw_dir)
    train = load_nsl_kdd(raw_dir / "KDDTrain+.txt")
    test = load_nsl_kdd(raw_dir / "KDDTest+.txt")
    return train, test


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return just the 41 model input features (drop labels/difficulty/targets)."""
    return df[FEATURE_COLUMNS].copy()
