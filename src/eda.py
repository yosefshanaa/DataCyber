"""Exploratory-data-analysis helpers for NSL-KDD.

Every function is small, returns a value (DataFrame / dict / Figure) and never
mutates its input. Plotting helpers save a figure to ``figures/`` and also
return the matplotlib Figure so the notebook can display it inline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"


def _save(fig: plt.Figure, name: str) -> plt.Figure:
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=130, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Structural inspection
# ---------------------------------------------------------------------------
def basic_overview(df: pd.DataFrame) -> dict:
    """Shape, dtype counts and missing-value totals in one dict."""
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "dtype_counts": {str(k): int(v) for k, v in df.dtypes.value_counts().items()},
        "total_missing": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def constant_features(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Return features that take exactly one value (zero information / redundant)."""
    return [c for c in cols if df[c].nunique(dropna=False) <= 1]


def duplicate_feature_pairs(df: pd.DataFrame, cols: list[str]) -> list[tuple[str, str]]:
    """Find pairs of columns that are identical value-for-value (exact duplicates)."""
    numeric = df[cols].select_dtypes("number")
    pairs: list[tuple[str, str]] = []
    seen = numeric.T.duplicated(keep=False)
    dup_cols = seen[seen].index.tolist()
    for i, a in enumerate(dup_cols):
        for b in dup_cols[i + 1:]:
            if numeric[a].equals(numeric[b]):
                pairs.append((a, b))
    return pairs


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing counts and percentages, sorted high to low."""
    miss = df.isna().sum()
    out = pd.DataFrame({"missing": miss, "pct": (miss / len(df) * 100).round(3)})
    return out[out["missing"] > 0].sort_values("missing", ascending=False)


# ---------------------------------------------------------------------------
# Robust outlier analysis (Z-score vs. IQR vs. Modified-Z / MAD)
# ---------------------------------------------------------------------------
def outlier_summary(series: pd.Series) -> dict:
    """Compare classic and robust outlier counts for one numeric feature.

    * Z-score: |x - mean| / std > 3   (sensitive to heavy tails)
    * IQR    : outside [Q1-1.5IQR, Q3+1.5IQR]
    * MAD    : modified Z = 0.6745 * (x - median) / MAD, flagged if |.| > 3.5
    """
    x = series.astype(float).to_numpy()
    mean, std = x.mean(), x.std()
    z = np.zeros_like(x) if std == 0 else np.abs((x - mean) / std)

    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    iqr_mask = (x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)

    med = np.median(x)
    mad = np.median(np.abs(x - med))
    mod_z = np.zeros_like(x) if mad == 0 else 0.6745 * np.abs(x - med) / mad

    return {
        "feature": series.name,
        "skew": float(pd.Series(x).skew()),
        "z_gt3": int((z > 3).sum()),
        "iqr_out": int(iqr_mask.sum()),
        "mad_gt3p5": int((mod_z > 3.5).sum()),
        "max": float(x.max()),
        "median": float(med),
        "mean": float(mean),
    }


# ---------------------------------------------------------------------------
# Class balance / prevalence
# ---------------------------------------------------------------------------
def class_prevalence(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Counts and percentage prevalence for a categorical target."""
    counts = df[target].value_counts()
    return pd.DataFrame({
        "count": counts,
        "prevalence_pct": (counts / len(df) * 100).round(3),
    })


def plot_class_balance(train: pd.DataFrame, test: pd.DataFrame, target: str,
                       order: list[str], name: str = "class_balance.png") -> plt.Figure:
    """Side-by-side train/test prevalence — exposes the train↔test class shift."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, df, title in zip(axes, [train, test], ["KDDTrain+", "KDDTest+"]):
        prev = df[target].value_counts(normalize=True).reindex(order).fillna(0) * 100
        ax.bar(prev.index, prev.values, color="#4C72B0")
        ax.set_title(f"{title} — class prevalence (%)")
        ax.set_ylabel("% of records")
        for i, v in enumerate(prev.values):
            ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)
    fig.tight_layout()
    return _save(fig, name)


# ---------------------------------------------------------------------------
# Correlation (Spearman chosen for skewed count features — justified in report)
# ---------------------------------------------------------------------------
def correlation_matrix(df: pd.DataFrame, cols: list[str], method: str = "spearman"
                       ) -> pd.DataFrame:
    return df[cols].corr(method=method)


def high_correlation_pairs(corr: pd.DataFrame, threshold: float = 0.9
                           ) -> pd.DataFrame:
    """Upper-triangle pairs with |corr| >= threshold — redundancy candidates."""
    keep = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = (keep.stack()
             .rename("corr")
             .reset_index()
             .rename(columns={"level_0": "feat_a", "level_1": "feat_b"}))
    pairs = pairs[pairs["corr"].abs() >= threshold]
    return pairs.sort_values("corr", key=lambda s: s.abs(), ascending=False)


def plot_correlation_heatmap(corr: pd.DataFrame, name: str = "corr_heatmap.png"
                             ) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(corr.columns, fontsize=6)
    ax.set_title("Spearman correlation — numeric features")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _save(fig, name)


def plot_log_transform_effect(train: pd.DataFrame, col: str = "src_bytes",
                              name: str = "log_transform.png") -> plt.Figure:
    """Show a heavy-tailed feature before/after log1p — motivates the transform."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    raw = train[col].clip(lower=0)
    axes[0].hist(raw, bins=60, color="#C44E52")
    axes[0].set_title(f"{col} (raw) — skew={raw.skew():.1f}")
    axes[0].set_yscale("log")
    logged = np.log1p(raw)
    axes[1].hist(logged, bins=60, color="#55A868")
    axes[1].set_title(f"log1p({col}) — skew={logged.skew():.1f}")
    fig.tight_layout()
    return _save(fig, name)
