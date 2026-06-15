"""Evaluation metrics and error analysis.

We deliberately report a *suite* — Accuracy alongside the metrics that survive
class imbalance (Balanced Accuracy, MCC, macro-F1, F-beta, PR-AUC) — because the
central claim under test is that headline Accuracy is misleading on this problem.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score, fbeta_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

# In intrusion detection a missed attack (False Negative) is usually costlier
# than a false alarm, so we weight recall above precision with beta = 2.
FBETA = 2.0


def binary_metrics(y_true, y_pred, y_score=None) -> dict:
    """Full binary-classification metric set (attack = positive class = 1)."""
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        f"f{FBETA:g}": fbeta_score(y_true, y_pred, beta=FBETA, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if y_score is not None:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
        out["pr_auc"] = average_precision_score(y_true, y_score)
    return out


def multiclass_metrics(y_true, y_pred) -> dict:
    """Headline + imbalance-robust metrics for the 5-class target."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def per_class_recall(y_true, y_pred, labels: list[str]) -> pd.Series:
    """Recall per class — surfaces the R2L/U2R collapse that Accuracy hides."""
    rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return pd.Series(rec, index=labels, name="recall")


def per_class_detection_rate(y_true_multi, y_pred_binary, labels: list[str]
                             ) -> pd.Series:
    """Fraction of each true class flagged as *attack* (predicted positive).

    For attack families this is recall; for ``normal`` it is the false-positive
    rate. Lets us score one-class anomaly detectors (which only output
    attack/normal) on the same per-family axis as the supervised models.
    """
    y_true_multi = np.asarray(y_true_multi)
    y_pred_binary = np.asarray(y_pred_binary)
    rates = {}
    for cls in labels:
        mask = y_true_multi == cls
        rates[cls] = float(y_pred_binary[mask].mean()) if mask.any() else float("nan")
    return pd.Series(rates, name="detection_rate")


def plot_detection_comparison(table: pd.DataFrame, title: str, name: str
                              ) -> plt.Figure:
    """Grouped bar chart of per-class detection rate across models/paradigms."""
    classes = list(table.index)
    methods = list(table.columns)
    x = np.arange(len(classes))
    width = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, m in enumerate(methods):
        ax.bar(x + i * width, table[m].to_numpy(), width, label=m)
    ax.set_xticks(x + width * (len(methods) - 1) / 2, classes)
    ax.set_ylabel("Detection rate (recall; FPR for 'normal')")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=130, bbox_inches="tight")
    return fig


def class_report_df(y_true, y_pred, labels: list[str]) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True,
                                zero_division=0)
    return pd.DataFrame(rep).T


def plot_confusion(y_true, y_pred, labels: list[str], title: str, name: str
                   ) -> plt.Figure:
    """Row-normalised confusion matrix (recall per true class)."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=130, bbox_inches="tight")
    return fig


def metrics_table(results: dict[str, dict]) -> pd.DataFrame:
    """Stack per-model metric dicts into a comparison table (rounded)."""
    return pd.DataFrame(results).T.round(4)


def plot_pr_and_roc(curves: dict[str, tuple], name: str = "pr_roc.png") -> plt.Figure:
    """Overlay PR and ROC curves. ``curves[name] = (y_true, y_score)``."""
    from sklearn.metrics import precision_recall_curve, roc_curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for label, (y_true, y_score) in curves.items():
        p, r, _ = precision_recall_curve(y_true, y_score)
        axes[0].plot(r, p, label=label)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        axes[1].plot(fpr, tpr, label=label)
    base_rate = np.mean(list(curves.values())[0][0])
    axes[0].axhline(base_rate, ls="--", c="grey", label=f"prevalence={base_rate:.2f}")
    axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision-Recall (KDDTest+)"); axes[0].legend(fontsize=8)
    axes[1].plot([0, 1], [0, 1], ls="--", c="grey")
    axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC (KDDTest+)"); axes[1].legend(fontsize=8)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=130, bbox_inches="tight")
    return fig


def threshold_sweep(y_true, y_score) -> pd.DataFrame:
    """FP/FN trade-off across decision thresholds (operating-point analysis)."""
    rows = []
    for thr in np.linspace(0.1, 0.9, 9):
        y_pred = (y_score >= thr).astype(int)
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        rows.append({
            "threshold": round(thr, 2),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            f"f{FBETA:g}": fbeta_score(y_true, y_pred, beta=FBETA, zero_division=0),
            "false_alarms_FP": fp, "missed_attacks_FN": fn,
        })
    return pd.DataFrame(rows)
