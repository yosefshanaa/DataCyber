"""Tests for src/evaluate.py — metric correctness and operating-point logic.

The argument of the project is metric-driven, so the metric helpers must compute
exactly what they claim, including the per-class detection rate that lets a
one-class detector be scored on the same axis as the supervised models.
"""
from __future__ import annotations

import numpy as np

from src.evaluate import (
    binary_metrics, multiclass_metrics, per_class_detection_rate,
    per_class_recall, threshold_sweep,
)


def test_binary_metrics_perfect_prediction():
    y = [0, 1, 0, 1, 1]
    m = binary_metrics(y, y)
    assert m["accuracy"] == 1.0
    assert m["recall"] == 1.0
    assert m["mcc"] == 1.0
    assert m["f2"] == 1.0


def test_binary_metrics_known_values():
    # tp=1, fn=1, fp=0, tn=2  (attack = positive class = 1)
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 0, 0]
    m = binary_metrics(y_true, y_pred)
    assert m["precision"] == 1.0          # 1 / (1 + 0)
    assert m["recall"] == 0.5             # 1 / (1 + 1)
    assert m["accuracy"] == 0.75          # 3 / 4


def test_binary_metrics_includes_auc_when_scores_given():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.4, 0.35, 0.8]
    m = binary_metrics(y_true, (np.array(y_score) >= 0.5).astype(int), y_score)
    assert "roc_auc" in m and "pr_auc" in m
    assert 0.0 <= m["roc_auc"] <= 1.0


def test_per_class_detection_rate():
    # normal -> FPR; attack family -> recall.
    y_true_multi = ["normal", "DoS", "DoS", "R2L"]
    y_pred_binary = [1, 1, 0, 0]
    rates = per_class_detection_rate(y_true_multi, y_pred_binary,
                                     ["normal", "DoS", "R2L"])
    assert rates["normal"] == 1.0   # 1/1 flagged -> false positive rate
    assert rates["DoS"] == 0.5      # 1 of 2 detected
    assert rates["R2L"] == 0.0      # missed entirely


def test_per_class_detection_rate_absent_class_is_nan():
    rates = per_class_detection_rate(["normal"], [0], ["normal", "U2R"])
    assert np.isnan(rates["U2R"])


def test_per_class_recall_orders_by_labels():
    y_true = ["DoS", "DoS", "Probe", "normal"]
    y_pred = ["DoS", "normal", "Probe", "normal"]
    rec = per_class_recall(y_true, y_pred, ["normal", "DoS", "Probe"])
    assert list(rec.index) == ["normal", "DoS", "Probe"]
    assert rec["DoS"] == 0.5
    assert rec["Probe"] == 1.0


def test_multiclass_metrics_keys():
    y = ["normal", "DoS", "Probe", "R2L"]
    m = multiclass_metrics(y, y)
    assert {"accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "mcc"} <= m.keys()
    assert m["accuracy"] == 1.0


def test_threshold_sweep_is_monotone():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_score = rng.random(200)
    sweep = threshold_sweep(y_true, y_score)
    # As the threshold rises, fewer positives are predicted, so false alarms
    # cannot increase and missed attacks cannot decrease.
    assert sweep["false_alarms_FP"].is_monotonic_decreasing
    assert sweep["missed_attacks_FN"].is_monotonic_increasing
