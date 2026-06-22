"""Tests for src/models.py — the model factory wiring.

These guard the contracts the analysis relies on: every supervised model is a
Pipeline that bundles the shared (leakage-safe) preprocessor, cost-sensitive
re-weighting is actually forwarded, and the anomaly detectors are the expected
one-class estimators.
"""
from __future__ import annotations

from sklearn.pipeline import Pipeline

from src.features import build_preprocessor, feature_groups
from src.models import make_anomaly_detectors, make_models


def _preprocessor(raw_frame):
    return build_preprocessor(feature_groups(raw_frame))


def test_make_models_returns_expected_pipelines(raw_frame):
    models = make_models(_preprocessor(raw_frame))
    assert set(models) == {
        "Baseline (majority)", "Logistic Regression",
        "Random Forest", "Gradient Boosting",
    }
    for name, pipe in models.items():
        assert isinstance(pipe, Pipeline), name
        # Each model bundles the preprocessor ahead of the classifier.
        assert list(pipe.named_steps) == ["prep", "clf"], name


def test_make_models_forwards_class_weight(raw_frame):
    models = make_models(_preprocessor(raw_frame), class_weight="balanced")
    rf = models["Random Forest"].named_steps["clf"]
    lr = models["Logistic Regression"].named_steps["clf"]
    assert rf.get_params()["class_weight"] == "balanced"
    assert lr.get_params()["class_weight"] == "balanced"


def test_make_anomaly_detectors(raw_frame):
    det = make_anomaly_detectors(svm_nu=0.2)
    assert set(det) == {"Isolation Forest", "One-Class SVM"}
    assert det["One-Class SVM"].get_params()["nu"] == 0.2
