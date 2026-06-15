"""Model factory. Each model is a scikit-learn Pipeline that bundles the shared
leakage-safe preprocessor with a classifier, so ``fit`` learns all transform
parameters from the training fold only. A single ``RANDOM_STATE`` is threaded
through every estimator for reproducibility.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM

RANDOM_STATE = 42


def make_models(preprocessor: ColumnTransformer, class_weight: str | None = None
                ) -> dict[str, Pipeline]:
    """Return the model zoo as name -> fitted-able Pipeline.

    ``class_weight`` is forwarded to the cost-sensitive learners so we can show
    the effect of re-weighting the rare attack classes (R2L/U2R).
    """
    def pipe(clf):
        return Pipeline([("prep", preprocessor), ("clf", clf)])

    return {
        # Trivial reference point: always predict the majority class. Its score
        # is the yardstick that exposes Accuracy as a misleading metric.
        "Baseline (majority)": pipe(
            DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)),
        "Logistic Regression": pipe(
            LogisticRegression(max_iter=2000, class_weight=class_weight,
                               random_state=RANDOM_STATE)),
        "Random Forest": pipe(
            RandomForestClassifier(n_estimators=200, class_weight=class_weight,
                                   random_state=RANDOM_STATE, n_jobs=-1)),
        "Gradient Boosting": pipe(
            HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.1,
                class_weight=class_weight, random_state=RANDOM_STATE)),
    }


def make_anomaly_detectors(svm_nu: float = 0.1) -> dict[str, object]:
    """Semi-supervised (one-class) anomaly detectors for the rare-attack thesis.

    These are trained on *normal traffic only* and flag deviations as attacks —
    the paradigm our analysis recommends for R2L/U2R. They consume an
    already-preprocessed feature matrix (scaling matters for One-Class SVM), so
    the caller shares the leakage-safe preprocessor. ``IsolationForest`` scales
    to the full normal set; ``OneClassSVM`` is fit on a capped subsample (RBF SVM
    is O(n^2)). Convention: ``predict`` returns -1 for anomalies, +1 for inliers.
    """
    return {
        "Isolation Forest": IsolationForest(
            n_estimators=300, contamination="auto",
            random_state=RANDOM_STATE, n_jobs=-1),
        "One-Class SVM": OneClassSVM(kernel="rbf", gamma="scale", nu=svm_nu),
    }
