"""Model factory. Each model is a scikit-learn Pipeline that bundles the shared
leakage-safe preprocessor with a classifier, so ``fit`` learns all transform
parameters from the training fold only. A single ``RANDOM_STATE`` is threaded
through every estimator for reproducibility.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

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
                               random_state=RANDOM_STATE, n_jobs=-1)),
        "Random Forest": pipe(
            RandomForestClassifier(n_estimators=200, class_weight=class_weight,
                                   random_state=RANDOM_STATE, n_jobs=-1)),
        "Gradient Boosting": pipe(
            HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.1,
                class_weight=class_weight, random_state=RANDOM_STATE)),
    }
