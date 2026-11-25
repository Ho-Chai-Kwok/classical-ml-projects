"""Compare my AdaBoost against scikit-learn's, plus gradient boosting and XGBoost.

Run: python compare.py
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from adaboost import AdaBoost


def main() -> None:
    X, y01 = load_breast_cancer(return_X_y=True)
    y = np.where(y01 == 1, 1, -1)
    X_tr, X_te, y_tr, y_te, y01_tr, y01_te = train_test_split(
        X, y, y01, test_size=0.25, random_state=42, stratify=y
    )

    rows = []

    start = time.perf_counter()
    mine = AdaBoost(n_estimators=100).fit(X_tr, y_tr)
    rows.append(("my AdaBoost (100 stumps)", accuracy_score(y_te, mine.predict(X_te)), time.perf_counter() - start))

    models = {
        "sklearn AdaBoost (100 stumps)": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1), n_estimators=100, random_state=0
        ),
        "sklearn GradientBoosting": GradientBoostingClassifier(random_state=0),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, eval_metric="logloss"),
    }
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X_tr, y01_tr)
        rows.append((name, accuracy_score(y01_te, model.predict(X_te)), time.perf_counter() - start))

    print(f"{'model':32s} {'test acc':>9s} {'fit time':>9s}")
    for name, acc, secs in rows:
        print(f"{name:32s} {acc:9.3f} {secs:8.2f}s")

    print("\ntraining error of my AdaBoost by round:")
    for k in (1, 5, 10, 25, 50, 100):
        print(f"  after {k:3d} stumps: {mine.train_errors[k - 1]:.3f}")


if __name__ == "__main__":
    main()
