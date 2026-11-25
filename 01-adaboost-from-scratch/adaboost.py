"""AdaBoost (discrete, two-class) written with numpy only.

Labels are expected as -1 / +1. The weak learner is a decision stump:
one feature, one threshold, one direction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Stump:
    feature: int
    threshold: float
    polarity: int  # +1 means "predict +1 when x >= threshold"
    alpha: float = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        column = X[:, self.feature]
        preds = np.where(column >= self.threshold, 1, -1)
        return preds * self.polarity


def fit_stump(X: np.ndarray, y: np.ndarray, w: np.ndarray, n_thresholds: int = 20) -> tuple[Stump, float]:
    """Find the stump with the lowest weighted error.

    Trying every unique value is slow on bigger data, so I only try a set of
    quantiles per feature. It barely changes accuracy and is much faster.
    """
    best: Stump | None = None
    best_err = np.inf
    quantiles = np.linspace(0, 1, n_thresholds + 2)[1:-1]

    for j in range(X.shape[1]):
        thresholds = np.unique(np.quantile(X[:, j], quantiles))
        for t in thresholds:
            base = np.where(X[:, j] >= t, 1, -1)
            for polarity in (1, -1):
                preds = base * polarity
                err = w[preds != y].sum()
                if err < best_err:
                    best_err = err
                    best = Stump(feature=j, threshold=float(t), polarity=polarity)

    assert best is not None
    return best, float(best_err)


class AdaBoost:
    def __init__(self, n_estimators: int = 50, n_thresholds: int = 20) -> None:
        self.n_estimators = n_estimators
        self.n_thresholds = n_thresholds
        self.stumps: list[Stump] = []
        self.train_errors: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaBoost":
        y = np.asarray(y)
        if set(np.unique(y)) - {-1, 1}:
            raise ValueError("labels must be -1 or +1")

        n = len(y)
        w = np.full(n, 1.0 / n)
        self.stumps = []
        self.train_errors = []

        for _ in range(self.n_estimators):
            stump, err = fit_stump(X, y, w, self.n_thresholds)
            err = np.clip(err, 1e-10, 1 - 1e-10)  # avoid log(0) when a stump is perfect
            stump.alpha = 0.5 * np.log((1 - err) / err)

            preds = stump.predict(X)
            # misclassified points get heavier, correct ones lighter
            w = w * np.exp(-stump.alpha * y * preds)
            w = w / w.sum()

            self.stumps.append(stump)
            self.train_errors.append(float(np.mean(self.predict(X) != y)))

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return sum(s.alpha * s.predict(X) for s in self.stumps)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.sign(self.decision_function(X)).astype(int)
