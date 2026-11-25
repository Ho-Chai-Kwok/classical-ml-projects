"""A small experiment that shows how data leakage fakes a good score.

The labels here are pure coin flips, so the honest accuracy is about 50%.
If you pick "useful" features using the whole dataset before cross-validation,
the model looks far better than chance. Putting the selection inside a
Pipeline fixes it, because each fold only sees its own training rows.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def run(seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(200, 5_000))
    y = rng.integers(0, 2, size=200)

    # wrong: feature selection sees every row, including the future test folds
    X_selected = SelectKBest(f_classif, k=20).fit_transform(X, y)
    leaky = cross_val_score(LogisticRegression(max_iter=1000), X_selected, y, cv=5).mean()

    # right: selection is refit inside every training fold
    pipe = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000))
    honest = cross_val_score(pipe, X, y, cv=5).mean()
    return float(leaky), float(honest)


if __name__ == "__main__":
    leaky, honest = run()
    print(f"labels are random, so the real accuracy should be about 0.50")
    print(f"leaky CV accuracy:  {leaky:.3f}")
    print(f"honest CV accuracy: {honest:.3f}")
