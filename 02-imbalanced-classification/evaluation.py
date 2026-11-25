"""Splitting, metrics and threshold tuning helpers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


@dataclass
class Splits:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def make_imbalanced_data(n_samples: int = 20_000, positive_rate: float = 0.03, seed: int = 7):
    """Synthetic stand-in for something like fraud: few positives, some noisy labels."""
    return make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=6,
        n_redundant=4,
        weights=[1 - positive_rate, positive_rate],
        flip_y=0.01,
        class_sep=0.8,
        random_state=seed,
    )


def three_way_split(X, y, seed: int = 0) -> Splits:
    """60 / 20 / 20, stratified so every split keeps the same positive rate."""
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed)
    return Splits(X_train, X_val, X_test, y_train, y_val, y_test)


def scores(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def best_f1_threshold(y_true, proba) -> tuple[float, float]:
    """Pick the threshold that maximises F1. Do this on validation, never on test."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # the last precision/recall pair has no threshold attached
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    i = int(np.argmax(f1))
    return float(thresholds[i]), float(f1[i])
