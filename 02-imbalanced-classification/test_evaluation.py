import numpy as np

from evaluation import best_f1_threshold, make_imbalanced_data, scores, three_way_split
from leakage_demo import run


def test_split_keeps_positive_rate():
    X, y = make_imbalanced_data(n_samples=5000)
    s = three_way_split(X, y)
    rates = [s.y_train.mean(), s.y_val.mean(), s.y_test.mean()]
    assert max(rates) - min(rates) < 0.005
    assert len(s.y_train) + len(s.y_val) + len(s.y_test) == 5000


def test_threshold_beats_default_on_skewed_scores():
    y = np.array([0] * 90 + [1] * 10)
    # positives score higher than negatives but all stay under 0.5
    proba = np.concatenate([np.linspace(0.0, 0.2, 90), np.linspace(0.25, 0.4, 10)])
    t, f1 = best_f1_threshold(y, proba)
    assert scores(y, proba, 0.5)["f1"] == 0
    assert f1 == 1.0 and 0.2 < t <= 0.25


def test_leakage_inflates_score():
    leaky, honest = run(seed=1)
    assert leaky > honest + 0.15
