import numpy as np
import pytest

from adaboost import AdaBoost, fit_stump


def test_stump_finds_obvious_split():
    X = np.array([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]])
    y = np.array([-1, -1, -1, 1, 1, 1])
    w = np.full(6, 1 / 6)
    stump, err = fit_stump(X, y, w)
    assert err == 0
    assert np.array_equal(stump.predict(X), y)


def test_boosting_beats_single_stump_on_xor_like_data():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(400, 2))
    y = np.where((X[:, 0] > 0) ^ (X[:, 1] > 0), 1, -1)
    model = AdaBoost(n_estimators=1).fit(X, y)
    one = np.mean(model.predict(X) == y)
    model = AdaBoost(n_estimators=60).fit(X, y)
    many = np.mean(model.predict(X) == y)
    assert many > one


def test_rejects_zero_one_labels():
    with pytest.raises(ValueError):
        AdaBoost().fit(np.zeros((3, 1)), np.array([0, 1, 1]))
