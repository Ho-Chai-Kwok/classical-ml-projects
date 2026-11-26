"""API tests. They train a small model on fake rows so they run offline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model import CATEGORICAL, FEATURES, as_category, build_preprocessor  # noqa: E402

EXAMPLE = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 2,
    "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
}


def fake_training_data(n: int = 120) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "age": rng.integers(30, 80, n), "trestbps": rng.normal(130, 15, n), "chol": rng.normal(240, 40, n),
        "thalach": rng.normal(150, 20, n), "oldpeak": rng.uniform(0, 4, n), "ca": rng.integers(0, 4, n),
        "sex": rng.integers(0, 2, n), "cp": rng.integers(1, 5, n), "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n), "exang": rng.integers(0, 2, n), "slope": rng.integers(1, 4, n),
        "thal": rng.choice([3, 6, 7], n),
    })[FEATURES]
    for col in CATEGORICAL:
        df[col] = as_category(df[col])
    y = (df["oldpeak"] > 2).astype(int).to_numpy()
    return df, y


@pytest.fixture()
def client(tmp_path, monkeypatch):
    X, y = fake_training_data()
    pipe = Pipeline([("prep", build_preprocessor()), ("clf", LogisticRegression(max_iter=500))]).fit(X, y)
    joblib.dump(pipe, tmp_path / "model.joblib")
    (tmp_path / "metadata.json").write_text(json.dumps({"model": "test_logreg", "threshold": 0.5}))

    import app.main as main
    monkeypatch.setattr(main, "ARTIFACTS", tmp_path)
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_probability(client):
    res = client.post("/predict", json=EXAMPLE)
    assert res.status_code == 200
    body = res.json()
    assert 0 <= body["probability"] <= 1
    assert body["model"] == "test_logreg"
    assert "X-Process-Time-Ms" in res.headers


def test_bad_input_is_rejected(client):
    bad = {**EXAMPLE, "age": 5, "thal": 4}
    res = client.post("/predict", json=bad)
    assert res.status_code == 422


def test_batch(client):
    res = client.post("/predict/batch", json={"patients": [EXAMPLE, {**EXAMPLE, "oldpeak": 0.1}]})
    assert res.status_code == 200
    preds = res.json()["predictions"]
    assert len(preds) == 2
    assert preds[0]["probability"] > preds[1]["probability"]
