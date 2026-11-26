"""Train and compare models, then save the winner for the API.

Run: python train.py
Outputs artifacts/model.joblib and artifacts/metadata.json
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from model import FEATURES, build_preprocessor, load_data

ARTIFACTS = Path(__file__).parent / "artifacts"


def candidates() -> dict[str, object]:
    return {
        "decision_tree": DecisionTreeClassifier(max_depth=4, random_state=0),
        "logistic_regression": LogisticRegression(max_iter=2000),
        "gradient_boosting": GradientBoostingClassifier(random_state=0),
        "xgboost": XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.9, eval_metric="logloss"
        ),
    }


def main() -> None:
    X, y = load_data()
    print(f"{len(X)} patients, {y.mean():.1%} with heart disease\n")

    # hold out a test set first. Model selection happens only on the rest.
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    results = {}
    print(f"{'model':22s} {'cv roc_auc':>12s} {'cv f1':>10s}")
    for name, clf in candidates().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        cv_res = cross_validate(pipe, X_dev, y_dev, cv=cv, scoring=["roc_auc", "f1"])
        auc, f1 = cv_res["test_roc_auc"], cv_res["test_f1"]
        results[name] = float(auc.mean())
        print(f"{name:22s} {auc.mean():6.3f} +- {auc.std():.3f} {f1.mean():6.3f}")

    best_name = max(results, key=results.get)
    best = Pipeline([("prep", build_preprocessor()), ("clf", candidates()[best_name])])
    best.fit(X_dev, y_dev)

    proba = best.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, proba)
    test_f1 = f1_score(y_test, (proba >= 0.5).astype(int))
    print(f"\npicked {best_name}. held-out test: roc_auc {test_auc:.3f}, f1 {test_f1:.3f}")

    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(best, ARTIFACTS / "model.joblib")
    metadata = {
        "model": best_name,
        "features": FEATURES,
        "threshold": 0.5,
        "cv_roc_auc": {k: round(v, 4) for k, v in results.items()},
        "test_roc_auc": round(float(test_auc), 4),
        "test_f1": round(float(test_f1), 4),
        "n_train": int(len(X_dev)),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (ARTIFACTS / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"saved model and metadata to {ARTIFACTS}/")


if __name__ == "__main__":
    np.random.seed(0)
    main()
