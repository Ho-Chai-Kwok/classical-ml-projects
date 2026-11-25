"""Compare models on an imbalanced problem, tune one, then touch the test set once.

Run: python train.py
Writes results/pr_curves.png and prints a metrics table.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from evaluation import best_f1_threshold, make_imbalanced_data, scores, three_way_split

RESULTS = Path(__file__).parent / "results"


def build_models(pos_weight: float) -> dict:
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "logreg_balanced": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")
        ),
        "svm_rbf_balanced": make_pipeline(
            StandardScaler(),
            # SVMs give margins, not probabilities, so calibrate them on top
            CalibratedClassifierCV(SVC(kernel="rbf", C=1.0, class_weight="balanced"), ensemble=False),
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=0),
        "xgboost_weighted": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, scale_pos_weight=pos_weight, eval_metric="aucpr"
        ),
    }


def compare_models(s) -> dict:
    pos_weight = float((s.y_train == 0).sum() / (s.y_train == 1).sum())
    fitted = {}
    print(f"positive rate in train: {s.y_train.mean():.3%}\n")
    print(f"{'model':20s} {'acc':>6s} {'f1@0.5':>7s} {'pr_auc':>7s} {'roc_auc':>8s} {'secs':>6s}")
    for name, model in build_models(pos_weight).items():
        start = time.perf_counter()
        model.fit(s.X_train, s.y_train)
        proba = model.predict_proba(s.X_val)[:, 1]
        m = scores(s.y_val, proba)
        print(
            f"{name:20s} {m['accuracy']:6.3f} {m['f1']:7.3f} {m['pr_auc']:7.3f} "
            f"{m['roc_auc']:8.3f} {time.perf_counter() - start:6.1f}"
        )
        fitted[name] = model
    always_zero = 1 - s.y_val.mean()
    print(f"\nfor reference, predicting 'negative' for everything gives accuracy {always_zero:.3f}")
    return fitted


def compare_search(s) -> XGBClassifier:
    """Grid search vs random search with roughly the same budget."""
    pos_weight = float((s.y_train == 0).sum() / (s.y_train == 1).sum())
    base = XGBClassifier(eval_metric="aucpr", scale_pos_weight=pos_weight, n_jobs=1)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

    grid = GridSearchCV(
        base,
        {"max_depth": [3, 5, 7], "learning_rate": [0.03, 0.1, 0.3], "n_estimators": [150, 300]},
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
    )
    rand = RandomizedSearchCV(
        base,
        {
            "max_depth": randint(2, 9),
            "learning_rate": loguniform(0.01, 0.4),
            "n_estimators": randint(100, 500),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.5, 0.5),
        },
        n_iter=18,  # same number of candidates as the grid above
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        random_state=0,
    )

    print("\nhyperparameter search, 18 candidates each, 3-fold CV on train:")
    best = None
    for name, search in (("grid", grid), ("random", rand)):
        start = time.perf_counter()
        search.fit(s.X_train, s.y_train)
        print(f"  {name:6s} best cv pr_auc {search.best_score_:.3f} in {time.perf_counter() - start:.1f}s")
        params = {k: round(v, 3) if isinstance(v, float) else v for k, v in search.best_params_.items()}
        print(f"         params {params}")
        if best is None or search.best_score_ > best.best_score_:
            best = search
    return best.best_estimator_


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    X, y = make_imbalanced_data()
    s = three_way_split(X, y)

    fitted = compare_models(s)
    tuned = compare_search(s)

    val_proba = tuned.predict_proba(s.X_val)[:, 1]
    threshold, val_f1 = best_f1_threshold(s.y_val, val_proba)
    print(f"\nthreshold picked on validation: {threshold:.3f} (val f1 {val_f1:.3f})")

    test_proba = tuned.predict_proba(s.X_test)[:, 1]
    default = scores(s.y_test, test_proba, 0.5)
    tuned_t = scores(s.y_test, test_proba, threshold)
    print("final test set, used once:")
    print(f"  f1 at 0.5:          {default['f1']:.3f}")
    print(f"  f1 at tuned thresh: {tuned_t['f1']:.3f}")
    print(f"  pr_auc:             {tuned_t['pr_auc']:.3f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    for name in ("logreg", "svm_rbf_balanced", "xgboost_weighted"):
        PrecisionRecallDisplay.from_estimator(fitted[name], s.X_val, s.y_val, ax=ax, name=name)
    PrecisionRecallDisplay.from_predictions(s.y_val, val_proba, ax=ax, name="xgboost_tuned")
    ax.axhline(s.y_val.mean(), linestyle="--", color="grey", label="chance")
    ax.set_title("Precision-recall on validation")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(RESULTS / "pr_curves.png", dpi=120)
    print(f"\nsaved {RESULTS / 'pr_curves.png'}")


if __name__ == "__main__":
    np.random.seed(0)
    main()
