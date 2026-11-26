# Classical ML projects

Three projects that finish off the classical machine learning part of my roadmap: boosting, SVMs, honest model evaluation, and hyperparameter tuning. They get harder as they go.

| # | project | level | what it covers |
|---|---|---|---|
| 01 | [AdaBoost from scratch](01-adaboost-from-scratch) | beginner | boosting maths in numpy, compared with scikit-learn AdaBoost, gradient boosting and XGBoost |
| 02 | [Imbalanced classification](02-imbalanced-classification) | intermediate | train/val/test splits, data leakage, F1, PR curves, threshold tuning, SVMs, grid vs random search |
| 03 | [Heart disease API](03-heart-disease-api) | intermediate to advanced | full pipeline on real data, model comparison, FastAPI, Pydantic validation, logging, Docker, tests |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Each folder has its own README with results and notes. Every project has tests you can run with `pytest` from inside its folder.

## The short version of what I learned

1. Accuracy is close to useless when one class is rare. Look at PR-AUC and F1, and tune the threshold on validation.
2. Leakage is easy to cause and makes models look great. Keep every fitted step inside a `Pipeline`.
3. Boosting is the safe default for tabular data, but on a small, mostly linear dataset a logistic regression beat it. Always keep a simple baseline in the comparison.

Part of my [AI engineering roadmap](https://github.com/BDCALL/ai-engineering-roadmap).
