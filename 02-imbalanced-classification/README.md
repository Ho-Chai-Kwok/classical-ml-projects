# Imbalanced classification done carefully

Level: intermediate

This one is about the boring parts that decide whether a model is actually any good: how you split the data, what metric you look at, how you tune, and how you avoid fooling yourself. The data is synthetic (20,000 rows, about 3.5% positives) so I can control the imbalance. Think of it as a stand-in for fraud or churn.

## What is in here

- `evaluation.py` - a stratified 60/20/20 train/validation/test split, metric helpers, and a function that picks the decision threshold that maximises F1 on validation.
- `leakage_demo.py` - a short experiment on purpose-built garbage data that shows how leakage fakes a good score.
- `train.py` - compares logistic regression, an RBF SVM, gradient boosting and XGBoost, runs grid search against random search, tunes the threshold, and only then looks at the test set. Saves precision-recall curves to `results/pr_curves.png`.
- `test_evaluation.py` - tests for the split, the threshold picker and the leakage demo.

## Run it

```bash
pip install -r ../requirements.txt
python leakage_demo.py
python train.py
pytest
```

## Leakage, in one number

The labels in `leakage_demo.py` are random coin flips, so no model can do better than about 50%. If I select the 20 "best" features from 5,000 noise columns using all of the data and then cross-validate, I get **0.795** accuracy. If the selection happens inside a `Pipeline`, so each fold only sees its own training rows, I get **0.510**. Same model, same data. The first number is a lie.

## Results on validation

| model | accuracy | F1 at 0.5 | PR-AUC | ROC-AUC |
|---|---|---|---|---|
| logistic regression | 0.965 | 0.000 | 0.125 | 0.607 |
| logistic regression, balanced | 0.588 | 0.088 | 0.083 | 0.620 |
| SVM (RBF), balanced, calibrated | 0.969 | 0.262 | 0.404 | 0.856 |
| gradient boosting | 0.968 | 0.198 | 0.357 | 0.836 |
| XGBoost with `scale_pos_weight` | 0.917 | 0.332 | 0.434 | 0.850 |

Predicting "negative" for every row gets 0.965 accuracy. Plain logistic regression basically does that, and its F1 is zero. This table is the whole reason accuracy is the wrong metric for imbalanced problems.

PR-AUC is the column I care about. Its baseline is the positive rate (about 0.035), not 0.5 like ROC-AUC, so it is much harsher and much more honest when positives are rare.

## Grid search vs random search

Both got the same budget: 18 candidates, 3-fold CV, scored on average precision.

| search | best CV PR-AUC | time |
|---|---|---|
| grid (3 x 3 x 2) | 0.562 | 6.7s |
| random (5 parameters) | 0.556 | 5.4s |

Basically a tie here. Random search tuned five parameters for the same price as the grid's three, which is why it usually wins once you have more than a couple of parameters that matter.

## Threshold tuning

The default 0.5 threshold is arbitrary. I picked the F1-maximising threshold on validation (0.294), then checked the test set once:

- F1 at 0.5: 0.543
- F1 at the tuned threshold: 0.607
- PR-AUC: 0.574

The tuned threshold came from validation only. If I had picked it on test, the 0.607 would be optimistic.

## Things I learned

- Stratify every split, or one of them might end up with almost no positives.
- Class weights and `scale_pos_weight` help the model pay attention to the rare class, but they mostly move the scores around. Threshold tuning is often the bigger win for F1.
- SVMs don't give probabilities. `SVC(probability=True)` is deprecated in recent scikit-learn, so I wrapped it in `CalibratedClassifierCV`. The SVM was also the slowest model by far. They are worth knowing, but on tabular data like this boosting is faster and at least as good.
