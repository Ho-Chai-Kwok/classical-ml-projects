# AdaBoost from scratch

Level: beginner

I wanted to understand boosting properly instead of just calling `XGBClassifier()` and hoping. So I wrote discrete AdaBoost with decision stumps using only numpy, then lined it up against scikit-learn's AdaBoost, gradient boosting and XGBoost on the breast cancer dataset that ships with scikit-learn.

## What is in here

- `adaboost.py` - the implementation. About 80 lines. A stump picks one feature and one threshold, and each round reweights the samples so the next stump focuses on the ones the ensemble keeps getting wrong.
- `compare.py` - trains all four models on the same split and prints accuracy and fit time.
- `test_adaboost.py` - small tests, including one on XOR-shaped data where a single stump can't do better than a coin flip but the ensemble can.

## Run it

```bash
pip install -r ../requirements.txt
python compare.py
pytest
```

## What I got

| model | test accuracy | fit time |
|---|---|---|
| my AdaBoost (100 stumps) | 0.972 | 0.60s |
| sklearn AdaBoost (100 stumps) | 0.965 | 0.12s |
| sklearn GradientBoosting | 0.958 | 0.17s |
| XGBoost | 0.965 | 0.06s |

The test set is only 143 rows, so a difference of one or two points here is one or two patients. I wouldn't read anything into my version "winning". The point is that it lands in the same place as the library, which tells me the maths is right.

Training error drops to zero after around 50 stumps, while test accuracy keeps holding up. That was the part that surprised me the first time: boosting keeps pushing the margin wider even after the training set is perfectly classified.

## Things I learned

- The `alpha` for each stump is `0.5 * log((1 - err) / err)`. A stump that is barely better than random gets a weight close to zero. A stump that is worse than random gets a negative weight, which flips it.
- You have to clip the error. A perfect stump gives `err = 0` and the log blows up.
- AdaBoost is gradient boosting with exponential loss. Gradient boosting generalises the idea to any differentiable loss, and XGBoost adds regularisation, second order gradients and a lot of engineering for speed. Writing the simple one first made the other two much easier to follow.
- My version is slow because it tries 20 thresholds per feature in pure Python loops. Fine for learning, not for production.
