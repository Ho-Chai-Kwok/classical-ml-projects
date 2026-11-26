# Heart disease risk model behind an API

Level: intermediate to advanced. This is the end-to-end one.

I started with a decision tree on the UCI heart disease data in a notebook. This project takes that idea and does it properly: a real train/test protocol, a fair comparison against stronger models, a saved pipeline, and a FastAPI service in a Docker image that anyone can call.

Not medical advice, obviously. It is 303 patients from Cleveland in 1988.

## How it works

```
train.py  ->  artifacts/model.joblib + metadata.json  ->  app/main.py (FastAPI)  ->  Docker
```

- `model.py` downloads the Cleveland data from the UCI archive on first run, cleans the `?` values, and builds one scikit-learn `Pipeline` (imputation, scaling, one-hot encoding, model). Training and serving use the same code, so the API can't quietly preprocess things differently.
- `train.py` holds out 20% as a test set first, then runs 5-fold CV on the rest for four models, picks the best by ROC-AUC, refits it and checks the test set once.
- `app/` has the API. Pydantic checks every field (a 5-year-old or a `thal` of 4 gets a 422 instead of a silent garbage prediction). The model loads once at startup. A middleware logs latency for every request and adds an `X-Process-Time-Ms` header.
- `tests/` trains a tiny model on fake data, so the API tests run in about a second with no download.

## Run it

```bash
pip install -r requirements.txt
python train.py
uvicorn app.main:app --reload
```

Then open http://localhost:8000/docs, or:

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
```

With Docker (the image trains the model during the build):

```bash
docker build -t heart-api .
docker run -p 8000:8000 heart-api
```

Endpoints: `GET /health`, `GET /model`, `POST /predict`, `POST /predict/batch` (up to 500 patients).

## Results

5-fold CV on the 242 development rows:

| model | CV ROC-AUC | CV F1 |
|---|---|---|
| decision tree (depth 4) | 0.788 +- 0.057 | 0.729 |
| logistic regression | 0.892 +- 0.022 | 0.803 |
| gradient boosting | 0.867 +- 0.026 | 0.756 |
| XGBoost | 0.872 +- 0.024 | 0.766 |

Logistic regression won, and the held-out test gave ROC-AUC 0.958 and F1 0.867. The test set is 61 people, so that number moves a lot with the random seed. The CV numbers are the more trustworthy ones.

I honestly expected XGBoost to win. It didn't, and I think that is the most useful thing this project taught me. With 300 rows and features that are mostly linear risk factors, a simple model is hard to beat. Boosting is the default for tabular data at work because work datasets are usually much bigger and messier than this. The decision tree I started with was the worst of the four and the least stable across folds.

## Deploying

The container reads `PORT` from the environment, so it runs as-is on Render, Railway, Fly.io or Google Cloud Run. Point the service at this folder's Dockerfile and it builds and trains in one go.

## What I would add next

- A model registry or at least versioned artifacts in object storage instead of baking the model into the image.
- Monitoring for input drift. If the API starts getting patients that look nothing like the 1988 data, the predictions mean nothing.
- Calibration plots. For a risk score, "0.7" should actually mean about 70%.
