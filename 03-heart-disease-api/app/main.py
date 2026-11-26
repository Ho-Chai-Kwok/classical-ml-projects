"""FastAPI service for the heart disease model.

Run locally:  uvicorn app.main:app --reload
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException, Request

sys.path.append(str(Path(__file__).resolve().parent.parent))
from model import to_frame  # noqa: E402

from .schemas import BatchPrediction, BatchRequest, Patient, Prediction  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("heart-api")

ARTIFACTS = Path(os.getenv("ARTIFACTS_DIR", Path(__file__).resolve().parent.parent / "artifacts"))
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load once at startup, not on every request
    model_path = ARTIFACTS / "model.joblib"
    if not model_path.exists():
        raise RuntimeError(f"no model at {model_path}. Run python train.py first.")
    state["model"] = joblib.load(model_path)
    state["meta"] = json.loads((ARTIFACTS / "metadata.json").read_text())
    log.info("loaded model %s", state["meta"]["model"])
    yield
    state.clear()


app = FastAPI(title="Heart disease risk API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def log_latency(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    log.info("%s %s -> %s in %.1fms", request.method, request.url.path, response.status_code, ms)
    response.headers["X-Process-Time-Ms"] = f"{ms:.1f}"
    return response


def _predict(patients: list[Patient]) -> list[Prediction]:
    model, meta = state.get("model"), state.get("meta")
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    frame = to_frame([p.model_dump() for p in patients])
    probs = model.predict_proba(frame)[:, 1]
    return [
        Prediction(probability=round(float(p), 4), has_disease=bool(p >= meta["threshold"]), model=meta["model"])
        for p in probs
    ]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/model")
def model_info() -> dict:
    return state.get("meta", {})


@app.post("/predict", response_model=Prediction)
def predict(patient: Patient) -> Prediction:
    return _predict([patient])[0]


@app.post("/predict/batch", response_model=BatchPrediction)
def predict_batch(body: BatchRequest) -> BatchPrediction:
    return BatchPrediction(predictions=_predict(body.patients))
