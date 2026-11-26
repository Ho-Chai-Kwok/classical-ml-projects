"""Data loading and the model pipeline, shared by training and the API."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
)
DATA_PATH = Path(__file__).parent / "data" / "cleveland.csv"

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
FEATURES = NUMERIC + CATEGORICAL
COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]


def as_category(series: pd.Series) -> pd.Series:
    """Codes like 1.0 or 3 become the strings '1' and '3'. Missing stays NaN."""
    return series.map(lambda v: np.nan if pd.isna(v) else str(int(v))).astype(object)


def load_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load the Cleveland subset of the UCI heart disease data.

    The raw file uses '?' for missing values (a few rows in `ca` and `thal`).
    The original target goes from 0 to 4; anything above 0 means disease,
    which is how almost everyone uses this dataset.
    """
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(DATA_URL, header=None).to_csv(path, index=False, header=False)

    df = pd.read_csv(path, header=None, names=COLUMNS, na_values="?")
    y = (df["num"] > 0).astype(int)
    X = df[FEATURES].copy()
    for col in CATEGORICAL:
        X[col] = as_category(X[col])
    return X, y


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])


def to_frame(records: list[dict]) -> pd.DataFrame:
    """Turn API input into the same dtypes the model saw during training."""
    df = pd.DataFrame.from_records(records)[FEATURES]
    for col in CATEGORICAL:
        df[col] = as_category(df[col])
    for col in NUMERIC:
        df[col] = df[col].astype(float)
    return df
