from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Patient(BaseModel):
    """One patient record. Ranges are loose sanity checks, not medical limits."""

    age: int = Field(ge=18, le=100)
    sex: Literal[0, 1] = Field(description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(description="chest pain type")
    trestbps: float = Field(ge=60, le=250, description="resting blood pressure, mm Hg")
    chol: float = Field(ge=80, le=700, description="serum cholesterol, mg/dl")
    fbs: Literal[0, 1] = Field(description="fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2]
    thalach: float = Field(ge=50, le=250, description="max heart rate reached")
    exang: Literal[0, 1] = Field(description="exercise induced angina")
    oldpeak: float = Field(ge=0, le=10)
    slope: Literal[1, 2, 3]
    ca: int = Field(ge=0, le=3, description="major vessels coloured by fluoroscopy")
    thal: Literal[3, 6, 7] = Field(description="3 normal, 6 fixed defect, 7 reversible defect")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 2,
                "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
            }
        }
    }


class Prediction(BaseModel):
    probability: float
    has_disease: bool
    model: str


class BatchRequest(BaseModel):
    patients: list[Patient] = Field(min_length=1, max_length=500)


class BatchPrediction(BaseModel):
    predictions: list[Prediction]
