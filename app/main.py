import json
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

# ---------- Load the trained model (once, when the app starts) ----------
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

model = joblib.load(ARTIFACTS_DIR / "model.joblib")

with open(ARTIFACTS_DIR / "model_info.json", encoding="utf-8") as f:
    model_info = json.load(f)

FEATURE_COLUMNS = model_info["feature_columns"]

# ---------- Create the app ----------
app = FastAPI(
    title="Medical Insurance Cost Prediction API",
    description="Predicts yearly medical insurance charges from personal details.",
    version="1.0.0",
)


# ---------- Request and response schemas ----------
class InsuranceInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "age": 35,
                    "sex": "female",
                    "bmi": 27.5,
                    "children": 2,
                    "smoker": "no",
                    "region": "northeast",
                }
            ]
        }
    )

    age: int = Field(..., ge=18, le=64, description="Age in years (18 to 64)")
    sex: Literal["female", "male"]
    bmi: float = Field(..., ge=15, le=55, description="Body mass index (15 to 55)")
    children: int = Field(..., ge=0, le=5, description="Number of children (0 to 5)")
    smoker: Literal["no", "yes"]
    region: Literal["northeast", "northwest", "southeast", "southwest"]


class PredictionResponse(BaseModel):
    predicted_charges: float


# ---------- Endpoints ----------
@app.get("/")
def root():
    return {
        "message": "Medical Insurance Cost Prediction API is running.",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: InsuranceInput):
    try:
        row = pd.DataFrame([data.model_dump()])[FEATURE_COLUMNS]
        prediction = model.predict(row)[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc

    return PredictionResponse(predicted_charges=round(float(prediction), 2))