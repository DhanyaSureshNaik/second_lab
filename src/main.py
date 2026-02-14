from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from predict import predict_cancer
import datetime
import os

app = FastAPI(title="Breast Cancer Classifier API")

# log predictions
LOG_FILE = os.path.join(os.path.dirname(__file__), "../logs/predictions.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

class CancerData(BaseModel):
    features: List[float] = Field(..., min_items=30, max_items=30)

@app.post("/predict")
async def predict(data: CancerData):
    try:
        result = predict_cancer(data.features)
        # Log prediction
        with open(LOG_FILE, "a") as f:
            f.write(f"{datetime.datetime.now()} - {data.features} -> {result}\n")
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/info")
async def model_info():
    """Return model metadata"""
    return {
        "model_type": "LogisticRegression",
        "dataset": "Breast Cancer (sklearn.datasets.load_breast_cancer)",
        "input_features": 30,
        "classes": ["malignant", "benign"]
    }
