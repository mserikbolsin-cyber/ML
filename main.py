"""FastAPI app that serves predictions from the trained Iris model."""
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "model.joblib"
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

app = FastAPI(title="Iris ML API", description="Predicts the Iris flower species")

# The trained model is loaded once when the API starts.
model = joblib.load(MODEL_PATH)


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Sepal length in cm", examples=[5.1])
    sepal_width: float = Field(..., gt=0, description="Sepal width in cm", examples=[3.5])
    petal_length: float = Field(..., gt=0, description="Petal length in cm", examples=[1.4])
    petal_width: float = Field(..., gt=0, description="Petal width in cm", examples=[0.2])


@app.get("/")
def root():
    return {"message": "ML API is running"}


@app.post("/predict")
def predict(features: IrisFeatures):
    x = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]]
    class_id = int(model.predict(x)[0])
    probability = float(model.predict_proba(x)[0][class_id])
    return {
        "prediction": class_id,
        "class_name": CLASS_NAMES[class_id],
        "probability": round(probability, 3),
    }
