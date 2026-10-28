from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any

ARTIFACT_DIR = Path("artifacts")
metadata_path = ARTIFACT_DIR / "model_metadata.json"

if not metadata_path.exists():
    raise FileNotFoundError(
        "Model artifacts were not found. Run the notebook through the Model Export cell first."
    )

with metadata_path.open("r", encoding="utf-8") as f:
    metadata = json.load(f)

label_encoder = joblib.load(ARTIFACT_DIR / "label_encoder.pkl")
feature_columns = joblib.load(ARTIFACT_DIR / "feature_columns.pkl")

if metadata["model_format"] == "keras":
    import tensorflow as tf
    model = tf.keras.models.load_model(ARTIFACT_DIR / "student_model.keras")
    preprocessor = joblib.load(ARTIFACT_DIR / "preprocessor.pkl")
elif metadata["model_format"] == "sklearn_pipeline":
    model = joblib.load(ARTIFACT_DIR / "selected_sklearn_pipeline.joblib")
    preprocessor = None  # The fitted preprocessor is already inside the sklearn pipeline.
else:
    raise ValueError(f"Unsupported model format: {metadata['model_format']}")

app = FastAPI(
    title="Student Academic Outcome Prediction API",
    description="Predicts Dropout, Enrolled, or Graduate from student input features.",
    version="1.0.0",
)

class PredictionRequest(BaseModel):
    features: dict[str, Any]

@app.get("/")
def root():
    return {
        "message": "Student Academic Outcome Prediction API",
        "selected_model": metadata["selected_model_name"],
        "docs": "/docs",
    }

@app.get("/health")
def health():
    return {"status": "ok", "selected_model": metadata["selected_model_name"]}

@app.post("/predict")
def predict(request: PredictionRequest):
    missing = [col for col in feature_columns if col not in request.features]
    if missing:
        raise HTTPException(
            status_code=400,
            detail={"message": "Missing required input features", "missing_features": missing},
        )

    try:
        row = pd.DataFrame(
            [[request.features[col] for col in feature_columns]],
            columns=feature_columns,
        )

        if metadata["model_format"] == "keras":
            transformed = preprocessor.transform(row)
            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()
            probabilities = model.predict(np.asarray(transformed), verbose=0)[0]
            predicted_id = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_id])
        else:
            predicted_id = int(model.predict(row)[0])
            confidence = None
            if hasattr(model, "predict_proba"):
                confidence = float(np.max(model.predict_proba(row)[0]))

        predicted_label = str(label_encoder.inverse_transform([predicted_id])[0])
        response = {
            "prediction": predicted_label,
            "class_id": predicted_id,
            "confidence": confidence,
            "selected_model": metadata["selected_model_name"],
        }
        return response
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not process input features: {exc}")
