import os
import uuid
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from train_service import (
    load_active_model, get_active_model, get_training_status,
    validate_csv, start_background_training
)
from config import settings

app = FastAPI(
    title="DiaSense API",
    description="Diabetes Risk Prediction API powered by Stacking Ensemble on CDC BRFSS 2015 data",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    load_active_model()


class PatientSurvey(BaseModel):
    HighBP: int = Field(..., ge=0, le=1, description="High Blood Pressure (0=No, 1=Yes)")
    HighChol: int = Field(..., ge=0, le=1, description="High Cholesterol (0=No, 1=Yes)")
    CholCheck: int = Field(..., ge=0, le=1, description="Cholesterol Check in past 5 years (0=No, 1=Yes)")
    Smoker: int = Field(..., ge=0, le=1, description="Have you smoked 100+ cigarettes in life (0=No, 1=Yes)")
    Stroke: int = Field(..., ge=0, le=1, description="Ever had a stroke (0=No, 1=Yes)")
    HeartDiseaseorAttack: int = Field(..., ge=0, le=1, description="CHD or MI (0=No, 1=Yes)")
    PhysActivity: int = Field(..., ge=0, le=1, description="Physical activity in past 30 days (0=No, 1=Yes)")
    Fruits: int = Field(..., ge=0, le=1, description="Consume fruit 1+ times/day (0=No, 1=Yes)")
    Veggies: int = Field(..., ge=0, le=1, description="Consume vegetables 1+ times/day (0=No, 1=Yes)")
    HvyAlcoholConsump: int = Field(..., ge=0, le=1, description="Heavy alcohol consumption (0=No, 1=Yes)")
    AnyHealthcare: int = Field(..., ge=0, le=1, description="Have any healthcare coverage (0=No, 1=Yes)")
    NoDocbcCost: int = Field(..., ge=0, le=1, description="Couldn't see doctor due to cost (0=No, 1=Yes)")
    DiffWalk: int = Field(..., ge=0, le=1, description="Difficulty walking or climbing stairs (0=No, 1=Yes)")
    Sex: int = Field(..., ge=0, le=1, description="Sex (0=Female, 1=Male)")
    BMI: int = Field(..., ge=10, le=99, description="Body Mass Index")
    GenHlth: int = Field(..., ge=1, le=5, description="General Health (1=Excellent to 5=Poor)")
    MentHlth: int = Field(..., ge=0, le=30, description="Days of poor mental health in past 30 days")
    PhysHlth: int = Field(..., ge=0, le=30, description="Days of poor physical health in past 30 days")
    Age: int = Field(..., ge=1, le=13, description="Age category (1=18-24 to 13=80+)")
    Education: int = Field(..., ge=1, le=6, description="Education level (1=Never attended to 6=College graduate)")
    Income: int = Field(..., ge=1, le=8, description="Income level (1=<$10k to 8=$75k+)")


class PredictionResponse(BaseModel):
    risk_score: float = Field(..., description="Probability of Diabetes/Prediabetes (0.0 to 1.0)")
    risk_percentage: float = Field(..., description="Risk score as percentage (0 to 100)")
    flagged_for_review: bool = Field(..., description="True if risk_score exceeds clinical threshold")
    risk_level: str = Field(..., description="Risk category: Low, Moderate, High, Very High")
    decision_threshold: float = Field(..., description="The threshold used for flagging")
    patient_id: str = Field(..., description="Unique patient assessment ID")


@app.get("/")
def root():
    return {"service": "DiaSense API", "status": "running", "version": "2.0.0"}


@app.get("/health")
def health_check():
    model, scaler, meta = get_active_model()
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": meta.get("model_type", "unknown") if meta else "unknown",
        "features": len(meta.get("feature_cols", [])) if meta else 0,
        "decision_threshold": meta.get("decision_threshold", 0) if meta else 0,
    }


def _engineer_features_for_prediction(data: dict, meta: dict) -> pd.DataFrame:
    base_feature_cols = meta.get("base_feature_cols", meta.get("feature_cols", []))
    engineered_cols = meta.get("engineered_cols", [])
    bmi_cap = meta.get("bmi_cap", 60)

    if not engineered_cols:
        return pd.DataFrame([data], columns=base_feature_cols)

    df = pd.DataFrame([data])

    bmi_val = min(data.get("BMI", 25), bmi_cap)
    df["BMI_x_Age"] = float(bmi_val * data.get("Age", 5))
    df["GenHlth_x_PhysHlth"] = float(data.get("GenHlth", 2) * data.get("PhysHlth", 0))
    df["BMI_x_HighBP"] = float(bmi_val * data.get("HighBP", 0))
    df["BMI_category"] = 3
    bmi = bmi_val
    if bmi <= 18.5:
        df["BMI_category"] = 1
    elif bmi <= 25:
        df["BMI_category"] = 2
    elif bmi <= 30:
        df["BMI_category"] = 3
    elif bmi <= 35:
        df["BMI_category"] = 4
    else:
        df["BMI_category"] = 5
    df["RiskScore_composite"] = (
        float(data.get("HighBP", 0)) +
        float(data.get("HighChol", 0)) +
        float(data.get("GenHlth", 2)) / 5.0 +
        float(1 if bmi > 30 else 0) +
        float(1 if data.get("Age", 5) > 7 else 0)
    ) / 5.0

    all_cols = base_feature_cols + engineered_cols
    return df[all_cols]


@app.post("/predict", response_model=PredictionResponse)
async def predict(survey: PatientSurvey):
    model, scaler, meta = get_active_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    data = survey.model_dump()
    df = _engineer_features_for_prediction(data, meta)

    continuous_cols = meta.get("continuous_cols", ["BMI", "MentHlth", "PhysHlth"])
    cols_to_scale = [c for c in continuous_cols if c in df.columns]
    if cols_to_scale:
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    feature_cols = meta.get("feature_cols", [])
    if feature_cols:
        df = df[feature_cols]

    risk_score = float(model.predict_proba(df)[:, 1][0])
    risk_percentage = round(risk_score * 100, 1)
    decision_threshold = meta.get("decision_threshold", 0.30)
    flagged = risk_score >= decision_threshold

    if risk_score < 0.20:
        risk_level = "Low"
    elif risk_score < 0.40:
        risk_level = "Moderate"
    elif risk_score < 0.60:
        risk_level = "High"
    else:
        risk_level = "Very High"

    patient_id = f"PAT-{uuid.uuid4().hex[:4].upper()}"

    return PredictionResponse(
        risk_score=round(risk_score, 4),
        risk_percentage=risk_percentage,
        flagged_for_review=flagged,
        risk_level=risk_level,
        decision_threshold=decision_threshold,
        patient_id=patient_id,
    )


@app.post("/train/upload")
async def upload_csv(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are accepted")

        contents = await file.read()
        max_size = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if len(contents) > max_size:
            raise HTTPException(status_code=400, detail=f"File too large (max {settings.MAX_UPLOAD_SIZE_MB}MB)")

        upload_dir = os.path.join(os.path.dirname(__file__), settings.UPLOAD_DIR)
        os.makedirs(upload_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = f"upload_{timestamp}_{file.filename}"
        file_path = os.path.join(upload_dir, safe_name)

        with open(file_path, "wb") as f:
            f.write(contents)

        validation = validate_csv(file_path)

        errs = validation.get("errors")
        if errs and not isinstance(errs, dict):
            errs = {"items": errs} if errs else None

        return {
            "filename_key": safe_name,
            "original_filename": file.filename,
            "valid": validation["valid"],
            "n_rows": validation["n_rows"],
            "n_valid_rows": validation.get("n_valid_rows", 0),
            "errors": validation.get("errors"),
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train/start")
async def start_training(filename_key: str):
    status = get_training_status()
    if status["status"] == "running":
        raise HTTPException(status_code=409, detail="Training already in progress")

    file_path = os.path.join(
        os.path.dirname(__file__), settings.UPLOAD_DIR, filename_key
    )
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Uploaded file not found on disk")

    try:
        start_background_training(file_path)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return {"status": "started", "message": "Training job dispatched to background."}


@app.get("/train/status")
async def training_status():
    return get_training_status()


from train_service import get_training_history

@app.get("/train/history")
async def training_history(limit: int = Query(20, ge=1, le=100)):
    return get_training_history(limit)


@app.get("/feature-importance")
async def feature_importance():
    _, _, meta = get_active_model()
    if meta is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "importance": meta.get("feature_importance", {}),
        "top_features": meta.get("top_features_for_api", {}),
    }





@app.get("/model/info")
async def model_info():
    _, _, meta = get_active_model()
    if meta is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_type": meta.get("model_type", "unknown"),
        "base_models": meta.get("base_models", []),
        "n_features": len(meta.get("feature_cols", [])),
        "decision_threshold": meta.get("decision_threshold", 0),
        "roc_auc": meta.get("roc_auc"),
        "recall_class1": meta.get("recall_class1"),
        "pr_auc": meta.get("pr_auc"),
        "f1_score": meta.get("f1_score"),
        "training_date": meta.get("training_date"),
        "n_training_samples": meta.get("n_training_samples"),
    }
