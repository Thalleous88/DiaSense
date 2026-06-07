import os
import uuid
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from database import get_db, init_db
from models import Assessment, TrainingRun, UploadedDataset
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
    await init_db()
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
async def predict(survey: PatientSurvey, db: AsyncSession = Depends(get_db)):
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

    assessment = Assessment(
        patient_id=patient_id,
        input_data=data,
        risk_score=round(risk_score, 4),
        risk_percentage=risk_percentage,
        risk_level=risk_level,
        flagged_for_review=flagged,
        decision_threshold=decision_threshold,
    )
    db.add(assessment)
    await db.commit()

    return PredictionResponse(
        risk_score=round(risk_score, 4),
        risk_percentage=risk_percentage,
        flagged_for_review=flagged,
        risk_level=risk_level,
        decision_threshold=decision_threshold,
        patient_id=patient_id,
    )


@app.post("/train/upload")
async def upload_csv(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
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

        uploaded = UploadedDataset(
            filename=safe_name,
            original_name=file.filename,
            n_rows=validation["n_rows"],
            n_valid_rows=validation.get("n_valid_rows", 0),
            validation_errors=errs,
        )
        db.add(uploaded)
        await db.commit()
        await db.refresh(uploaded)

        return {
            "upload_id": uploaded.id,
            "filename": file.filename,
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
async def start_training(upload_id: int, db: AsyncSession = Depends(get_db)):
    status = get_training_status()
    if status["status"] == "running":
        raise HTTPException(status_code=409, detail="Training already in progress")

    result = await db.execute(
        text("SELECT * FROM uploaded_datasets WHERE id = :id"), {"id": upload_id}
    )
    row = result.mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Upload not found")

    file_path = os.path.join(
        os.path.dirname(__file__), settings.UPLOAD_DIR, row["filename"]
    )
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Uploaded file not found on disk")

    training_run = TrainingRun(
        status="pending",
        n_new_samples=row["n_valid_rows"],
        started_at=datetime.utcnow(),
    )
    db.add(training_run)
    await db.commit()
    await db.refresh(training_run)

    try:
        db_url = settings.DATABASE_URL
        start_background_training(file_path, training_run.id, db_url)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return {"training_run_id": training_run.id, "status": "started"}


@app.get("/train/status")
async def training_status():
    return get_training_status()


@app.get("/train/history")
async def training_history(limit: int = Query(20, ge=1, le=100), db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        text("SELECT * FROM training_runs ORDER BY created_at DESC LIMIT :lim"),
        {"lim": limit}
    )
    runs = result.mappings().all()
    return [
        {
            "id": r["id"],
            "status": r["status"],
            "n_new_samples": r["n_new_samples"],
            "n_total_samples": r["n_total_samples"],
            "roc_auc": r["roc_auc"],
            "recall_class1": r["recall_class1"],
            "pr_auc": r["pr_auc"],
            "f1_score": r["f1_score"],
            "decision_threshold": r["decision_threshold"],
            "model_type": r["model_type"],
            "error_message": r["error_message"],
            "started_at": r["started_at"].isoformat() if r["started_at"] else None,
            "completed_at": r["completed_at"].isoformat() if r["completed_at"] else None,
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        }
        for r in runs
    ]


@app.get("/feature-importance")
async def feature_importance():
    _, _, meta = get_active_model()
    if meta is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "importance": meta.get("feature_importance", {}),
        "top_features": meta.get("top_features_for_api", {}),
    }


@app.get("/stats")
async def dashboard_stats(db: AsyncSession = Depends(get_db)):
    result = await db.execute(text("SELECT COUNT(*) as total FROM assessments"))
    total = result.scalar() or 0

    result = await db.execute(
        text("SELECT COUNT(*) as flagged FROM assessments WHERE flagged_for_review = true")
    )
    high_risk = result.scalar() or 0

    positivity = round((high_risk / total * 100), 1) if total > 0 else 0

    return {
        "total_screened": total,
        "high_risk_count": high_risk,
        "positivity_rate": f"{positivity}%",
        "screened_subtitle": f"{total} assessments completed",
        "high_risk_subtitle": f"{positivity}% positivity rate",
        "avg_time": "< 150ms",
    }


@app.get("/assessments/recent")
async def recent_assessments(limit: int = Query(5, ge=1, le=50), db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        text("SELECT * FROM assessments ORDER BY created_at DESC LIMIT :lim"),
        {"lim": limit}
    )
    rows = result.mappings().all()
    return [
        {
            "patient_id": r["patient_id"],
            "risk_score": r["risk_score"],
            "risk_percentage": r["risk_percentage"],
            "risk_level": r["risk_level"],
            "flagged_for_review": r["flagged_for_review"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        }
        for r in rows
    ]


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
