import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field



BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "xgb_diabetes_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
meta = joblib.load(os.path.join(BASE_DIR, "model_meta.pkl"))

FEATURE_COLS = meta["feature_cols"]
CONTINUOUS_COLS = meta["continuous_cols"]
DECISION_THRESHOLD = meta.get("decision_threshold", 0.30)



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
    """Response schema for the /predict endpoint."""
    risk_score: float = Field(..., description="Probability of Diabetes/Prediabetes (0.0 to 1.0)")
    risk_percentage: float = Field(..., description="Risk score as percentage (0 to 100)")
    flagged_for_review: bool = Field(..., description="True if risk_score exceeds clinical threshold")
    risk_level: str = Field(..., description="Risk category: Low, Moderate, High, Very High")
    decision_threshold: float = Field(..., description="The threshold used for flagging")



app = FastAPI(
    title="DiaSense API",
    description="Diabetes Risk Prediction API powered by XGBoost on CDC BRFSS 2015 data",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"service": "DiaSense API", "status": "running", "version": "1.0.0"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "features": len(FEATURE_COLS),
        "decision_threshold": DECISION_THRESHOLD,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(survey: PatientSurvey):
    """
    Accept a patient survey, run inference, return risk score.
    """
    data = survey.model_dump()
    df = pd.DataFrame([data], columns=FEATURE_COLS)

    df[CONTINUOUS_COLS] = scaler.transform(df[CONTINUOUS_COLS])

    risk_score = float(model.predict_proba(df)[:, 1][0])
    risk_percentage = round(risk_score * 100, 1)

    flagged = risk_score >= DECISION_THRESHOLD

    if risk_score < 0.20:
        risk_level = "Low"
    elif risk_score < 0.40:
        risk_level = "Moderate"
    elif risk_score < 0.60:
        risk_level = "High"
    else:
        risk_level = "Very High"

    return PredictionResponse(
        risk_score=round(risk_score, 4),
        risk_percentage=risk_percentage,
        flagged_for_review=flagged,
        risk_level=risk_level,
        decision_threshold=DECISION_THRESHOLD,
    )