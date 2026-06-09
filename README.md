# DiaSense

AI-powered diabetes risk prediction system with an ensemble ML model, real-time FastAPI backend, and React frontend. Designed for high availability, scalability, and stateless execution via Docker.

## Architecture

```
DiaSense/
├── backend/                  # FastAPI + ML pipeline
│   ├── main.py               # API endpoints
│   ├── train.py              # Full offline ML pipeline (EDA, training, SHAP)
│   ├── train_service.py      # Background training + model hot-swap
│   ├── config.py             # Pydantic settings
│   ├── requirements.txt      # Python dependencies
│   ├── Dockerfile            # Uvicorn FastAPI Server
│   ├── figures/              # EDA visualizations (generated)
│   └── uploads/              # Uploaded CSV datasets (generated)
├── frontend/                 # React + Vite + Tailwind
│   ├── src/
│   │   ├── App.jsx           # Router setup
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx          # Risk assessment form + results
│   │   │   └── TrainingPage.jsx       # CSV upload + training controls
│   │   ├── components/
│   │   │   ├── Header.jsx             # Top navbar with navigation
│   │   │   ├── HealthForm.jsx         # 21-field patient survey
│   │   │   ├── RiskGauge.jsx          # Animated risk gauge
│   │   │   ├── ResultPanel.jsx        # Risk level + recommendations
│   │   │   ├── FeatureBreakdown.jsx   # SHAP feature importance
│   │   │   ├── UploadArea.jsx        # Drag-and-drop CSV upload
│   │   │   ├── TrainingStatus.jsx    # Real-time training progress
│   │   │   ├── TrainingHistory.jsx   # Past training runs table
│   │   │   └── ModelInfo.jsx         # Current model metrics
│   │   ├── hooks/
│   │   │   └── useApi.js             # API fetch wrapper
│   │   └── lib/
│   │       └── utils.js              # cn() utility
│   ├── Dockerfile            # Nginx Multi-stage build
│   └── nginx.conf            # Nginx config and /api proxy
└── docker-compose.yml        # Orchestrates Frontend, Backend, and MLflow
```

## ML Pipeline

The model is a **StackingClassifier** with three base learners and a logistic regression meta-learner:

| Layer | Model | Key Hyperparameters |
|-------|-------|-------------------|
| Base | XGBClassifier | 400 estimators, max_depth=5, lr=0.05, scale_pos_weight=3 |
| Base | LGBMClassifier | 400 estimators, max_depth=5, lr=0.05, class_weight=balanced |
| Base | RandomForestClassifier | 400 estimators, max_depth=10, class_weight=balanced |
| Meta | LogisticRegression | max_iter=1000 |

### Preprocessing

- Duplicate removal and BMI capping at 99th percentile
- 5 engineered features: `BMI_x_Age`, `GenHlth_x_PhysHlth`, `BMI_x_HighBP`, `BMI_category`, `RiskScore_composite`
- StandardScaler on continuous features
- SMOTE oversampling on training set (40/60 train-temp split, then 50/50 val-test)
- Decision threshold optimized to **0.15** for clinical recall >= 0.70

### Performance (Baseline)

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.80 |
| Recall (Class 1) | 0.72 |
| PR-AUC | 0.41 |
| F1 Score | 0.45 |

### SHAP Explainability

Feature importance is computed using SHAP TreeExplainer on the RandomForest base model within the ensemble, evaluated on a validation sample.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Service info |
| `GET` | `/health` | Health check + model status |
| `POST` | `/predict` | Generate diabetes risk prediction |
| `GET` | `/feature-importance` | SHAP feature importance |
| `GET` | `/model/info` | Current model metrics and metadata |
| `POST` | `/train/upload` | Upload CSV dataset for training |
| `POST` | `/train/start` | Start background model training |
| `GET` | `/train/status` | Current training progress |
| `GET` | `/train/history` | Past training runs (fetched from MLflow) |

### Predict Request

```json
{
  "HighBP": 1,
  "HighChol": 1,
  "CholCheck": 1,
  "BMI": 35,
  "Smoker": 1,
  "Stroke": 0,
  "HeartDiseaseorAttack": 0,
  "PhysActivity": 0,
  "Fruits": 0,
  "Veggies": 0,
  "HvyAlcoholConsump": 0,
  "AnyHealthcare": 1,
  "NoDocbcCost": 0,
  "GenHlth": 4,
  "MentHlth": 5,
  "PhysHlth": 10,
  "DiffWalk": 1,
  "Sex": 1,
  "Age": 9,
  "Education": 4,
  "Income": 3
}
```

### Predict Response

```json
{
  "risk_score": 0.8669,
  "risk_percentage": 86.7,
  "risk_level": "Very High",
  "flagged_for_review": true,
  "decision_threshold": 0.15,
  "patient_id": "PAT-6C48"
}
```

## Getting Started

### Prerequisites

- Docker & Docker Compose

### 1. Build and Run the Stack

Run the following command in the root directory to spin up the entire application stack:

```bash
docker compose up -d --build
```

This starts three services:
- **MLflow Tracking Server**: `http://localhost:5000`
- **FastAPI Backend**: `http://localhost:8000`
- **React Frontend**: `http://localhost:80`

### 2. Initial Model Generation

If you have never trained the model before, you need to execute the initial ML pipeline.

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt

python train.py
```

This script will generate your local EDA visualizations (`figures/`) and create the base model artifacts: `ensemble_model.pkl`, `scaler.pkl`, and `model_meta.pkl`.

## Model Hot-Swap & Online Learning

When new training is triggered via the frontend Training page:

1. User uploads a CSV via `POST /train/upload`
2. User clicks "Start Training" → `POST /train/start`
3. Training runs in a **background thread** (capped at `n_jobs=2` to prevent API starvation).
4. All run parameters and performance metrics are logged to the localized MLflow server.
5. On completion, the new model, scaler, and metadata are **atomically swapped** into the live API memory.
6. Subsequent predictions use the new model immediately without requiring an API restart.

The original BRFSS2015 dataset is always merged with uploaded data before retraining to prevent catastrophic forgetting.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML | scikit-learn, XGBoost, LightGBM, SHAP, Optuna, imbalanced-learn |
| Backend | FastAPI, Pydantic, MLflow |
| Infrastructure | Docker, Docker Compose, Nginx |
| Frontend | React 19, React Router 7, Vite 8, Tailwind CSS 4, Lucide Icons |
| Data Source | CDC BRFSS 2015 Diabetes Health Indicators |
