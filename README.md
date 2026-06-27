# DiaSense

AI-powered diabetes risk prediction system with an ensemble ML model, real-time FastAPI backend, and React frontend. Designed for high availability, scalability, and stateless execution via Docker.

## Architecture

```
DiaSense/
├── backend/                      # FastAPI + ML pipeline (Python 3.11)
│   ├── main.py                   # API endpoints (DiaSense API v2.0.0)
│   ├── train.py                  # Full offline ML pipeline (EDA, Optuna, training, SHAP)
│   ├── train_service.py          # Background online retraining + model hot-swap
│   ├── config.py                 # Pydantic settings (env-based)
│   ├── patch_ipynb.py            # Script to sync train.ipynb with train.py
│   ├── train.ipynb               # Jupyter notebook version of training pipeline
│   ├── entrypoint.sh             # Container startup: trains model if missing, then runs uvicorn
│   ├── .env                      # Environment variables
│   ├── requirements.txt          # Python dependencies (17 packages)
│   ├── Dockerfile                # python:3.11-slim + uvicorn
│   ├── diabetes_binary_health_indicators_BRFSS2015.csv  # CDC BRFSS 2015 dataset
│   ├── ensemble_model.pkl        # Serialized stacking ensemble (active model)
│   ├── xgb_diabetes_model.pkl    # Fallback XGBoost model
│   ├── scaler.pkl                # Fitted StandardScaler
│   ├── model_meta.pkl            # Metadata (features, threshold, metrics, SHAP)
│   ├── figures/                  # 16 EDA / training visualizations (generated)
│   └── uploads/                  # User-uploaded CSV datasets (generated)
├── frontend/                     # React 19 + Vite 8 + Tailwind CSS 4
│   ├── public/
│   │   ├── favicon.svg
│   │   └── icons.svg
│   ├── src/
│   │   ├── main.jsx              # Entry point
│   │   ├── App.jsx               # BrowserRouter + Routes setup
│   │   ├── index.css             # Tailwind v4 theme, glass-card, animations
│   │   ├── assets/
│   │   │   ├── hero.png
│   │   │   ├── react.svg
│   │   │   └── vite.svg
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx          # `/` — risk assessment form + results
│   │   │   └── TrainingPage.jsx       # `/training` — CSV upload, training controls
│   │   ├── components/
│   │   │   ├── Header.jsx             # Sticky navbar with nav links
│   │   │   ├── HealthForm.jsx         # 20-field patient survey (checkboxes, sliders, selects)
│   │   │   ├── RiskGauge.jsx          # SVG animated donut gauge
│   │   │   ├── ResultPanel.jsx        # Risk level banner + clinical recommendations
│   │   │   ├── FeatureBreakdown.jsx   # SHAP horizontal bar chart (CSS)
│   │   │   ├── MetricCard.jsx         # Reusable metric display card
│   │   │   ├── RecentAssessments.jsx  # Recent prediction history list
│   │   │   ├── UploadArea.jsx         # Drag-and-drop CSV upload
│   │   │   ├── TrainingStatus.jsx     # Real-time training progress + progress bar
│   │   │   ├── TrainingHistory.jsx    # Past training runs table
│   │   │   └── ModelInfo.jsx          # Current model metrics & metadata
│   │   ├── hooks/
│   │   │   └── useApi.js             # apiFetch() wrapper (prefixed with /api)
│   │   └── lib/
│   │       └── utils.js              # cn() — clsx + tailwind-merge
│   ├── eslint.config.js         # ESLint flat config (react-hooks, react-refresh)
│   ├── Dockerfile               # Multi-stage: node:20-alpine build → nginx:alpine
│   ├── nginx.conf               # SPA fallback + /api proxy → backend:8000
│   └── vite.config.js           # Vite proxy /api → localhost:8000 (dev)
├── docker-compose.yml           # Orchestrates MLflow, Backend, Frontend
├── WORKFLOW.md                  # End-to-end data pipeline + model design notes
└── .gitignore
```

## ML Pipeline

The model is a **StackingClassifier** (`StackingClassifier`) with 3 base learners and a logistic regression meta-learner, trained with 5-fold StratifiedKFold cross-validation:

| Layer | Model | Key Hyperparameters |
|-------|-------|-------------------|
| Base | XGBClassifier | n_estimators=400, max_depth=5, lr=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, gamma=0.1, eval_metric=logloss |
| Base | LGBMClassifier | n_estimators=400, max_depth=5, lr=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=3 |
| Base | RandomForestClassifier | n_estimators=400, max_depth=10, min_samples_split=5, class_weight=balanced |
| Meta | LogisticRegression | max_iter=1000 |

**Additional config:** `passthrough=False`, `n_jobs=-1` (parallel training).

### Hyperparameter Tuning (Offline)

`train.py` runs **Optuna** Bayesian optimization for XGBoost and LightGBM (20 trials each, F2-beta objective). The online retraining service (`train_service.py`) uses fixed hyperparameters for speed.

### Preprocessing

1. **Type casting** — features downcast to `np.int8`/`np.float32`
2. **Deduplication** — remove duplicate rows
3. **BMI capping** — clipped at 99th percentile
4. **5 engineered features:**

   | Feature | Formula |
   |---------|--------|
   | `BMI_x_Age` | BMI × Age (interaction) |
   | `GenHlth_x_PhysHlth` | GenHlth × PhysHlth (interaction) |
   | `BMI_x_HighBP` | BMI × HighBP (interaction) |
   | `BMI_category` | BMI binned into 5 ordinal categories |
   | `RiskScore_composite` | Weighted sum of key binary risk indicators |

5. **StandardScaler** — applied to 9 continuous/numeric features (BMI, MentHlth, PhysHlth + 4 engineered continuous)
6. **Stratified split** — 60% train, 20% validation, 20% test
7. **SMOTE** — synthetic oversampling applied to training set only (addresses ~86/14 class imbalance)
8. **Decision threshold optimization** — clinical recall enforcement: finds highest precision threshold achieving recall ≥ 0.70 on validation set; falls back to **Youden's J statistic** (maximize TPR − FPR)

### Performance (Baseline)

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.80 |
| Recall (Class 1) | 0.72 |
| PR-AUC | 0.41 |
| F1 Score | 0.45 |

> Metrics may improve with user-uploaded retraining data via online learning.

### SHAP Explainability

Feature importance is computed using `shap.TreeExplainer` on the RandomForest base model within the ensemble, evaluated on a 500-sample validation subset. Top 10 features by mean \|SHAP\| are served via the API.

### Fallback Model

If the ensemble model is unavailable at startup, the API falls back to a standalone `xgb_diabetes_model.pkl` for inference.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Service info (name, status, version) |
| `GET` | `/health` | Health check + model status |
| `POST` | `/predict` | Generate diabetes risk prediction |
| `GET` | `/feature-importance` | SHAP feature importance (top 10) |
| `GET` | `/model/info` | Current model metrics and metadata |
| `POST` | `/train/upload` | Upload CSV dataset for retraining |
| `POST` | `/train/start` | Start background model training |
| `GET` | `/train/status` | Current training progress (polling) |
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

### Risk Level Mapping

| Risk Score | Level |
|------------|-------|
| < 0.20 | Low |
| 0.20 – 0.39 | Moderate |
| 0.40 – 0.59 | High |
| ≥ 0.60 | Very High |

### Feature Importance Response

```json
{
  "importance": {
    "BMI": 0.152,
    "GenHlth": 0.128,
    "Age": 0.094,
    ...
  },
  "top_features": {
    "BMI": 0.152,
    "GenHlth": 0.128,
    "Age": 0.094
  }
}
```

### Model Info Response

```json
{
  "model_type": "StackingClassifier",
  "base_models": ["XGBClassifier", "LGBMClassifier", "RandomForestClassifier"],
  "n_features": 26,
  "decision_threshold": 0.15,
  "roc_auc": 0.80,
  "recall_class1": 0.72,
  "pr_auc": 0.41,
  "f1_score": 0.45,
  "training_date": "2025-06-01T12:00:00",
  "n_training_samples": 253680
}
```

## Frontend Overview

### Pages

| Route | Component | Description |
|-------|-----------|-------------|
| `/` | `Dashboard` | Patient risk assessment: health form (20 fields), animated risk gauge, result panel with recommendations |
| `/training` | `TrainingPage` | CSV upload, start retraining, training progress, model info, training history table |

### Key Components

| Component | Description |
|-----------|-------------|
| `Header` | Sticky navbar with logo and route links (active state via `NavLink`) |
| `HealthForm` | 20-field form: core metrics (BMI, Age, Sex), health status (GenHlth slider, MentHlth, PhysHlth), risk factor checkboxes, socioeconomic selects |
| `RiskGauge` | SVG circular donut gauge with animated arc (1000ms), color-coded by risk level |
| `ResultPanel` | Post-prediction banner with clinical recommendations and review flag |
| `FeatureBreakdown` | CSS-based horizontal bar chart of SHAP feature importance |
| `UploadArea` | Drag-and-drop CSV upload with row validation and error reporting |
| `TrainingStatus` | Status card with animated progress bar (idle/running/completed/failed) |
| `TrainingHistory` | Table of past training runs fetched from MLflow |
| `ModelInfo` | Key-value display of current model metrics |

### Development Proxy

In development, `vite.config.js` proxies `/api` requests to `http://localhost:8000` with automatic path rewrite. In production, `nginx.conf` handles the same proxy to `http://backend:8000`.

## Getting Started

### Prerequisites

- Docker & Docker Compose

### 1. Build and Run the Stack

```bash
docker compose up -d --build
```

This starts three services:

| Service | URL |
|---------|-----|
| **MLflow Tracking Server** | http://localhost:5000 |
| **FastAPI Backend** | http://localhost:8000 |
| **React Frontend** | http://localhost:80 |

### 2. Initial Model Generation

If `ensemble_model.pkl` does not exist, the container will automatically run `train.py` at startup. To run it manually:

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt

python train.py
```

This generates 16 EDA visualizations in `figures/` and creates model artifacts: `ensemble_model.pkl`, `xgb_diabetes_model.pkl`, `scaler.pkl`, and `model_meta.pkl`.

### 3. Development Setup (Frontend Standalone)

```bash
cd frontend
npm install
npm run dev          # Vite dev server at http://localhost:5173
```

The dev server proxies `/api` → `http://localhost:8000` so you can run the backend separately.

## Model Hot-Swap & Online Learning

When new training is triggered via the frontend Training page:

1. User uploads a CSV via `POST /train/upload` (validates columns + row counts)
2. User clicks "Start Training" → `POST /train/start?filename_key=...`
3. Training runs in a **background daemon thread** (capped at `n_jobs=2` to prevent API starvation)
4. All run parameters and metrics are logged to MLflow (experiment: `DiaSense_Online_Learning`)
5. On completion, the new model, scaler, and metadata are **atomically swapped** into the live API process memory
6. Subsequent predictions use the new model immediately without requiring an API restart

The original BRFSS2015 dataset is always merged with uploaded data before retraining to prevent catastrophic forgetting.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `localhost:5000` | MLflow server address |
| `ALLOWED_ORIGINS` | `*` | CORS allowed origins |
| `UPLOAD_DIR` | `uploads` | Directory for uploaded CSVs |
| `MAX_UPLOAD_SIZE_MB` | `50` | Maximum CSV upload size |

## EDA Figures

The training pipeline generates 16 visualization PNGs in `backend/figures/`:

| # | File | Description |
|---|------|-------------|
| 1 | `01_target_distribution.png` | Target variable bar + pie chart |
| 2 | `02_continuous_distributions.png` | BMI, MentHlth, PhysHlth histograms |
| 3 | `03_binary_distributions.png` | 14 binary features vs. diabetes |
| 4 | `04_ordinal_distributions.png` | GenHlth, Age, Education, Income |
| 5 | `05_correlation_heatmap.png` | Full feature correlation matrix |
| 6 | `06_target_correlation.png` | Correlation with target variable |
| 7 | `07_boxplots_continuous.png` | Box plots of continuous features |
| 8 | `08_prevalence_binary.png` | Diabetes prevalence by binary features |
| 9 | `08b_base_model_metrics.png` | Base model performance comparison |
| 10 | `09_prevalence_age_income.png` | Prevalence by age group and income |
| 11 | `10_confusion_matrix.png` | Test set confusion matrix |
| 12 | `11_roc_curves.png` | ROC curves (all models + ensemble) |
| 13 | `12_pr_curves.png` | Precision-Recall curves |
| 14 | `13_calibration_curve.png` | Probability calibration curve |
| 15 | `14_shap_summary.png` | SHAP beeswarm summary plot |
| 16 | `15_shap_importance.png` | SHAP bar plot feature importance |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML | scikit-learn, XGBoost, LightGBM, SHAP, Optuna, imbalanced-learn |
| Backend | Python 3.11, FastAPI, Pydantic, MLflow |
| Infrastructure | Docker, Docker Compose, Nginx (alpine) |
| Frontend | React 19.2.4, React Router 7.17.0, Vite 8.0.1, Tailwind CSS 4.2.2, Lucide Icons 1.16.0 |
| Data Source | CDC BRFSS 2015 Diabetes Health Indicators (253,680 rows, 21 features) |
