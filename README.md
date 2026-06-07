# DiaSense

AI-powered diabetes risk prediction system with an ensemble ML model, real-time FastAPI backend, and React frontend.

## Architecture

```
DiaSense/
├── backend/                  # FastAPI + ML pipeline
│   ├── main.py               # API endpoints
│   ├── train.py              # Full ML pipeline (EDA, training, SHAP)
│   ├── train_service.py      # Background training + model hot-swap
│   ├── config.py             # Pydantic settings (from .env)
│   ├── database.py           # Async SQLAlchemy engine + sessions
│   ├── models.py             # ORM models (Assessment, TrainingRun, UploadedDataset)
│   ├── requirements.txt      # Python dependencies
│   ├── .env                  # Environment variables (not committed)
│   ├── figures/              # EDA visualizations (15 PNGs, generated)
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
│   │   │   ├── RecentAssessments.jsx  # Latest predictions
│   │   │   ├── MetricCard.jsx        # Stats metric cards
│   │   │   ├── UploadArea.jsx        # Drag-and-drop CSV upload
│   │   │   ├── TrainingStatus.jsx    # Real-time training progress
│   │   │   ├── TrainingHistory.jsx   # Past training runs table
│   │   │   └── ModelInfo.jsx         # Current model metrics
│   │   ├── hooks/
│   │   │   └── useApi.js             # API fetch wrapper (/api proxy)
│   │   └── lib/
│   │       └── utils.js              # cn() utility
│   └── vite.config.js        # Dev server with /api proxy
└── docker-compose.yml        # PostgreSQL 16 service
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

### Performance

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.80 |
| Recall (Class 1) | 0.72 |
| PR-AUC | 0.41 |
| F1 Score | 0.45 |

### SHAP Explainability

Feature importance is computed using SHAP TreeExplainer on the RandomForest base model within the ensemble, evaluated on a 200-row validation sample.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Service info |
| `GET` | `/health` | Health check + model status |
| `POST` | `/predict` | Generate diabetes risk prediction |
| `GET` | `/stats` | Screening statistics |
| `GET` | `/feature-importance` | SHAP feature importance |
| `GET` | `/model/info` | Current model metrics and metadata |
| `GET` | `/assessments/recent` | Recent risk assessments |
| `POST` | `/train/upload` | Upload CSV dataset for training |
| `POST` | `/train/start` | Start background model training |
| `GET` | `/train/status` | Current training progress |
| `GET` | `/train/history` | Past training runs |

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

- Python 3.10+
- Node.js 18+
- Docker & Docker Compose

### 1. Start PostgreSQL

```bash
docker compose up -d
```

This starts PostgreSQL 16 on port **5433** (to avoid conflicts with local PG installations).

### 2. Backend Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

Create a `.env` file in `backend/`:

```
DATABASE_URL=postgresql+asyncpg://diasense:diasense@localhost:5433/diasense
```

Run the ML pipeline to generate model artifacts:

```bash
python train.py
```

This produces:
- `ensemble_model.pkl` — trained stacking ensemble
- `scaler.pkl` — fitted StandardScaler
- `model_meta.pkl` — threshold, metrics, feature lists
- `figures/` — 15 EDA visualizations

Start the API server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 3. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The Vite dev server runs on `http://localhost:5173` and proxies `/api/*` requests to the backend at `http://localhost:8000`.

### 4. Access the App

Open `http://localhost:5173` in your browser.

## Model Hot-Swap

When new training is triggered via the Training page:

1. User uploads a CSV via `POST /train/upload`
2. User clicks "Start Training" → `POST /train/start`
3. Training runs in a **background thread** — predictions continue using the current model
4. On completion, the new model, scaler, and metadata are **atomically swapped** into memory
5. Subsequent predictions use the new model immediately

The original BRFSS2015 dataset is always merged with uploaded data before retraining to prevent catastrophic forgetting.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML | scikit-learn, XGBoost, LightGBM, SHAP, Optuna, imbalanced-learn |
| Backend | FastAPI, SQLAlchemy (async), asyncpg, Pydantic |
| Database | PostgreSQL 16 (Docker) |
| Frontend | React 19, React Router 7, Vite 8, Tailwind CSS 4, Lucide Icons |
| Data Source | CDC BRFSS 2015 Diabetes Health Indicators |

## License

MIT
