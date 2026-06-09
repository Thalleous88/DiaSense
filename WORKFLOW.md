# DiaSense Workflow Documentation

This document explains the complete, end-to-end workflow of the DiaSense application, detailing how data is processed, how models are trained and selected, and how predictions are served to the user in a stateless production environment.

## 1. Overview

DiaSense is a microservice-based architecture composed of three core components:
1. **Frontend**: A React.js Single Page Application (SPA) providing a user interface for patient screening and model training.
2. **Backend**: A FastAPI Python application handling prediction logic, dataset validation, feature engineering, and background training orchestration.
3. **MLflow Server**: A dedicated tracking server used for versioning machine learning models, logging hyperparameters, and recording evaluation metrics.

The architecture is entirely **stateless**. Historical patient assessments are not saved to a persistent database. Instead, the focus is entirely on real-time inference and active model lifecycle management.

## 2. Data Ingestion & Preprocessing

The foundation of DiaSense is built upon the **CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015** dataset.

When the application prepares data for training (either offline via `train.py` or online via `train_service.py`), the following preprocessing pipeline is executed:

1. **Type Conversion**: Variables are strictly typed into `np.int8` for binary/ordinal features and `np.float32` for continuous features to minimize memory footprint.
2. **Deduplication & Outlier Handling**: Duplicate rows are dropped. BMI values are capped at the 99th percentile to prevent extreme outliers from skewing model weights.
3. **Feature Engineering**: Five new features are dynamically created to capture complex, non-linear health interactions:
   - `BMI_x_Age`: Interaction between physical mass and age.
   - `GenHlth_x_PhysHlth`: Interaction between self-reported general and physical health.
   - `BMI_x_HighBP`: Interaction between BMI and Blood Pressure.
   - `BMI_category`: Binned categorization of BMI (Underweight to Obese).
   - `RiskScore_composite`: A heuristic composite scoring algorithm combining multiple high-risk indicators into a single continuous feature.

## 3. Base Model Training & Optimization

DiaSense uses a **Stacking Ensemble** approach, leveraging three distinct, powerful gradient-boosting and decision-tree algorithms.

Before training, the data is split into Train (60%), Validation (20%), and Test (20%) sets. Because the BRFSS dataset suffers from extreme class imbalance (~86% negative, ~14% positive), we apply **SMOTE** (Synthetic Minority Over-sampling Technique) exclusively to the *training* set. This ensures the models learn the minority class without leaking synthetic data into the evaluation sets.

### Base Learners
1. **XGBoost (`XGBClassifier`)**: Highly optimized gradient boosting framework known for its execution speed and model performance on tabular data.
2. **LightGBM (`LGBMClassifier`)**: A gradient boosting framework that uses tree-based learning algorithms, particularly effective for handling large datasets with lower memory usage.
3. **Random Forest (`RandomForestClassifier`)**: An ensemble learning method that builds multiple decision trees, providing robust resistance to overfitting and serving as the primary model for SHAP interpretability extraction.

### Hyperparameter Tuning
We use **Optuna** to perform Bayesian optimization on the XGBoost and LightGBM models. Optuna intelligently navigates the hyperparameter search space (learning rate, max depth, subsample, etc.) over multiple trials, optimizing for the highest **ROC-AUC** score.

## 4. Stacking Ensemble & Explainability

Once the base models are trained, their predictions are fed into a **Meta-Learner**—in this case, a `LogisticRegression` model.

### The Stacking Logic
The `StackingClassifier` uses cross-validation (5-fold) to train the meta-learner on the output probabilities of the base models. This allows the final ensemble to learn *when* to trust XGBoost over LightGBM, or vice-versa, resulting in a highly calibrated final prediction score.

### Decision Threshold Optimization
By default, ML models predict a positive class if the probability is >= `0.50`. However, in a medical context, a False Negative (missing a diabetic patient) is significantly more dangerous than a False Positive.
Therefore, DiaSense dynamically calculates the **Youden J-statistic** on the Precision-Recall curve to find the optimal decision threshold that maximizes **Recall** (Sensitivity). This threshold is clamped to a minimum of `0.30` to avoid excessive false alarms, but generally hovers significantly below 0.50.

### SHAP (SHapley Additive exPlanations)
To prevent the ensemble from acting as a "black box," DiaSense utilizes SHAP. After training, SHAP values are computed against the Random Forest base model. This generates a definitive list of feature importances, allowing the API to return the "Top Contributing Factors" for every single prediction, giving clinicians interpretable insights into *why* a specific score was generated.

## 5. Deployment & Prediction API

In production, the application is deployed using `docker-compose`.

When a patient survey is submitted via the React frontend:
1. **Request Reception**: The FastAPI backend receives the 21-feature JSON payload at `POST /predict`.
2. **In-Memory Transformation**: The backend mirrors the exact preprocessing pipeline (calculating the 5 engineered features and applying the fitted `StandardScaler`).
3. **Inference**: The in-memory `ensemble_model.pkl` processes the transformed vector, outputting a continuous risk score (0.0 to 1.0).
4. **Threshold Evaluation**: The score is compared against the pre-calculated `decision_threshold`. If it exceeds the threshold, the `flagged_for_review` boolean is set to `True`.
5. **Response**: The API instantly returns the risk percentage, risk level, and flagging status to the frontend. No data is saved to a disk or database.

## 6. Online Learning & Model Hot-Swapping

DiaSense supports real-time retraining capabilities, allowing the model to adapt as new data is collected, without requiring server downtime.

### The Retraining Lifecycle
1. **Upload**: An administrator uploads a new CSV dataset via the frontend (`POST /train/upload`). The backend validates the CSV against required headers and data types.
2. **Dispatch**: The administrator clicks "Start Training" (`POST /train/start`). The backend spawns a detached background thread to execute the training pipeline.
3. **Resource Protection**: To prevent the CPU-intensive XGBoost/LightGBM algorithms from locking up the server and causing API timeouts, `n_jobs` is strictly capped at `2`. The FastAPI server remains fully responsive to incoming predictions using the *current* model.
4. **Data Merging**: The uploaded CSV is securely merged with the original base BRFSS dataset. This ensures the model benefits from new data while preventing "catastrophic forgetting" of the original broad population baselines.
5. **MLflow Tracking**: Throughout the background training, parameters (thresholds, model types) and metrics (ROC-AUC, F1, Recall) are logged to the localized `mlflow` container instance.
6. **Atomic Swap**: Upon successful completion, the newly trained `ensemble_model.pkl`, `scaler.pkl`, and `model_meta.pkl` are saved to disk and immediately hot-swapped into the running memory of the FastAPI application.
7. **Immediate Effect**: The very next `POST /predict` request will utilize the newly trained model parameters.
