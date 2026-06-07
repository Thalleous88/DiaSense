import os
import threading
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, recall_score, f1_score,
    precision_recall_curve
)
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_active_model = None
_active_scaler = None
_active_meta = None
_training_lock = threading.Lock()
_training_status = {
    "status": "idle",
    "progress": 0,
    "started_at": None,
    "message": ""
}

FEATURE_COLS = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
]

BINARY_COLS = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex"
]

ORDINAL_COLS = ["GenHlth", "Age", "Education", "Income"]
BASE_CONTINUOUS_COLS = ["BMI", "MentHlth", "PhysHlth"]

ENGINEERED_COLS = ["BMI_x_Age", "GenHlth_x_PhysHlth", "BMI_x_HighBP", "BMI_category", "RiskScore_composite"]

REQUIRED_CSV_COLUMNS = set(FEATURE_COLS + ["Diabetes_binary"])


def load_active_model():
    global _active_model, _active_scaler, _active_meta
    model_path = os.path.join(BASE_DIR, "ensemble_model.pkl")
    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
    meta_path = os.path.join(BASE_DIR, "model_meta.pkl")

    if not os.path.exists(model_path):
        model_path = os.path.join(BASE_DIR, "xgb_diabetes_model.pkl")

    _active_model = joblib.load(model_path)
    _active_scaler = joblib.load(scaler_path)
    _active_meta = joblib.load(meta_path)


def get_active_model():
    return _active_model, _active_scaler, _active_meta


def get_training_status():
    return _training_status.copy()


def validate_csv(file_path: str) -> dict:
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return {"valid": False, "errors": [f"Could not read CSV: {str(e)}"], "n_rows": 0}

    errors = []
    csv_cols = set(df.columns.str.strip())

    missing = REQUIRED_CSV_COLUMNS - csv_cols
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")

    extra = csv_cols - REQUIRED_CSV_COLUMNS
    if extra:
        pass

    n_rows = len(df)
    if n_rows == 0:
        errors.append("CSV file contains no data rows")

    common_cols = sorted(REQUIRED_CSV_COLUMNS & csv_cols)
    for col in common_cols:
        if df[col].isnull().any():
            errors.append(f"Column '{col}' contains {df[col].isnull().sum()} null values")

    n_valid_rows = 0
    if not errors and common_cols:
        n_valid_rows = int(n_rows - df[common_cols].isnull().any(axis=1).sum())

    return {
        "valid": len(errors) == 0,
        "errors": errors if errors else None,
        "n_rows": int(n_rows),
        "n_valid_rows": n_valid_rows
    }


def _engineer_features(df):
    df["BMI_x_Age"] = (df["BMI"] * df["Age"]).astype(np.float32)
    df["GenHlth_x_PhysHlth"] = (df["GenHlth"] * df["PhysHlth"]).astype(np.float32)
    df["BMI_x_HighBP"] = (df["BMI"] * df["HighBP"]).astype(np.float32)
    df["BMI_category"] = pd.cut(
        df["BMI"],
        bins=[0, 18.5, 25, 30, 35, 100],
        labels=[1, 2, 3, 4, 5]
    ).astype(np.int8)
    df["RiskScore_composite"] = (
        df["HighBP"].astype(np.float32) +
        df["HighChol"].astype(np.float32) +
        df["GenHlth"].astype(np.float32) / 5.0 +
        (df["BMI"] > 30).astype(np.float32) +
        (df["Age"] > 7).astype(np.float32)
    ) / 5.0
    df["RiskScore_composite"] = df["RiskScore_composite"].astype(np.float32)
    return df


def _run_training(new_csv_path: str, training_run_id: int, db_url: str):
    global _active_model, _active_scaler, _active_meta, _training_status

    _training_status["status"] = "running"
    _training_status["progress"] = 10
    _training_status["started_at"] = datetime.utcnow().isoformat()
    _training_status["message"] = "Loading datasets..."

    try:
        import sqlalchemy
        from sqlalchemy import text
        engine = sqlalchemy.create_engine(db_url.replace("+asyncpg", ""))

        original_path = os.path.join(BASE_DIR, "diabetes_binary_health_indicators_BRFSS2015.csv")
        df_original = pd.read_csv(original_path)

        _training_status["progress"] = 20
        _training_status["message"] = "Merging new data..."

        df_new = pd.read_csv(new_csv_path)
        df_new.columns = df_new.columns.str.strip()
        df = pd.concat([df_original, df_new], ignore_index=True)
        df = df.drop_duplicates()

        n_new_samples = len(df_new)
        n_total_samples = len(df)

        _training_status["progress"] = 30
        _training_status["message"] = "Preprocessing..."

        for col in BINARY_COLS + ORDINAL_COLS:
            if col in df.columns:
                df[col] = df[col].astype(np.int8)
        for col in BASE_CONTINUOUS_COLS:
            if col in df.columns:
                df[col] = df[col].astype(np.float32)

        bmi_cap = df["BMI"].quantile(0.99)
        df["BMI"] = df["BMI"].clip(upper=bmi_cap)

        df = _engineer_features(df)

        ALL_FEATURE_COLS = FEATURE_COLS + ENGINEERED_COLS
        ALL_CONTINUOUS_COLS = BASE_CONTINUOUS_COLS + ["BMI_x_Age", "GenHlth_x_PhysHlth", "BMI_x_HighBP", "RiskScore_composite"]

        X = df[ALL_FEATURE_COLS]
        y = df["Diabetes_binary"]

        _training_status["progress"] = 40
        _training_status["message"] = "Splitting data..."

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.40, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
        )

        scaler = StandardScaler()
        X_train[ALL_CONTINUOUS_COLS] = scaler.fit_transform(X_train[ALL_CONTINUOUS_COLS])
        X_val[ALL_CONTINUOUS_COLS] = scaler.transform(X_val[ALL_CONTINUOUS_COLS])
        X_test[ALL_CONTINUOUS_COLS] = scaler.transform(X_test[ALL_CONTINUOUS_COLS])

        _training_status["progress"] = 50
        _training_status["message"] = "Applying SMOTE..."

        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        _training_status["progress"] = 60
        _training_status["message"] = "Training base models..."

        tuned_xgb = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            gamma=0.1, scale_pos_weight=3, eval_metric="logloss",
            random_state=42, n_jobs=-1
        )

        tuned_lgbm = LGBMClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1
        )

        rf = RandomForestClassifier(
            n_estimators=400, max_depth=10, min_samples_split=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        )

        _training_status["progress"] = 70
        _training_status["message"] = "Training stacking ensemble..."

        stacking_model = StackingClassifier(
            estimators=[
                ("xgb", tuned_xgb),
                ("lgbm", tuned_lgbm),
                ("rf", rf),
            ],
            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            passthrough=False,
            n_jobs=-1,
        )

        stacking_model.fit(X_train_res, y_train_res)

        _training_status["progress"] = 85
        _training_status["message"] = "Evaluating model..."

        y_prob_val = stacking_model.predict_proba(X_val)[:, 1]
        precision_arr, recall_arr, thresholds_arr = precision_recall_curve(y_val, y_prob_val)
        j_scores = recall_arr[:-1] - (1 - precision_arr[:-1])
        best_youden_idx = np.argmax(j_scores)
        decision_threshold = float(thresholds_arr[best_youden_idx])
        if decision_threshold < 0.20:
            decision_threshold = 0.15

        y_prob_test = stacking_model.predict_proba(X_test)[:, 1]
        y_pred_test = (y_prob_test >= decision_threshold).astype(int)

        test_roc_auc = float(roc_auc_score(y_test, y_prob_test))
        test_pr_auc = float(average_precision_score(y_test, y_prob_test))
        test_recall1 = float(recall_score(y_test, y_pred_test, pos_label=1))
        test_f1 = float(f1_score(y_test, y_pred_test))

        import shap
        X_val_sample = X_val.iloc[:200]
        try:
            rf_from_stack = stacking_model.named_estimators_["rf"]
            explainer = shap.TreeExplainer(rf_from_stack)
            shap_values_raw = explainer.shap_values(X_val_sample)
            if isinstance(shap_values_raw, list):
                sv = np.array(shap_values_raw[1], dtype=np.float64)
            else:
                sv = np.array(shap_values_raw, dtype=np.float64)
            mean_abs_shap = np.abs(sv).mean(axis=0)
            if mean_abs_shap.ndim > 1:
                mean_abs_shap = mean_abs_shap.mean(axis=1)
        except Exception:
            rf_standalone = RandomForestClassifier(
                n_estimators=200, max_depth=10,
                class_weight="balanced", random_state=42, n_jobs=-1
            )
            rf_standalone.fit(X_train_res, y_train_res)
            explainer = shap.TreeExplainer(rf_standalone)
            shap_values_raw = explainer.shap_values(X_val_sample)
            if isinstance(shap_values_raw, list):
                sv = np.array(shap_values_raw[1], dtype=np.float64)
            else:
                sv = np.array(shap_values_raw, dtype=np.float64)
            mean_abs_shap = np.abs(sv).mean(axis=0)
            if mean_abs_shap.ndim > 1:
                mean_abs_shap = mean_abs_shap.mean(axis=1)

        feature_importance = {}
        for i, col in enumerate(ALL_FEATURE_COLS):
            feature_importance[col] = round(float(mean_abs_shap[i]), 6)
        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features_for_api = {feat: round(val * 100 / sorted_importance[0][1], 1)
                                for feat, val in sorted_importance[:10]}

        _training_status["progress"] = 95
        _training_status["message"] = "Saving model artifacts..."

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(BASE_DIR, f"ensemble_model_v{timestamp}.pkl")
        scaler_path = os.path.join(BASE_DIR, f"scaler_v{timestamp}.pkl")
        meta_path = os.path.join(BASE_DIR, f"model_meta_v{timestamp}.pkl")

        joblib.dump(stacking_model, model_path)
        joblib.dump(scaler, scaler_path)

        meta = {
            "feature_cols": ALL_FEATURE_COLS,
            "continuous_cols": ALL_CONTINUOUS_COLS,
            "engineered_cols": ENGINEERED_COLS,
            "base_feature_cols": FEATURE_COLS,
            "base_continuous_cols": BASE_CONTINUOUS_COLS,
            "decision_threshold": decision_threshold,
            "recall_class1": test_recall1,
            "roc_auc": test_roc_auc,
            "pr_auc": test_pr_auc,
            "f1_score": test_f1,
            "model_type": "StackingClassifier",
            "base_models": ["XGBClassifier", "LGBMClassifier", "RandomForestClassifier"],
            "training_date": datetime.now().isoformat(),
            "feature_importance": feature_importance,
            "top_features_for_api": top_features_for_api,
            "n_training_samples": len(X_train),
            "n_test_samples": len(X_test),
            "bmi_cap": float(bmi_cap),
        }
        joblib.dump(meta, meta_path)

        _active_model = stacking_model
        _active_scaler = scaler
        _active_meta = meta

        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE training_runs
                SET status = 'completed',
                    n_new_samples = :n_new,
                    n_total_samples = :n_total,
                    roc_auc = :roc_auc,
                    recall_class1 = :recall1,
                    pr_auc = :pr_auc,
                    f1_score = :f1,
                    decision_threshold = :thr,
                    model_type = 'StackingClassifier',
                    completed_at = NOW()
                WHERE id = :rid
            """), {
                "n_new": n_new_samples,
                "n_total": n_total_samples,
                "roc_auc": test_roc_auc,
                "recall1": test_recall1,
                "pr_auc": test_pr_auc,
                "f1": test_f1,
                "thr": decision_threshold,
                "rid": training_run_id,
            })

        _training_status["status"] = "completed"
        _training_status["progress"] = 100
        _training_status["message"] = (
            f"Training complete. ROC-AUC={test_roc_auc:.4f}, "
            f"Recall(1)={test_recall1:.4f}, Threshold={decision_threshold:.4f}"
        )

    except Exception as e:
        _training_status["status"] = "failed"
        _training_status["message"] = str(e)

        try:
            import sqlalchemy
            from sqlalchemy import text
            eng = sqlalchemy.create_engine(db_url.replace("+asyncpg", ""))
            with eng.begin() as conn:
                conn.execute(text("""
                    UPDATE training_runs
                    SET status = 'failed', error_message = :err, completed_at = NOW()
                    WHERE id = :rid
                """), {"err": str(e), "rid": training_run_id})
        except Exception:
            pass


def start_background_training(new_csv_path: str, training_run_id: int, db_url: str):
    if _training_status["status"] == "running":
        raise RuntimeError("Training already in progress")

    thread = threading.Thread(
        target=_run_training,
        args=(new_csv_path, training_run_id, db_url),
        daemon=True
    )
    thread.start()
