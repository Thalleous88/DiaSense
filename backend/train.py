import os
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, average_precision_score,
    f1_score, recall_score
)
from sklearn.calibration import calibration_curve
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
import shap
import joblib
from datetime import datetime

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "diabetes_binary_health_indicators_BRFSS2015.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

BINARY_COLS = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex"
]
ORDINAL_COLS = ["GenHlth", "Age", "Education", "Income"]
CONTINUOUS_COLS = ["BMI", "MentHlth", "PhysHlth"]

FEATURE_COLS = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
]

print("=" * 70)
print("DiaSense - Full ML Pipeline: EDA > Preprocessing > Ensemble > SHAP")
print("=" * 70)

# =============================================================================
# CELL 2 — Load Data & Overview
# =============================================================================
print("\n[1] Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

print(f"\nMissing values:\n{df.isnull().sum().to_string()}")
print(f"\nTarget distribution:")
target_counts = df["Diabetes_binary"].value_counts()
print(f"  No Diabetes (0): {target_counts[0]:,} ({target_counts[0]/len(df)*100:.1f}%)")
print(f"  Diabetes (1):    {target_counts[1]:,} ({target_counts[1]/len(df)*100:.1f}%)")
print(f"  Imbalance ratio: {target_counts[0]/target_counts[1]:.2f}:1")

print(f"\nDescriptive statistics:\n{df.describe().to_string()}")

# =============================================================================
# CELL 3 — EDA: Target & Feature Distributions
# =============================================================================
print("\n[2] Generating EDA visualizations...")

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
target_counts.plot(kind="bar", color=["#10b981", "#f43f5e"], ax=ax[0])
ax[0].set_title("Target Distribution: Diabetes_binary", fontsize=14, fontweight="bold")
ax[0].set_xlabel("Class")
ax[0].set_ylabel("Count")
ax[0].set_xticklabels(["No Diabetes", "Diabetes"], rotation=0)
for i, v in enumerate(target_counts.values):
    ax[0].text(i, v + 2000, f"{v:,}\n({v/len(df)*100:.1f}%)", ha="center", fontsize=10)

sizes = target_counts.values
labels = [f"No Diabetes\n{sizes[0]:,}", f"Diabetes\n{sizes[1]:,}"]
ax[1].pie(sizes, labels=labels, colors=["#10b981", "#f43f5e"], autopct="%1.1f%%",
          startangle=90, textprops={"fontsize": 11})
ax[1].set_title("Class Proportion", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "01_target_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(CONTINUOUS_COLS):
    for label, color in zip([0, 1], ["#10b981", "#f43f5e"]):
        subset = df[df["Diabetes_binary"] == label][col]
        axes[i].hist(subset, bins=40, alpha=0.6, color=color,
                     label=f"{'No Diabetes' if label == 0 else 'Diabetes'}", density=True)
    axes[i].set_title(f"Distribution of {col}", fontsize=13, fontweight="bold")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Density")
    axes[i].legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "02_continuous_distributions.png"), dpi=150, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(5, 3, figsize=(18, 22))
axes = axes.flatten()
for i, col in enumerate(BINARY_COLS):
    ct = pd.crosstab(df[col], df["Diabetes_binary"], normalize="index")
    ct.plot(kind="bar", color=["#10b981", "#f43f5e"], ax=axes[i], stacked=False)
    axes[i].set_title(col, fontsize=11, fontweight="bold")
    axes[i].set_xlabel("")
    axes[i].set_xticklabels(["No", "Yes"], rotation=0)
    axes[i].legend(["No Diabetes", "Diabetes"], fontsize=8)
if len(BINARY_COLS) < len(axes):
    for j in range(len(BINARY_COLS), len(axes)):
        axes[j].set_visible(False)
plt.suptitle("Binary Feature Distributions by Diabetes Status", fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "03_binary_distributions.png"), dpi=150, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for i, col in enumerate(ORDINAL_COLS):
    ax = axes[i // 2][i % 2]
    ct = pd.crosstab(df[col], df["Diabetes_binary"], normalize="index")
    ct.plot(kind="bar", color=["#10b981", "#f43f5e"], ax=ax, stacked=True)
    ax.set_title(f"{col} vs Diabetes Status", fontsize=13, fontweight="bold")
    ax.set_xlabel(col)
    ax.set_ylabel("Proportion")
    ax.legend(["No Diabetes", "Diabetes"], fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "04_ordinal_distributions.png"), dpi=150, bbox_inches="tight")
plt.close()

# =============================================================================
# CELL 4 — EDA: Correlation & Relationships
# =============================================================================
print("[3] Generating correlation and relationship plots...")

corr = df.corr()
fig, ax = plt.subplots(figsize=(16, 14))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=False, cmap="RdBu_r", center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "05_correlation_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()

diabetes_corr = corr["Diabetes_binary"].drop("Diabetes_binary").sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 8))
diabetes_corr.plot(kind="barh", color=np.where(diabetes_corr > 0, "#f43f5e", "#10b981"), ax=ax)
ax.set_title("Feature Correlation with Diabetes_binary", fontsize=14, fontweight="bold")
ax.set_xlabel("Pearson Correlation")
ax.axvline(x=0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "06_target_correlation.png"), dpi=150, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, col in enumerate(CONTINUOUS_COLS):
    sns.boxplot(x="Diabetes_binary", y=col, data=df, ax=axes[i], palette=["#10b981", "#f43f5e"])
    axes[i].set_title(f"{col} by Diabetes Status", fontsize=13, fontweight="bold")
    axes[i].set_xticklabels(["No Diabetes", "Diabetes"])
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "07_boxplots_continuous.png"), dpi=150, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(5, 3, figsize=(18, 22))
axes = axes.flatten()
for i, col in enumerate(BINARY_COLS):
    prev = df.groupby(col)["Diabetes_binary"].mean() * 100
    prev.plot(kind="bar", color=["#10b981", "#f43f5e"], ax=axes[i])
    axes[i].set_title(f"Diabetes Prevalence by {col}", fontsize=11, fontweight="bold")
    axes[i].set_ylabel("Prevalence (%)")
    axes[i].set_xticklabels(["No", "Yes"], rotation=0)
    for j, v in enumerate(prev.values):
        axes[i].text(j, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)
if len(BINARY_COLS) < len(axes):
    for j in range(len(BINARY_COLS), len(axes)):
        axes[j].set_visible(False)
plt.suptitle("Diabetes Prevalence by Binary Features", fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "08_prevalence_binary.png"), dpi=150, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
age_prev = df.groupby("Age")["Diabetes_binary"].mean() * 100
age_labels = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49",
              "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"]
axes[0].bar(range(len(age_prev)), age_prev.values, color="#0284c7")
axes[0].set_xticks(range(len(age_prev)))
axes[0].set_xticklabels(age_labels, rotation=45, ha="right", fontsize=9)
axes[0].set_title("Diabetes Prevalence by Age Group", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Prevalence (%)")

income_prev = df.groupby("Income")["Diabetes_binary"].mean() * 100
income_labels = ["<$10k", "$10-15k", "$15-20k", "$20-25k", "$25-35k",
                 "$35-50k", "$50-75k", "$75k+"]
axes[1].bar(range(len(income_prev)), income_prev.values, color="#059669")
axes[1].set_xticks(range(len(income_prev)))
axes[1].set_xticklabels(income_labels, rotation=45, ha="right", fontsize=9)
axes[1].set_title("Diabetes Prevalence by Income Level", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Prevalence (%)")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "09_prevalence_age_income.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"  Saved 9 EDA figures to {FIGURES_DIR}/")

# =============================================================================
# CELL 5 — Preprocessing
# =============================================================================
print("\n[4] Preprocessing...")

for col in BINARY_COLS + ORDINAL_COLS:
    df[col] = df[col].astype(np.int8)
for col in CONTINUOUS_COLS:
    df[col] = df[col].astype(np.float32)

n_before = len(df)
df = df.drop_duplicates()
n_after = len(df)
print(f"  Dropped duplicates: {n_before - n_after} rows ({(n_before - n_after)/n_before*100:.2f}%)")

bmi_cap = df["BMI"].quantile(0.99)
n_capped = (df["BMI"] > bmi_cap).sum()
df["BMI"] = df["BMI"].clip(upper=bmi_cap)
print(f"  Capped BMI at 99th percentile ({bmi_cap:.1f}): {n_capped} rows affected")

print("  Engineering new features...")
df["BMI_x_Age"] = (df["BMI"] * df["Age"]).astype(np.float32)
df["GenHlth_x_PhysHlth"] = (df["GenHlth"] * df["PhysHlth"]).astype(np.float32)
df["BMI_x_HighBP"] = (df["BMI"] * df["HighBP"]).astype(np.float32)
df["BMI_category"] = pd.cut(df["BMI"], bins=[0, 18.5, 25, 30, 35, 100],
                            labels=[1, 2, 3, 4, 5]).astype(np.int8)
df["RiskScore_composite"] = (
    df["HighBP"].astype(np.float32) +
    df["HighChol"].astype(np.float32) +
    df["GenHlth"].astype(np.float32) / 5.0 +
    (df["BMI"] > 30).astype(np.float32) +
    (df["Age"] > 7).astype(np.float32)
) / 5.0
df["RiskScore_composite"] = df["RiskScore_composite"].astype(np.float32)

ENGINEERED_COLS = ["BMI_x_Age", "GenHlth_x_PhysHlth", "BMI_x_HighBP", "BMI_category", "RiskScore_composite"]
ALL_FEATURE_COLS = FEATURE_COLS + ENGINEERED_COLS
ALL_CONTINUOUS_COLS = CONTINUOUS_COLS + ["BMI_x_Age", "GenHlth_x_PhysHlth", "BMI_x_HighBP", "RiskScore_composite"]

print(f"  Original features: {len(FEATURE_COLS)}")
print(f"  Engineered features: {len(ENGINEERED_COLS)}")
print(f"  Total features: {len(ALL_FEATURE_COLS)}")
print(f"  Final dataset shape: {df.shape}")

# =============================================================================
# CELL 6 — Train/Val/Test Split + Scaling + Resampling
# =============================================================================
print("\n[5] Splitting data and applying transformations...")

X = df[ALL_FEATURE_COLS]
y = df["Diabetes_binary"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"  Train: {X_train.shape[0]:,} | Val: {X_val.shape[0]:,} | Test: {X_test.shape[0]:,}")

scaler = StandardScaler()
X_train[ALL_CONTINUOUS_COLS] = scaler.fit_transform(X_train[ALL_CONTINUOUS_COLS])
X_val[ALL_CONTINUOUS_COLS] = scaler.transform(X_val[ALL_CONTINUOUS_COLS])
X_test[ALL_CONTINUOUS_COLS] = scaler.transform(X_test[ALL_CONTINUOUS_COLS])
print(f"  Scaled {len(ALL_CONTINUOUS_COLS)} continuous features")

print("  Applying SMOTE to training data...")
print(f"    Before SMOTE: Class 0={int((y_train==0).sum()):,}, Class 1={int((y_train==1).sum()):,}")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"    After SMOTE:  Class 0={int((y_train_res==0).sum()):,}, Class 1={int((y_train_res==1).sum()):,}")

# =============================================================================
# CELL 7 — Base Model Training
# =============================================================================
print("\n[6] Training base models...")

base_models = {
    "XGBoost": XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=3, gamma=0.1, scale_pos_weight=3,
        eval_metric="logloss", random_state=42, n_jobs=-1
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=3, class_weight="balanced",
        random_state=42, n_jobs=-1, verbose=-1
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=400, max_depth=10, min_samples_split=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "LogisticRegression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42, n_jobs=-1
    ),
}

base_results = {}
for name, model in base_models.items():
    print(f"  Training {name}...")
    model.fit(X_train_res, y_train_res)
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    roc_auc = roc_auc_score(y_val, y_prob)
    pr_auc = average_precision_score(y_val, y_prob)
    f1 = f1_score(y_val, y_pred)
    recall1 = recall_score(y_val, y_pred, pos_label=1)
    base_results[name] = {
        "model": model, "roc_auc": roc_auc, "pr_auc": pr_auc,
        "f1": f1, "recall1": recall1, "y_prob": y_prob
    }
    print(f"    ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}  F1={f1:.4f}  Recall(1)={recall1:.4f}")

print("\n  Base Model Comparison:")
print(f"  {'Model':<20} {'ROC-AUC':>8} {'PR-AUC':>8} {'F1':>8} {'Recall(1)':>10}")
print("  " + "-" * 56)
for name, r in base_results.items():
    print(f"  {name:<20} {r['roc_auc']:>8.4f} {r['pr_auc']:>8.4f} {r['f1']:>8.4f} {r['recall1']:>10.4f}")

# =============================================================================
# CELL 8 — Hyperparameter Tuning (Optuna)
# =============================================================================
print("\n[7] Hyperparameter tuning with Optuna...")
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def xgb_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
        "gamma": trial.suggest_float("gamma", 0.0, 0.3),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "scale_pos_weight": 3,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    }
    m = XGBClassifier(**params)
    m.fit(X_train_res, y_train_res, verbose=False)
    y_prob = m.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, y_prob)

def lgbm_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    m = LGBMClassifier(**params)
    m.fit(X_train_res, y_train_res)
    y_prob = m.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, y_prob)

print("  Tuning XGBoost (20 trials)...")
xgb_study = optuna.create_study(direction="maximize")
xgb_study.optimize(xgb_objective, n_trials=20, show_progress_bar=False)
print(f"    Best XGBoost ROC-AUC: {xgb_study.best_value:.4f}")
print(f"    Best params: {xgb_study.best_params}")

print("  Tuning LightGBM (20 trials)...")
lgbm_study = optuna.create_study(direction="maximize")
lgbm_study.optimize(lgbm_objective, n_trials=20, show_progress_bar=False)
print(f"    Best LightGBM ROC-AUC: {lgbm_study.best_value:.4f}")
print(f"    Best params: {lgbm_study.best_params}")

tuned_xgb = XGBClassifier(
    **xgb_study.best_params,
    scale_pos_weight=3, eval_metric="logloss", random_state=42, n_jobs=-1
)
tuned_lgbm = LGBMClassifier(
    **lgbm_study.best_params,
    class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1
)

# =============================================================================
# CELL 9 — Stacking Ensemble
# =============================================================================
print("\n[8] Building stacking ensemble...")

stacking_model = StackingClassifier(
    estimators=[
        ("xgb", tuned_xgb),
        ("lgbm", tuned_lgbm),
        ("rf", base_models["RandomForest"]),
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    passthrough=False,
    n_jobs=-1,
)

print("  Training stacking ensemble...")
stacking_model.fit(X_train_res, y_train_res)
print("  Training complete.")

y_prob_ensemble = stacking_model.predict_proba(X_val)[:, 1]
roc_auc_ensemble = roc_auc_score(y_val, y_prob_ensemble)
pr_auc_ensemble = average_precision_score(y_val, y_prob_ensemble)
print(f"  Ensemble Val ROC-AUC: {roc_auc_ensemble:.4f}")
print(f"  Ensemble Val PR-AUC:  {pr_auc_ensemble:.4f}")

# =============================================================================
# CELL 10 — Threshold Optimization & Final Evaluation
# =============================================================================
print("\n[9] Optimizing decision threshold...")

precision_arr, recall_arr, thresholds_arr = precision_recall_curve(y_val, y_prob_ensemble)
f1_arr = 2 * precision_arr * recall_arr / (precision_arr + recall_arr + 1e-8)
best_f1_idx = np.argmax(f1_arr)
best_f1_threshold = thresholds_arr[best_f1_idx]

j_scores = recall_arr[:-1] - (1 - precision_arr[:-1])
best_youden_idx = np.argmax(j_scores)
youden_threshold = thresholds_arr[best_youden_idx]

print(f"  Best F1 threshold: {best_f1_threshold:.4f} (F1={f1_arr[best_f1_idx]:.4f})")
print(f"  Youden J threshold: {youden_threshold:.4f}")

DECISION_THRESHOLD = youden_threshold
if DECISION_THRESHOLD < 0.20:
    DECISION_THRESHOLD = 0.30
    print(f"  Clamped threshold to {DECISION_THRESHOLD:.2f} (clinical minimum)")

print(f"\n  Final Decision Threshold: {DECISION_THRESHOLD:.4f}")

print("\n[10] Final evaluation on TEST set...")
y_prob_test = stacking_model.predict_proba(X_test)[:, 1]
y_pred_test = (y_prob_test >= DECISION_THRESHOLD).astype(int)

print(f"\n  Classification Report (Test Set, threshold={DECISION_THRESHOLD:.4f}):")
print(classification_report(y_test, y_pred_test, target_names=["No Diabetes", "Diabetes"]))

cm = confusion_matrix(y_test, y_pred_test)
print(f"  Confusion Matrix:")
print(f"    TN={cm[0][0]:,}  FP={cm[0][1]:,}")
print(f"    FN={cm[1][0]:,}  TP={cm[1][1]:,}")

test_roc_auc = roc_auc_score(y_test, y_prob_test)
test_pr_auc = average_precision_score(y_test, y_prob_test)
test_recall1 = recall_score(y_test, y_pred_test, pos_label=1)
test_f1 = f1_score(y_test, y_pred_test)
print(f"\n  Test ROC-AUC:    {test_roc_auc:.4f}")
print(f"  Test PR-AUC:     {test_pr_auc:.4f}")
print(f"  Test F1:         {test_f1:.4f}")
print(f"  Test Recall(1):  {test_recall1:.4f}")

if test_recall1 < 0.70:
    print("  WARNING: Recall below 0.70 clinical threshold!")
else:
    print("  Recall meets clinical threshold (>= 0.70)")

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["No Diabetes", "Diabetes"],
            yticklabels=["No Diabetes", "Diabetes"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix (threshold={DECISION_THRESHOLD:.3f})", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "10_confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(10, 8))
for name, r in base_results.items():
    fpr, tpr, _ = roc_curve(y_val, r["y_prob"])
    ax.plot(fpr, tpr, alpha=0.6, label=f"{name} (AUC={r['roc_auc']:.3f})")
fpr_ens, tpr_ens, _ = roc_curve(y_val, y_prob_ensemble)
ax.plot(fpr_ens, tpr_ens, linewidth=2.5, color="#0284c7",
        label=f"Stacking Ensemble (AUC={roc_auc_ensemble:.3f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "11_roc_curves.png"), dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(10, 8))
for name, r in base_results.items():
    prec, rec, _ = precision_recall_curve(y_val, r["y_prob"])
    ax.plot(rec, prec, alpha=0.6, label=f"{name} (AP={r['pr_auc']:.3f})")
prec_ens, rec_ens, _ = precision_recall_curve(y_val, y_prob_ensemble)
ax.plot(rec_ens, prec_ens, linewidth=2.5, color="#0284c7",
        label=f"Stacking Ensemble (AP={pr_auc_ensemble:.3f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves — Model Comparison", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "12_pr_curves.png"), dpi=150, bbox_inches="tight")
plt.close()

prob_true, prob_pred = calibration_curve(y_val, y_prob_ensemble, n_bins=10)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(prob_pred, prob_true, "o-", color="#0284c7", label="Stacking Ensemble")
ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfectly Calibrated")
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives")
ax.set_title("Calibration Curve", fontsize=14, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "13_calibration_curve.png"), dpi=150, bbox_inches="tight")
plt.close()

# =============================================================================
# CELL 11 — SHAP Explainability
# =============================================================================
print("\n[11] Computing SHAP values...")

rf_for_shap = base_models["RandomForest"]
rf_for_shap.fit(X_train_res, y_train_res)

X_val_sample = X_val.iloc[:500]
explainer = shap.TreeExplainer(rf_for_shap)
shap_values = explainer.shap_values(X_val_sample)

sv = shap_values[1] if isinstance(shap_values, list) else shap_values

fig, ax = plt.subplots(figsize=(12, 8))
shap.summary_plot(sv, X_val_sample, show=False, max_display=20)
plt.title("SHAP Summary Plot (RandomForest)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "14_shap_summary.png"), dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(sv, X_val_sample, plot_type="bar", show=False, max_display=20)
plt.title("SHAP Feature Importance (Mean |SHAP|)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "15_shap_importance.png"), dpi=150, bbox_inches="tight")
plt.close()

mean_abs_shap = np.abs(sv).mean(axis=0)
if mean_abs_shap.ndim > 1:
    mean_abs_shap = mean_abs_shap.mean(axis=1)

feature_importance = {}
for i, col in enumerate(ALL_FEATURE_COLS):
    feature_importance[col] = round(float(mean_abs_shap[i]), 6)

sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
print("  Top 10 features by SHAP importance:")
for feat, val in sorted_importance[:10]:
    print(f"    {feat:<25} {val:.6f}")

top_features_for_api = {feat: round(val * 100 / sorted_importance[0][1], 1)
                        for feat, val in sorted_importance[:10]}

# =============================================================================
# CELL 12 — Save Artifacts
# =============================================================================
print("\n[12] Saving model artifacts...")

MODEL_PATH = os.path.join(BASE_DIR, "ensemble_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
META_PATH = os.path.join(BASE_DIR, "model_meta.pkl")

joblib.dump(stacking_model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

meta = {
    "feature_cols": ALL_FEATURE_COLS,
    "continuous_cols": ALL_CONTINUOUS_COLS,
    "engineered_cols": ENGINEERED_COLS,
    "base_feature_cols": FEATURE_COLS,
    "base_continuous_cols": CONTINUOUS_COLS,
    "decision_threshold": float(DECISION_THRESHOLD),
    "recall_class1": float(test_recall1),
    "roc_auc": float(test_roc_auc),
    "pr_auc": float(test_pr_auc),
    "f1_score": float(test_f1),
    "model_type": "StackingClassifier",
    "base_models": ["XGBClassifier", "LGBMClassifier", "RandomForestClassifier"],
    "training_date": datetime.now().isoformat(),
    "feature_importance": feature_importance,
    "top_features_for_api": top_features_for_api,
    "n_training_samples": len(X_train),
    "n_test_samples": len(X_test),
    "bmi_cap": float(bmi_cap),
    "xgb_best_params": dict(xgb_study.best_params),
    "lgbm_best_params": dict(lgbm_study.best_params),
}
joblib.dump(meta, META_PATH)

print(f"  Model:  {MODEL_PATH}")
print(f"  Scaler: {SCALER_PATH}")
print(f"  Meta:   {META_PATH}")
print(f"\n{'='*70}")
print("Pipeline complete. All figures saved to", FIGURES_DIR)
print(f"{'='*70}")
