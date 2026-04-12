# =============================================================================
# NOTEBOOK 4: DOT Predictive Analytics – ML Models
# Databricks Notebook | Language: Python | MLflow + scikit-learn
# Description: Three ML models:
#   1. Incident severity classifier
#   2. Bridge risk score regressor
#   3. Pavement deterioration forecast
#
# Uses scikit-learn instead of pyspark.ml to avoid the Spark Connect
# ML model cache size limit (1 GB) on Serverless compute.
# =============================================================================

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import joblib
import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score as sklearn_f1_score,
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split, cross_val_score

import sys

# ── Configuration ────────────────────────────────────────────────────────────
# When run by a job: parameters arrive via sys.argv
# When run interactively: dbutils widgets provide a UI with dev defaults
if len(sys.argv) >= 3 and not sys.argv[1].startswith("-"):
    BASE_PATH = sys.argv[1]
    CATALOG   = sys.argv[2]
else:
    dbutils.widgets.text("base_path", "/Volumes/main/default/dot_lakehouse")
    dbutils.widgets.text("catalog", "main")
    BASE_PATH = dbutils.widgets.get("base_path")
    CATALOG   = dbutils.widgets.get("catalog")

print(f"  BASE_PATH = {BASE_PATH}")
print(f"  CATALOG   = {CATALOG}")

spark = SparkSession.builder.appName("DOT_MLModels").getOrCreate()

SILVER_PATH = f"{BASE_PATH}/silver"
MODEL_PATH  = f"{BASE_PATH}/models"

CV_FOLDS = 5

mlflow.set_experiment("/DOT_Transportation_ML")

# ═════════════════════════════════════════════════════════════════════════════
# MODEL 1 – Incident Severity Classifier
# Goal: Predict severity class from contextual features at time of incident
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("MODEL 1: Incident Severity Classifier")
print("=" * 60)

pdf_inc = (
    spark.read.format("delta").load(f"{SILVER_PATH}/traffic_incidents")
    .select(
        "severity",            # target
        "incident_type",
        "state_code",
        "road_condition",
        "weather_condition",
        "incident_hour",
        "day_of_week",
        "is_weekend",
        "vehicles_involved",
        "severity_score",
    )
    .dropna()  
    .withColumn("is_weekend_int", F.col("is_weekend").cast("int"))
    .toPandas()
)

cat_cols = ["incident_type", "state_code", "road_condition", "weather_condition"]
num_cols = ["incident_hour", "day_of_week", "vehicles_involved", "severity_score", "is_weekend"]

label_enc = LabelEncoder()
y_inc = label_enc.fit_transform(pdf_inc["severity"])
X_inc = pdf_inc[cat_cols + num_cols]

inc_preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ("num", StandardScaler(), num_cols),
])

inc_pipeline = Pipeline([
    ("preprocessor", inc_preprocessor),
    ("clf", GradientBoostingClassifier(n_estimators=150, max_depth=8, learning_rate=0.05, subsample=0.8, min_samples_leaf=10, random_state=42)),
])

X_train, X_test, y_train, y_test = train_test_split(X_inc, y_inc, test_size=0.2, random_state=42)

# Cross-validation on training set
cv_scores_inc = cross_val_score(inc_pipeline, X_train, y_train, cv=CV_FOLDS, scoring="accuracy", n_jobs=-1)
print(f"  CV Accuracy ({CV_FOLDS}-fold): {cv_scores_inc.mean():.4f} ± {cv_scores_inc.std():.4f}")

with mlflow.start_run(run_name="incident_severity_gbt"):
    mlflow.log_param("model_type", "GradientBoostingClassifier")
    mlflow.log_param("n_estimators", 150)
    mlflow.log_param("max_depth",    8)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_param("subsample",    0.8)
    mlflow.log_param("min_samples_leaf", 10)
    mlflow.log_param("cv_folds",     CV_FOLDS)
    mlflow.log_param("train_rows", len(X_train))
    mlflow.log_param("test_rows",  len(X_test))

    mlflow.log_metric("cv_accuracy_mean", cv_scores_inc.mean())
    mlflow.log_metric("cv_accuracy_std",  cv_scores_inc.std())

    inc_pipeline.fit(X_train, y_train)
    predictions = inc_pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1_val   = sklearn_f1_score(y_test, predictions, average="weighted")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1_val)
    mlflow.sklearn.log_model(inc_pipeline, "incident_severity_model")

    os.makedirs(f"{MODEL_PATH}/incident_severity_gbt", exist_ok=True)
    joblib.dump(inc_pipeline, f"{MODEL_PATH}/incident_severity_gbt/model.joblib")

    print(f"  Holdout Accuracy: {accuracy:.4f}")
    print(f"  Holdout F1 Score: {f1_val:.4f}")
    print(f"  Model saved to: {MODEL_PATH}/incident_severity_gbt")

del inc_pipeline, predictions

# ═════════════════════════════════════════════════════════════════════════════
# MODEL 2 – Bridge Risk Score Regressor
# Goal: Predict composite risk_score from structural / traffic features
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODEL 2: Bridge Risk Score Regressor")
print("=" * 60)

pdf_brg = (
    spark.read.format("delta").load(f"{SILVER_PATH}/bridge_inspections")
    .select(
        "risk_score",          # target
        "bridge_type",
        "material_type",
        "owner_type",
        "state_code",
        "bridge_age_years",
        "span_length_ft",
        "deck_width_ft",
        "avg_daily_traffic",
        "sufficiency_rating",
        "days_since_inspection",
    )
    .dropna(subset=["risk_score"])
    .toPandas()
)

brg_cat_cols = ["bridge_type", "material_type", "owner_type", "state_code"]
brg_num_cols = ["bridge_age_years", "span_length_ft", "deck_width_ft",
                "avg_daily_traffic", "sufficiency_rating", "days_since_inspection"]

y_brg = pdf_brg["risk_score"].values
X_brg = pdf_brg[brg_cat_cols + brg_num_cols]

brg_preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), brg_cat_cols),
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), brg_num_cols),
])

brg_pipeline = Pipeline([
    ("preprocessor", brg_preprocessor),
    ("regressor", GradientBoostingRegressor(n_estimators=150, max_depth=8, learning_rate=0.05, subsample=0.8, min_samples_leaf=10, random_state=42)),
])

X_brg_train, X_brg_test, y_brg_train, y_brg_test = train_test_split(
    X_brg, y_brg, test_size=0.2, random_state=42
)

# Cross-validation on training set
cv_scores_brg = cross_val_score(brg_pipeline, X_brg_train, y_brg_train, cv=CV_FOLDS, scoring="r2", n_jobs=-1)
print(f"  CV R² ({CV_FOLDS}-fold): {cv_scores_brg.mean():.4f} ± {cv_scores_brg.std():.4f}")

with mlflow.start_run(run_name="bridge_risk_score_gbt"):
    mlflow.log_param("model_type", "GradientBoostingRegressor")
    mlflow.log_param("n_estimators", 150)
    mlflow.log_param("max_depth",    8)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_param("subsample",    0.8)
    mlflow.log_param("min_samples_leaf", 10)
    mlflow.log_param("cv_folds",     CV_FOLDS)

    mlflow.log_metric("cv_r2_mean", cv_scores_brg.mean())
    mlflow.log_metric("cv_r2_std",  cv_scores_brg.std())

    brg_pipeline.fit(X_brg_train, y_brg_train)
    brg_preds = brg_pipeline.predict(X_brg_test)

    rmse = np.sqrt(mean_squared_error(y_brg_test, brg_preds))
    r2   = r2_score(y_brg_test, brg_preds)
    mae  = mean_absolute_error(y_brg_test, brg_preds)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2",   r2)
    mlflow.log_metric("mae",  mae)
    mlflow.sklearn.log_model(brg_pipeline, "bridge_risk_model")

    os.makedirs(f"{MODEL_PATH}/bridge_risk_gbt", exist_ok=True)
    joblib.dump(brg_pipeline, f"{MODEL_PATH}/bridge_risk_gbt/model.joblib")

    print(f"  Holdout RMSE: {rmse:.4f}")
    print(f"  Holdout R²:   {r2:.4f}")
    print(f"  Holdout MAE:  {mae:.4f}")

del brg_pipeline, brg_preds

# ═════════════════════════════════════════════════════════════════════════════
# MODEL 3 – Pavement Condition Deterioration Forecast
# Goal: Binary classifier – will pavement condition degrade to Poor within
#       2 years?  (based on current metrics + traffic load)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODEL 3: Pavement Deterioration Binary Classifier")
print("=" * 60)

pdf_pav = (
    spark.read.format("delta").load(f"{SILVER_PATH}/pavement_conditions")
    .select(
        "condition_rating",
        "pavement_type",
        "functional_class",
        "state_code",
        "pavement_age",
        "iri",
        "psi",
        "pcr",
        "cracking_percent",
        "rutting_in",
        "aadt",
        "truck_percent",
        "distress_index",
    )
    .dropna()
    .toPandas()
)

# Label: 1 if condition is Poor/Very Poor (high deterioration risk)
y_pav = pdf_pav["condition_rating"].isin(["Poor", "Very Poor"]).astype(float).values

pav_cat_cols = ["pavement_type", "functional_class", "state_code"]
pav_num_cols = ["pavement_age", "iri", "psi", "pcr", "cracking_percent",
                "rutting_in", "aadt", "truck_percent", "distress_index"]

X_pav = pdf_pav[pav_cat_cols + pav_num_cols]

pav_preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), pav_cat_cols),
    ("num", StandardScaler(), pav_num_cols),
])

pav_pipeline = Pipeline([
    ("preprocessor", pav_preprocessor),
    ("clf", RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42)),
])

X_pav_train, X_pav_test, y_pav_train, y_pav_test = train_test_split(
    X_pav, y_pav, test_size=0.2, random_state=42
)

# Cross-validation on training set
cv_scores_pav = cross_val_score(pav_pipeline, X_pav_train, y_pav_train, cv=CV_FOLDS, scoring="roc_auc", n_jobs=-1)
print(f"  CV AUC-ROC ({CV_FOLDS}-fold): {cv_scores_pav.mean():.4f} ± {cv_scores_pav.std():.4f}")

with mlflow.start_run(run_name="pavement_deterioration_rf"):
    mlflow.log_param("model_type",  "RandomForestClassifier")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth",    10)
    mlflow.log_param("min_samples_leaf", 5)
    mlflow.log_param("cv_folds",     CV_FOLDS)
    mlflow.log_param("label_definition", "1=Poor/VeryPoor condition")

    mlflow.log_metric("cv_auc_roc_mean", cv_scores_pav.mean())
    mlflow.log_metric("cv_auc_roc_std",  cv_scores_pav.std())

    pav_pipeline.fit(X_pav_train, y_pav_train)
    pav_preds = pav_pipeline.predict(X_pav_test)
    pav_proba = pav_pipeline.predict_proba(X_pav_test)[:, 1]

    auc      = roc_auc_score(y_pav_test, pav_proba)
    auc_pr   = average_precision_score(y_pav_test, pav_proba)
    accuracy = accuracy_score(y_pav_test, pav_preds)

    mlflow.log_metric("auc_roc",  auc)
    mlflow.log_metric("auc_pr",   auc_pr)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(pav_pipeline, "pavement_deterioration_model")

    os.makedirs(f"{MODEL_PATH}/pavement_deterioration_rf", exist_ok=True)
    joblib.dump(pav_pipeline, f"{MODEL_PATH}/pavement_deterioration_rf/model.joblib")

    print(f"  Holdout AUC-ROC:  {auc:.4f}")
    print(f"  Holdout AUC-PR:   {auc_pr:.4f}")
    print(f"  Holdout Accuracy: {accuracy:.4f}")

# ── Feature Importance (Pavement model) ─────────────────────────────────────
rf_stage      = pav_pipeline.named_steps["clf"]
feature_names = pav_pipeline.named_steps["preprocessor"].get_feature_names_out()
importances   = rf_stage.feature_importances_

fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
fi_df = fi_df.sort_values("importance", ascending=False).head(10)
print("\n  Top 10 Feature Importances (Pavement Deterioration):")
print(fi_df.to_string(index=False))

del pav_pipeline, pav_preds

print("\n✅  All ML models trained and logged to MLflow.")
print(f"   Model artifacts: {MODEL_PATH}/")
