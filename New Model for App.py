# Databricks notebook source
# MAGIC %pip install --upgrade mlflow[databricks] scikit-learn==1.6.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, functions as F
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
from sklearn.model_selection import train_test_split

spark = SparkSession.builder.getOrCreate()

SILVER_PATH = "/Volumes/main/default/dot_lakehouse/silver"
MODEL_PATH = "/Volumes/main/default/dot_lakehouse/models"

mlflow.set_experiment("/DOT_Transportation_ML")
print(f"sklearn: {__import__('sklearn').__version__}, numpy: {np.__version__}, mlflow: {mlflow.__version__}")

# COMMAND ----------

# ═══ MODEL 1: Incident Severity Classifier ═══
print("=" * 60)
print("MODEL 1: Incident Severity Classifier")
print("=" * 60)

pdf_inc = (
    spark.read.format("delta").load(f"{SILVER_PATH}/traffic_incidents")
    .select(
        "severity", "incident_type", "state_code", "road_condition",
        "weather_condition", "incident_hour", "day_of_week", "is_weekend",
        "vehicles_involved", "severity_score",
    )
    .dropna()
    .withColumn("is_weekend_int", F.col("is_weekend").cast("int"))
    .toPandas()
)

cat_cols = ["incident_type", "state_code", "road_condition", "weather_condition"]
num_cols = ["incident_hour", "day_of_week", "vehicles_involved", "severity_score", "is_weekend_int"]

label_enc = LabelEncoder()
y_inc = label_enc.fit_transform(pdf_inc["severity"])
X_inc = pdf_inc[cat_cols + num_cols]

inc_preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ("num", StandardScaler(), num_cols),
])

severity_model = Pipeline([
    ("preprocessor", inc_preprocessor),
    ("clf", GradientBoostingClassifier(n_estimators=25, max_depth=5, random_state=42)),
])

X_train, X_test, y_train, y_test = train_test_split(X_inc, y_inc, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="incident_severity_gbt"):
    mlflow.log_param("model_type", "GradientBoostingClassifier")
    mlflow.log_param("n_estimators", 25)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("train_rows", len(X_train))
    mlflow.log_param("test_rows", len(X_test))

    severity_model.fit(X_train, y_train)
    predictions = severity_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1_val = sklearn_f1_score(y_test, predictions, average="weighted")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1_val)
    mlflow.sklearn.log_model(severity_model, "incident_severity_model")

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1_val:.4f}")

# Sample prediction
sample = pd.DataFrame([{
    "incident_type": "COLLISION", "state_code": "NC", "road_condition": "Wet",
    "weather_condition": "Rain", "incident_hour": 17, "day_of_week": 5,
    "vehicles_involved": 3, "severity_score": 4, "is_weekend_int": 0,
}])
prediction = severity_model.predict(sample)
print(f"\nSample prediction - Predicted severity class: {prediction[0]}")

# COMMAND ----------

# ═══ MODEL 2: Bridge Risk Score Regressor ═══
print("=" * 60)
print("MODEL 2: Bridge Risk Score Regressor")
print("=" * 60)

pdf_brg = (
    spark.read.format("delta").load(f"{SILVER_PATH}/bridge_inspections")
    .select(
        "risk_score", "bridge_type", "material_type", "owner_type",
        "state_code", "bridge_age_years", "span_length_ft", "deck_width_ft",
        "avg_daily_traffic", "sufficiency_rating", "days_since_inspection",
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

bridge_model = Pipeline([
    ("preprocessor", brg_preprocessor),
    ("regressor", GradientBoostingRegressor(n_estimators=25, max_depth=5, random_state=42)),
])

X_brg_train, X_brg_test, y_brg_train, y_brg_test = train_test_split(
    X_brg, y_brg, test_size=0.2, random_state=42
)

with mlflow.start_run(run_name="bridge_risk_score_gbt"):
    mlflow.log_param("model_type", "GradientBoostingRegressor")
    mlflow.log_param("n_estimators", 25)
    mlflow.log_param("max_depth", 5)

    bridge_model.fit(X_brg_train, y_brg_train)
    brg_preds = bridge_model.predict(X_brg_test)

    rmse = np.sqrt(mean_squared_error(y_brg_test, brg_preds))
    r2 = r2_score(y_brg_test, brg_preds)
    mae = mean_absolute_error(y_brg_test, brg_preds)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    mlflow.sklearn.log_model(bridge_model, "bridge_risk_model")

    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAE:  {mae:.4f}")

# Sample prediction
sample_bridge = pd.DataFrame([{
    "bridge_type": "Steel", "material_type": "Steel", "owner_type": "State",
    "state_code": "NC", "bridge_age_years": 55, "span_length_ft": 120.0,
    "deck_width_ft": 36.0, "avg_daily_traffic": 25000,
    "sufficiency_rating": 42.0, "days_since_inspection": 800,
}])
risk_score = bridge_model.predict(sample_bridge)
print(f"\nSample prediction - Predicted bridge risk score: {risk_score[0]:.1f}")

# COMMAND ----------

# DBTITLE 1,Train Model 3: Pavement Deterioration
# ═══ MODEL 3: Pavement Deterioration Binary Classifier ═══
print("=" * 60)
print("MODEL 3: Pavement Deterioration Binary Classifier")
print("=" * 60)

pdf_pav = (
    spark.read.format("delta").load(f"{SILVER_PATH}/pavement_conditions")
    .select(
        "condition_rating", "pavement_type", "functional_class", "state_code",
        "pavement_age", "iri", "psi", "pcr", "cracking_percent",
        "rutting_in", "aadt", "truck_percent", "distress_index",
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

pav_model = Pipeline([
    ("preprocessor", pav_preprocessor),
    ("clf", RandomForestClassifier(n_estimators=40, max_depth=5, random_state=42)),
])

X_pav_train, X_pav_test, y_pav_train, y_pav_test = train_test_split(
    X_pav, y_pav, test_size=0.2, random_state=42
)

with mlflow.start_run(run_name="pavement_deterioration_rf"):
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 40)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("label_definition", "1=Poor/VeryPoor condition")

    pav_model.fit(X_pav_train, y_pav_train)
    pav_preds = pav_model.predict(X_pav_test)
    pav_proba = pav_model.predict_proba(X_pav_test)[:, 1]

    auc = roc_auc_score(y_pav_test, pav_proba)
    auc_pr = average_precision_score(y_pav_test, pav_proba)
    accuracy = accuracy_score(y_pav_test, pav_preds)

    mlflow.log_metric("auc_roc", auc)
    mlflow.log_metric("auc_pr", auc_pr)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(pav_model, "pavement_deterioration_model")

    print(f"  AUC-ROC:  {auc:.4f}")
    print(f"  AUC-PR:   {auc_pr:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")

print("\n✅ All 3 models trained and logged to MLflow")

# COMMAND ----------

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

# Load silver bridge data and score all bridges
pdf_bridges = (
    spark.read.format("delta")
    .load("/Volumes/main/default/dot_lakehouse/silver/bridge_inspections")
    .select("bridge_id", "bridge_type", "material_type", "owner_type",
            "state_code", "bridge_age_years", "span_length_ft", "deck_width_ft",
            "avg_daily_traffic", "sufficiency_rating", "days_since_inspection")
    .dropna(subset=["sufficiency_rating"])
    .toPandas()
)

pdf_bridges["predicted_risk_score"] = bridge_model.predict(
    pdf_bridges.drop(columns=["bridge_id"])
)

# Show highest risk bridges
display(
    spark.createDataFrame(
        pdf_bridges.nlargest(20, "predicted_risk_score")
    )
)

# COMMAND ----------

# ═══ Batch Evaluation on Full Datasets ═══
mlflow.end_run()  # end any stale active run

# ── 1. Score incident severity model ─────────────────────────────────────
with mlflow.start_run(run_name="eval_incident_severity"):
    pdf_inc_eval = (
        spark.read.format("delta").load(f"{SILVER_PATH}/traffic_incidents")
        .select("incident_id", "severity", "incident_type", "state_code",
                "road_condition", "weather_condition", "incident_hour",
                "day_of_week", "is_weekend", "vehicles_involved", "severity_score")
        .dropna()
        .withColumn("is_weekend_int", F.col("is_weekend").cast("int"))
        .toPandas()
    )
    feature_cols = ["incident_type", "state_code", "road_condition",
                    "weather_condition", "incident_hour", "day_of_week",
                    "vehicles_involved", "severity_score", "is_weekend_int"]
    preds = severity_model.predict(pdf_inc_eval[feature_cols])

    mlflow.log_param("dataset", "silver/traffic_incidents")
    mlflow.log_param("model_type", "GradientBoostingClassifier")
    mlflow.log_metric("records_scored", len(pdf_inc_eval))
    mlflow.log_metric("unique_classes_predicted", len(np.unique(preds)))
    mlflow.set_tag("eval_type", "batch_scoring")
    print(f"✓ Incident severity: scored {len(pdf_inc_eval):,} records")

# ── 2. Score bridge risk model ─────────────────────────────────────────
with mlflow.start_run(run_name="eval_bridge_risk"):
    pdf_brg_eval = (
        spark.read.format("delta").load(f"{SILVER_PATH}/bridge_inspections")
        .select("bridge_id", "bridge_type", "material_type", "owner_type",
                "state_code", "bridge_age_years", "span_length_ft", "deck_width_ft",
                "avg_daily_traffic", "sufficiency_rating", "days_since_inspection",
                "risk_score")
        .dropna(subset=["sufficiency_rating", "risk_score"])
        .toPandas()
    )
    brg_feature_cols = ["bridge_type", "material_type", "owner_type", "state_code",
                        "bridge_age_years", "span_length_ft", "deck_width_ft",
                        "avg_daily_traffic", "sufficiency_rating", "days_since_inspection"]
    pdf_brg_eval["predicted_risk"] = bridge_model.predict(pdf_brg_eval[brg_feature_cols])

    rmse = np.sqrt(mean_squared_error(pdf_brg_eval["risk_score"], pdf_brg_eval["predicted_risk"]))
    r2 = r2_score(pdf_brg_eval["risk_score"], pdf_brg_eval["predicted_risk"])
    mae = mean_absolute_error(pdf_brg_eval["risk_score"], pdf_brg_eval["predicted_risk"])

    mlflow.log_param("dataset", "silver/bridge_inspections")
    mlflow.log_param("model_type", "GradientBoostingRegressor")
    mlflow.log_metric("records_scored", len(pdf_brg_eval))
    mlflow.log_metric("eval_rmse", rmse)
    mlflow.log_metric("eval_r2", r2)
    mlflow.log_metric("eval_mae", mae)
    mlflow.log_metric("mean_predicted_risk", pdf_brg_eval["predicted_risk"].mean())
    mlflow.log_metric("max_predicted_risk", pdf_brg_eval["predicted_risk"].max())
    mlflow.log_metric("pct_high_risk", (pdf_brg_eval["predicted_risk"] >= 70).mean())
    mlflow.set_tag("eval_type", "batch_scoring")
    print(f"✓ Bridge risk: scored {len(pdf_brg_eval):,} records | RMSE={rmse:.2f} | R²={r2:.4f}")

# ── 3. Score pavement deterioration model ─────────────────────────────
with mlflow.start_run(run_name="eval_pavement_deterioration"):
    pdf_pav_eval = (
        spark.read.format("delta").load(f"{SILVER_PATH}/pavement_conditions")
        .select("segment_id", "condition_rating", "pavement_type", "functional_class",
                "state_code", "pavement_age", "iri", "psi", "pcr",
                "cracking_percent", "rutting_in", "aadt", "truck_percent", "distress_index")
        .dropna()
        .toPandas()
    )
    y_true = pdf_pav_eval["condition_rating"].isin(["Poor", "Very Poor"]).astype(float).values
    pav_feature_cols = ["pavement_type", "functional_class", "state_code", "pavement_age",
                        "iri", "psi", "pcr", "cracking_percent", "rutting_in",
                        "aadt", "truck_percent", "distress_index"]
    preds = pav_model.predict(pdf_pav_eval[pav_feature_cols])
    proba = pav_model.predict_proba(pdf_pav_eval[pav_feature_cols])[:, 1]

    acc = accuracy_score(y_true, preds)
    auc = roc_auc_score(y_true, proba)
    auc_pr = average_precision_score(y_true, proba)

    mlflow.log_param("dataset", "silver/pavement_conditions")
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_metric("records_scored", len(pdf_pav_eval))
    mlflow.log_metric("eval_accuracy", acc)
    mlflow.log_metric("eval_auc_roc", auc)
    mlflow.log_metric("eval_auc_pr", auc_pr)
    mlflow.log_metric("pct_predicted_poor", preds.mean())
    mlflow.set_tag("eval_type", "batch_scoring")
    print(f"✓ Pavement: scored {len(pdf_pav_eval):,} records | AUC={auc:.4f} | Accuracy={acc:.4f}")

print("\n✅ All evaluation runs logged to /DOT_Transportation_ML")

# COMMAND ----------

from mlflow.models import infer_signature

mlflow.set_registry_uri("databricks-uc")
mlflow.end_run()  # end any stale run

CATALOG = "main"
SCHEMA = "default"

# Build sample data for signature inference
sample_inc = pd.DataFrame([{
    "incident_type": "COLLISION", "state_code": "NC", "road_condition": "Wet",
    "weather_condition": "Rain", "incident_hour": 17, "day_of_week": 5,
    "vehicles_involved": 3, "severity_score": 4, "is_weekend_int": 0,
}])
sample_brg = pd.DataFrame([{
    "bridge_type": "Steel", "material_type": "Steel", "owner_type": "State",
    "state_code": "NC", "bridge_age_years": 55, "span_length_ft": 120.0,
    "deck_width_ft": 36.0, "avg_daily_traffic": 25000,
    "sufficiency_rating": 42.0, "days_since_inspection": 800,
}])
sample_pav = pd.DataFrame([{
    "pavement_type": "Asphalt", "functional_class": "Interstate",
    "state_code": "NC", "pavement_age": 15, "iri": 120.0, "psi": 2.5,
    "pcr": 55.0, "cracking_percent": 12.0, "rutting_in": 0.3,
    "aadt": 30000, "truck_percent": 15.0, "distress_index": 45.0,
}])

models_to_register = [
    {"run_name": "incident_severity_gbt", "artifact": "incident_severity_model",
     "uc_name": f"{CATALOG}.{SCHEMA}.dot_incident_severity",
     "model_obj": severity_model, "sample": sample_inc},
    {"run_name": "bridge_risk_score_gbt", "artifact": "bridge_risk_model",
     "uc_name": f"{CATALOG}.{SCHEMA}.dot_bridge_risk",
     "model_obj": bridge_model, "sample": sample_brg},
    {"run_name": "pavement_deterioration_rf", "artifact": "pavement_deterioration_model",
     "uc_name": f"{CATALOG}.{SCHEMA}.dot_pavement_deterioration",
     "model_obj": pav_model, "sample": sample_pav},
]

for m in models_to_register:
    sig = infer_signature(m["sample"], m["model_obj"].predict(m["sample"]))

    with mlflow.start_run(run_name=f"{m['run_name']}_uc"):
        mlflow.sklearn.log_model(
            m["model_obj"],
            artifact_path=m["artifact"],
            signature=sig,
            input_example=m["sample"],
            registered_model_name=m["uc_name"],
        )
        print(f"✓ Registered {m['uc_name']}")

print("\n✅ All models registered to Unity Catalog")
