# Databricks notebook source
# MAGIC %pip install --upgrade mlflow[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import pandas as pd

mlflow.set_experiment("/DOT_Transportation_ML")

# Find the latest run for each model
runs = mlflow.search_runs(
    experiment_ids=["947763301386294"],
    order_by=["start_time DESC"],
)
display(runs[["run_id", "tags.mlflow.runName", "start_time",
              "metrics.accuracy", "metrics.f1_score",
              "metrics.rmse", "metrics.r2",
              "metrics.auc_roc"]])

# COMMAND ----------

# Get the run ID for the incident severity model
severity_run = runs[runs["tags.mlflow.runName"] == "incident_severity_gbt"].iloc[0]
model_uri = f"runs:/{severity_run.run_id}/incident_severity_model"
severity_model = mlflow.sklearn.load_model(model_uri)

# Sample prediction
sample = pd.DataFrame([{
    "incident_type": "COLLISION",
    "state_code": "NC",
    "road_condition": "Wet",
    "weather_condition": "Rain",
    "incident_hour": 17,
    "day_of_week": 5,
    "vehicles_involved": 3,
    "severity_score": 4,
    "is_weekend_int": 0,
}])

prediction = severity_model.predict(sample)
print(f"Predicted severity class: {prediction[0]}")

# COMMAND ----------

bridge_run = runs[runs["tags.mlflow.runName"] == "bridge_risk_score_gbt"].iloc[0]
bridge_model = mlflow.sklearn.load_model(f"runs:/{bridge_run.run_id}/bridge_risk_model")

sample_bridge = pd.DataFrame([{
    "bridge_type": "Steel",
    "material_type": "Steel",
    "owner_type": "State",
    "state_code": "NC",
    "bridge_age_years": 55,
    "span_length_ft": 120.0,
    "deck_width_ft": 36.0,
    "avg_daily_traffic": 25000,
    "sufficiency_rating": 42.0,
    "days_since_inspection": 800,
}])

risk_score = bridge_model.predict(sample_bridge)
print(f"Predicted bridge risk score: {risk_score[0]:.1f}")

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

import numpy as np
from pyspark.sql import SparkSession, functions as F
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, roc_auc_score, average_precision_score

spark = SparkSession.builder.getOrCreate()
SILVER_PATH = "/Volumes/main/default/dot_lakehouse/silver"

# End any stale active run from a previous failed execution
mlflow.end_run()

# ── 1. Score & log incident severity model ───────────────────────────────────
with mlflow.start_run(run_name="eval_incident_severity"):
    pdf_inc = (
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
    preds = severity_model.predict(pdf_inc[feature_cols])

    mlflow.log_param("model_source_run", severity_run.run_id)
    mlflow.log_param("dataset", "silver/traffic_incidents")
    mlflow.log_param("model_type", "GradientBoostingClassifier")
    mlflow.log_metric("records_scored", len(pdf_inc))
    mlflow.log_metric("unique_classes_predicted", len(np.unique(preds)))
    mlflow.set_tag("eval_type", "batch_scoring")
    print(f"✓ Incident severity: scored {len(pdf_inc):,} records")

# ── 2. Score & log bridge risk model ─────────────────────────────────────────
with mlflow.start_run(run_name="eval_bridge_risk"):
    pdf_brg = (
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
    pdf_brg["predicted_risk"] = bridge_model.predict(pdf_brg[brg_feature_cols])

    rmse = np.sqrt(mean_squared_error(pdf_brg["risk_score"], pdf_brg["predicted_risk"]))
    r2 = r2_score(pdf_brg["risk_score"], pdf_brg["predicted_risk"])
    mae = mean_absolute_error(pdf_brg["risk_score"], pdf_brg["predicted_risk"])

    mlflow.log_param("model_source_run", bridge_run.run_id)
    mlflow.log_param("dataset", "silver/bridge_inspections")
    mlflow.log_param("model_type", "GradientBoostingRegressor")
    mlflow.log_metric("records_scored", len(pdf_brg))
    mlflow.log_metric("eval_rmse", rmse)
    mlflow.log_metric("eval_r2", r2)
    mlflow.log_metric("eval_mae", mae)
    mlflow.log_metric("mean_predicted_risk", pdf_brg["predicted_risk"].mean())
    mlflow.log_metric("max_predicted_risk", pdf_brg["predicted_risk"].max())
    mlflow.log_metric("pct_high_risk", (pdf_brg["predicted_risk"] >= 70).mean())
    mlflow.set_tag("eval_type", "batch_scoring")
    print(f"✓ Bridge risk: scored {len(pdf_brg):,} records | RMSE={rmse:.2f} | R²={r2:.4f}")

# ── 3. Score & log pavement deterioration model ──────────────────────────────
pav_run = runs[runs["tags.mlflow.runName"] == "pavement_deterioration_rf"].iloc[0]
pav_model = mlflow.sklearn.load_model(f"runs:/{pav_run.run_id}/pavement_deterioration_model")

with mlflow.start_run(run_name="eval_pavement_deterioration"):
    pdf_pav = (
        spark.read.format("delta").load(f"{SILVER_PATH}/pavement_conditions")
        .select("segment_id", "condition_rating", "pavement_type", "functional_class",
                "state_code", "pavement_age", "iri", "psi", "pcr",
                "cracking_percent", "rutting_in", "aadt", "truck_percent", "distress_index")
        .dropna()
        .toPandas()
    )
    y_true = pdf_pav["condition_rating"].isin(["Poor", "Very Poor"]).astype(float).values
    pav_feature_cols = ["pavement_type", "functional_class", "state_code", "pavement_age",
                        "iri", "psi", "pcr", "cracking_percent", "rutting_in",
                        "aadt", "truck_percent", "distress_index"]
    preds = pav_model.predict(pdf_pav[pav_feature_cols])
    proba = pav_model.predict_proba(pdf_pav[pav_feature_cols])[:, 1]

    acc = accuracy_score(y_true, preds)
    auc = roc_auc_score(y_true, proba)
    auc_pr = average_precision_score(y_true, proba)

    mlflow.log_param("model_source_run", pav_run.run_id)
    mlflow.log_param("dataset", "silver/pavement_conditions")
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_metric("records_scored", len(pdf_pav))
    mlflow.log_metric("eval_accuracy", acc)
    mlflow.log_metric("eval_auc_roc", auc)
    mlflow.log_metric("eval_auc_pr", auc_pr)
    mlflow.log_metric("pct_predicted_poor", preds.mean())
    mlflow.set_tag("eval_type", "batch_scoring")
    print(f"✓ Pavement: scored {len(pdf_pav):,} records | AUC={auc:.4f} | Accuracy={acc:.4f}")

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
     "model_obj": severity_model, "sample": sample_inc, "flavor": "sklearn"},
    {"run_name": "bridge_risk_score_gbt", "artifact": "bridge_risk_model",
     "uc_name": f"{CATALOG}.{SCHEMA}.dot_bridge_risk",
     "model_obj": bridge_model, "sample": sample_brg, "flavor": "sklearn"},
    {"run_name": "pavement_deterioration_rf", "artifact": "pavement_deterioration_model",
     "uc_name": f"{CATALOG}.{SCHEMA}.dot_pavement_deterioration",
     "model_obj": pav_model, "sample": sample_pav, "flavor": "sklearn"},
]

for m in models_to_register:
    match = runs[runs["tags.mlflow.runName"] == m["run_name"]]
    if match.empty:
        print(f"\u26a0 No run found for {m['run_name']}, skipping")
        continue

    sig = infer_signature(m["sample"], m["model_obj"].predict(m["sample"]))

    # Re-log the model with a signature so it can be registered to UC
    with mlflow.start_run(run_name=f"{m['run_name']}_uc"):
        mlflow.sklearn.log_model(
            m["model_obj"],
            artifact_path=m["artifact"],
            signature=sig,
            input_example=m["sample"],
            registered_model_name=m["uc_name"],
        )
        print(f"\u2713 Registered {m['uc_name']}")

print("\n\u2705 All models registered to Unity Catalog")
