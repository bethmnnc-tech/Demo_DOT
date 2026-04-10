# Databricks notebook source
# DBTITLE 1,Install dependencies
# MAGIC %pip install --upgrade mlflow[databricks] scikit-learn==1.6.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Imports and setup
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

# DBTITLE 1,Regenerate pavement bronze data with realistic feature-label relationships
# ═══ Fix: Regenerate pavement bronze data with condition_rating derived from features ═══
# The original generator used random.choice() for condition_rating,
# making it impossible for any model to predict. This version creates
# realistic correlations between pavement metrics and condition.

import random, uuid
from datetime import datetime, timedelta
from pyspark.sql.types import (
    DoubleType, IntegerType, StringType, StructField, StructType, TimestampType,
)

random.seed(42)
BASE_PATH = "/Volumes/main/default/dot_lakehouse"

states = ["NC","VA","SC","TN","GA","FL","TX","CA","NY","IL"]
pavement_types = ["Asphalt","Concrete","Composite","Gravel","Chip Seal"]
functional_classes = ["Interstate","US Route","State Route","County Road","Local Road","Ramp"]

def rand_date(start_year=2019, end_year=2024):
    start = datetime(start_year, 1, 1)
    delta = (datetime(end_year, 12, 31) - start).days
    return start + timedelta(days=random.randint(0, delta))

pavements = []
for i in range(20_000):
    insp_date = rand_date()
    year_constructed = random.randint(1960, 2023)
    pavement_age = 2024 - year_constructed
    state = random.choice(states)
    pav_type = random.choice(pavement_types)
    func_class = random.choice(functional_classes)
    aadt = random.randint(500, 120_000)
    truck_pct = random.randint(0, 50)

    # Generate correlated features based on age + traffic load + noise
    # Older pavements with heavy traffic → worse metrics
    age_factor = min(pavement_age / 60.0, 1.0)  # 0-1 scale
    traffic_factor = min(aadt / 100_000, 1.0) * (1 + truck_pct / 100)
    base_deterioration = 0.5 * age_factor + 0.3 * traffic_factor + 0.2 * random.random()
    base_deterioration = max(0, min(base_deterioration, 1.0))

    # IRI: International Roughness Index (higher = rougher, range 0-400)
    iri = int(40 + base_deterioration * 300 + random.gauss(0, 25))
    iri = max(0, min(iri, 400))

    # PSI: Present Serviceability Index (higher = better, range 0-5)
    psi = round(4.5 - base_deterioration * 4.0 + random.gauss(0, 0.3), 1)
    psi = max(0.0, min(psi, 5.0))

    # PCR: Pavement Condition Rating (higher = better, range 0-100)
    pcr = round(95 - base_deterioration * 80 + random.gauss(0, 8), 1)
    pcr = max(0.0, min(pcr, 100.0))

    # Cracking percent (higher = worse, range 0-30)
    cracking = round(base_deterioration * 25 + random.gauss(0, 3), 1)
    cracking = max(0.0, min(cracking, 30.0))

    # Rutting in inches (higher = worse, range 0-2)
    rutting = round(base_deterioration * 1.5 + random.gauss(0, 0.15), 2)
    rutting = max(0.0, min(rutting, 2.0))

    # Derive condition_rating from a composite score of ALL features
    composite = (
        0.25 * (iri / 400) +          # higher IRI → worse
        0.20 * (1 - psi / 5.0) +      # lower PSI → worse
        0.20 * (1 - pcr / 100) +      # lower PCR → worse
        0.15 * (cracking / 30) +      # higher cracking → worse
        0.10 * (rutting / 2.0) +      # higher rutting → worse
        0.10 * age_factor              # older → worse
    )
    # Add noise so it's learnable but not trivially deterministic
    composite += random.gauss(0, 0.06)
    composite = max(0, min(composite, 1.0))

    if composite < 0.30:
        condition = "Good"
    elif composite < 0.50:
        condition = "Fair"
    elif composite < 0.70:
        condition = "Poor"
    else:
        condition = "Very Poor"

    pavements.append((
        f"SEG-{i+1:07d}",
        state,
        f"COUNTY_{random.randint(1,100):03d}",
        f"ROUTE_{random.randint(1,500)}",
        round(random.uniform(0, 100), 2),   # begin_milepost
        round(random.uniform(0, 5), 2),     # segment_length_mi
        func_class,
        pav_type,
        year_constructed,
        iri, psi, pcr, cracking, rutting,
        condition,
        aadt, truck_pct,
        insp_date,
        datetime.now(),
    ))

pavement_schema = StructType([
    StructField("segment_id",         StringType(),    False),
    StructField("state_code",         StringType(),    True),
    StructField("county_code",        StringType(),    True),
    StructField("route_id",           StringType(),    True),
    StructField("begin_milepost",     DoubleType(),    True),
    StructField("segment_length_mi",  DoubleType(),    True),
    StructField("functional_class",   StringType(),    True),
    StructField("pavement_type",      StringType(),    True),
    StructField("year_constructed",   IntegerType(),   True),
    StructField("iri",                IntegerType(),   True),
    StructField("psi",                DoubleType(),    True),
    StructField("pcr",                DoubleType(),    True),
    StructField("cracking_percent",   DoubleType(),    True),
    StructField("rutting_in",         DoubleType(),    True),
    StructField("condition_rating",   StringType(),    True),
    StructField("aadt",               IntegerType(),   True),
    StructField("truck_percent",      IntegerType(),   True),
    StructField("inspection_date",    TimestampType(), True),
    StructField("ingestion_timestamp",TimestampType(), False),
])

df_pavements = spark.createDataFrame(pavements, schema=pavement_schema)
df_pavements.write.format("delta").mode("overwrite").option("overwriteSchema","true") \
    .partitionBy("state_code","functional_class") \
    .save(f"{BASE_PATH}/bronze/pavement_conditions")

# Show distribution
print(f"✓ Regenerated bronze pavement_conditions → {df_pavements.count():,} rows")
df_pavements.groupBy("condition_rating").count().orderBy("condition_rating").show()

# COMMAND ----------

# DBTITLE 1,Re-run silver transformation for pavement
# ═══ Re-run Silver Transformation for Pavement ═══
from pyspark.sql import functions as F

BASE_PATH = "/Volumes/main/default/dot_lakehouse"
BRONZE_PATH = f"{BASE_PATH}/bronze"
SILVER_PATH = f"{BASE_PATH}/silver"

df_pav_raw = spark.read.format("delta").load(f"{BRONZE_PATH}/pavement_conditions")

df_pav_silver = (
    df_pav_raw
    .withColumn("state_code", F.upper(F.col("state_code")))
    .withColumn("pavement_age", F.lit(2024) - F.col("year_constructed"))
    .withColumn("iri_category",
        F.when(F.col("iri") <= 60,  "Very Good")
         .when(F.col("iri") <= 95,  "Good")
         .when(F.col("iri") <= 170, "Fair")
         .when(F.col("iri") <= 220, "Mediocre")
         .otherwise("Poor"))
    .withColumn("distress_index",
        F.round(F.col("cracking_percent") * 0.4 + F.col("rutting_in") * 5 +
                (100 - F.col("pcr")) * 0.6, 1))
    .withColumn("maintenance_priority",
        F.when(F.col("distress_index") >= 60, "Immediate Rehab")
         .when(F.col("distress_index") >= 40, "Preventive Treatment")
         .when(F.col("distress_index") >= 20, "Monitor")
         .otherwise("No Action"))
    .withColumn("traffic_category",
        F.when(F.col("aadt") >= 50000, "High Volume")
         .when(F.col("aadt") >= 10000, "Medium Volume")
         .otherwise("Low Volume"))
    .dropDuplicates(["segment_id"])
    .withColumn("silver_timestamp", F.current_timestamp())
)

df_pav_silver.write.format("delta").mode("overwrite").option("overwriteSchema","true") \
    .partitionBy("state_code","functional_class") \
    .save(f"{SILVER_PATH}/pavement_conditions")

# Also update the registered table
spark.sql("CREATE DATABASE IF NOT EXISTS main.dot_silver")
spark.sql(f"""
    CREATE OR REPLACE TABLE main.dot_silver.pavement_conditions
    AS SELECT * FROM delta.`{SILVER_PATH}/pavement_conditions`
""")

print(f"✓ Silver pavement_conditions → {df_pav_silver.count():,} rows")
df_pav_silver.groupBy("condition_rating").count().orderBy("condition_rating").show()

# COMMAND ----------

# DBTITLE 1,Train Model 1: Incident Severity
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

# DBTITLE 1,Train Model 2: Bridge Risk Score
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
print(f"  Class balance: {y_pav.mean():.1%} positive (Poor/VeryPoor)")

pav_cat_cols = ["pavement_type", "functional_class", "state_code"]
pav_num_cols = ["pavement_age", "iri", "psi", "pcr", "cracking_percent",
                "rutting_in", "aadt", "truck_percent", "distress_index"]

X_pav = pdf_pav[pav_cat_cols + pav_num_cols]

pav_preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), pav_cat_cols),
    ("num", StandardScaler(), pav_num_cols),
])

# Upgraded: GradientBoostingClassifier with stronger hyperparameters
pav_model = Pipeline([
    ("preprocessor", pav_preprocessor),
    ("clf", GradientBoostingClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        min_samples_split=10,
        subsample=0.8,
        random_state=42,
    )),
])

X_pav_train, X_pav_test, y_pav_train, y_pav_test = train_test_split(
    X_pav, y_pav, test_size=0.2, random_state=42
)

with mlflow.start_run(run_name="pavement_deterioration_gbt"):
    mlflow.log_param("model_type", "GradientBoostingClassifier")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 8)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("subsample", 0.8)
    mlflow.log_param("label_definition", "1=Poor/VeryPoor condition")

    pav_model.fit(X_pav_train, y_pav_train)
    pav_preds = pav_model.predict(X_pav_test)
    pav_proba = pav_model.predict_proba(X_pav_test)[:, 1]

    auc = roc_auc_score(y_pav_test, pav_proba)
    auc_pr = average_precision_score(y_pav_test, pav_proba)
    accuracy = accuracy_score(y_pav_test, pav_preds)
    f1 = sklearn_f1_score(y_pav_test, pav_preds)

    mlflow.log_metric("auc_roc", auc)
    mlflow.log_metric("auc_pr", auc_pr)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(pav_model, "pavement_deterioration_model")

    print(f"  AUC-ROC:  {auc:.4f}")
    print(f"  AUC-PR:   {auc_pr:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")

# Feature importance
gbt_stage = pav_model.named_steps["clf"]
feature_names = pav_model.named_steps["preprocessor"].get_feature_names_out()
importances = gbt_stage.feature_importances_
fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
fi_df = fi_df.sort_values("importance", ascending=False).head(10)
print("\n  Top 10 Feature Importances:")
print(fi_df.to_string(index=False))

print("\n✅ All 3 models trained and logged to MLflow")

# COMMAND ----------

# DBTITLE 1,Score all bridges
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

# DBTITLE 1,Batch evaluation on full datasets
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

# DBTITLE 1,Register models to Unity Catalog
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
