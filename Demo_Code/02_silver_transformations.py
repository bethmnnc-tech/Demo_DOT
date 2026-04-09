# =============================================================================
# NOTEBOOK 2: DOT Silver Layer – Data Quality & Transformations
# Databricks Notebook | Language: Python + SQL
# Description: Reads Bronze Delta tables, applies data quality rules,
#              enriches records, and writes clean Silver tables.
# =============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from delta.tables import DeltaTable

spark = SparkSession.builder.appName("main.dot_silverTransformations").getOrCreate()

BASE_PATH    = "/Volumes/main/default/dot_lakehouse"
BRONZE_PATH  = f"{BASE_PATH}/bronze"
SILVER_PATH  = f"{BASE_PATH}/silver"

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Data Quality Checker
# ─────────────────────────────────────────────────────────────────────────────
def run_dq_checks(df, domain: str, checks: list[dict]) -> None:
    """
    Lightweight DQ framework.  Each check dict has:
        name   : str   – human label
        expr   : str   – SQL expression that evaluates to a count of BAD rows
        threshold: float – max allowed failure rate (0-1)
    """
    total = df.count()
    print(f"\n── DQ Report: {domain} (total rows = {total:,}) ──")
    for chk in checks:
        bad_count = df.filter(F.expr(chk["expr"])).count()
        rate      = bad_count / total if total > 0 else 0
        status    = "✅ PASS" if rate <= chk["threshold"] else "❌ FAIL"
        print(f"  {status} | {chk['name']:<40} | bad={bad_count:,} ({rate:.2%}) | threshold={chk['threshold']:.0%}")

# ─────────────────────────────────────────────────────────────────────────────
# SILVER 1 – TRAFFIC INCIDENTS
# ─────────────────────────────────────────────────────────────────────────────
df_inc_raw = spark.read.format("delta").load(f"{BRONZE_PATH}/traffic_incidents")

# ── DQ Checks ───────────────────────────────────────────────────────────────
run_dq_checks(df_inc_raw, "traffic_incidents", [
    {"name": "incident_id not null",       "expr": "incident_id IS NULL",                   "threshold": 0.00},
    {"name": "incident_datetime not null", "expr": "incident_datetime IS NULL",              "threshold": 0.00},
    {"name": "severity is valid",          "expr": "severity NOT IN ('Fatal','Serious Injury','Minor Injury','Property Damage Only','Unknown')", "threshold": 0.01},
    {"name": "latitude in valid range",    "expr": "latitude < -90 OR latitude > 90",        "threshold": 0.00},
    {"name": "fatalities non-negative",    "expr": "fatalities < 0",                         "threshold": 0.00},
    {"name": "no future incidents",        "expr": "incident_datetime > current_timestamp()", "threshold": 0.00},
])

# ── Transformations ──────────────────────────────────────────────────────────
df_inc_silver = (
    df_inc_raw
    # Standardise strings
    .withColumn("incident_type",     F.upper(F.trim(F.col("incident_type"))))
    .withColumn("severity",          F.trim(F.col("severity")))
    .withColumn("state_code",        F.upper(F.col("state_code")))
    # Derive date parts for partitioning & reporting
    .withColumn("incident_year",     F.year("incident_datetime"))
    .withColumn("incident_month",    F.month("incident_datetime"))
    .withColumn("incident_hour",     F.hour("incident_datetime"))
    .withColumn("day_of_week",       F.dayofweek("incident_datetime"))
    .withColumn("is_weekend",        F.when(F.dayofweek("incident_datetime").isin(1,7), True).otherwise(False))
    # Casualty flag
    .withColumn("has_fatality",      F.when(F.col("fatalities") > 0, True).otherwise(False))
    .withColumn("total_casualties",  F.col("fatalities") + F.col("injuries"))
    # Severity numeric score for ranking
    .withColumn("severity_score",
        F.when(F.col("severity") == "Fatal", 5)
         .when(F.col("severity") == "Serious Injury", 4)
         .when(F.col("severity") == "Minor Injury", 3)
         .when(F.col("severity") == "Property Damage Only", 2)
         .otherwise(1))
    # Drop duplicates on natural key
    .dropDuplicates(["incident_id"])
    # Remove records with null critical fields
    .filter(F.col("incident_id").isNotNull())
    .filter(F.col("incident_datetime").isNotNull())
    .withColumn("silver_timestamp", F.current_timestamp())
)

df_inc_silver.write.format("delta").mode("overwrite").option("overwriteSchema","true") \
    .partitionBy("state_code","incident_year","incident_month") \
    .save(f"{SILVER_PATH}/traffic_incidents")

print(f"\n✓ Silver traffic_incidents → {df_inc_silver.count():,} rows")

# ─────────────────────────────────────────────────────────────────────────────
# SILVER 2 – BRIDGE INSPECTIONS
# ─────────────────────────────────────────────────────────────────────────────
df_brg_raw = spark.read.format("delta").load(f"{BRONZE_PATH}/bridge_inspections")

run_dq_checks(df_brg_raw, "bridge_inspections", [
    {"name": "bridge_id not null",             "expr": "bridge_id IS NULL",                   "threshold": 0.00},
    {"name": "year_built plausible",           "expr": "year_built < 1800 OR year_built > 2025", "threshold": 0.01},
    {"name": "sufficiency_rating 0-100",       "expr": "sufficiency_rating < 0 OR sufficiency_rating > 100", "threshold": 0.01},
    {"name": "avg_daily_traffic non-negative", "expr": "avg_daily_traffic < 0",               "threshold": 0.00},
])

# Age buckets and risk classification
df_brg_silver = (
    df_brg_raw
    .withColumn("state_code", F.upper(F.col("state_code")))
    .withColumn("bridge_age_years", F.lit(2024) - F.col("year_built"))
    .withColumn("age_category",
        F.when(F.col("bridge_age_years") < 20, "New (<20 yrs)")
         .when(F.col("bridge_age_years") < 40, "Mature (20-40 yrs)")
         .when(F.col("bridge_age_years") < 60, "Aging (40-60 yrs)")
         .otherwise("Old (60+ yrs)"))
    # Combine worst condition code for overall rating
    .withColumn("worst_condition",
        F.when(
            F.greatest(
                F.when(F.col("deck_condition").startswith("P"), 1).otherwise(0),
                F.when(F.col("superstructure_condition").startswith("P"), 1).otherwise(0),
                F.when(F.col("substructure_condition").startswith("P"), 1).otherwise(0),
            ) == 1, "Poor")
         .when(
            F.greatest(
                F.when(F.col("deck_condition").startswith("F"), 1).otherwise(0),
                F.when(F.col("superstructure_condition").startswith("F"), 1).otherwise(0),
                F.when(F.col("substructure_condition").startswith("F"), 1).otherwise(0),
            ) == 1, "Fair")
         .otherwise("Good"))
    # Risk score (simple composite)
    .withColumn("risk_score",
        F.round(
            (100 - F.col("sufficiency_rating")) * 0.5 +
            (F.lit(2024) - F.col("year_built")).cast("double") * 0.3 +
            F.when(F.col("structurally_deficient") == "true", 20).otherwise(0), 1))
    .withColumn("priority_tier",
        F.when(F.col("risk_score") >= 70, "Priority 1 – Immediate")
         .when(F.col("risk_score") >= 45, "Priority 2 – Near Term")
         .when(F.col("risk_score") >= 20, "Priority 3 – Planned")
         .otherwise("Priority 4 – Monitor"))
    # Days since last inspection
    .withColumn("days_since_inspection",
        F.datediff(F.current_date(), F.to_date("last_inspection_date")))
    .withColumn("inspection_overdue",
        F.when(F.col("days_since_inspection") > 730, True).otherwise(False))  # 2-yr cycle
    .dropDuplicates(["bridge_id"])
    .withColumn("silver_timestamp", F.current_timestamp())
)

df_brg_silver.write.format("delta").mode("overwrite").option("overwriteSchema","true") \
    .partitionBy("state_code") \
    .save(f"{SILVER_PATH}/bridge_inspections")

print(f"✓ Silver bridge_inspections → {df_brg_silver.count():,} rows")

# ─────────────────────────────────────────────────────────────────────────────
# SILVER 3 – VEHICLE REGISTRATIONS
# ─────────────────────────────────────────────────────────────────────────────
df_veh_raw = spark.read.format("delta").load(f"{BRONZE_PATH}/vehicle_registrations")

run_dq_checks(df_veh_raw, "vehicle_registrations", [
    {"name": "vin not null",            "expr": "vin IS NULL",                          "threshold": 0.00},
    {"name": "model_year plausible",    "expr": "model_year < 1900 OR model_year > 2025","threshold": 0.01},
    {"name": "gvwr positive",           "expr": "gvwr_lbs <= 0",                        "threshold": 0.00},
])

df_veh_silver = (
    df_veh_raw
    .withColumn("state_code",   F.upper(F.col("state_code")))
    .withColumn("vehicle_age",  F.lit(2024) - F.col("model_year"))
    .withColumn("weight_class",
        F.when(F.col("gvwr_lbs") <= 6000,  "Class 1 – Light Duty")
         .when(F.col("gvwr_lbs") <= 10000, "Class 2 – Light-Medium")
         .when(F.col("gvwr_lbs") <= 14000, "Class 3")
         .when(F.col("gvwr_lbs") <= 16000, "Class 4")
         .when(F.col("gvwr_lbs") <= 19500, "Class 5")
         .when(F.col("gvwr_lbs") <= 26000, "Class 6")
         .when(F.col("gvwr_lbs") <= 33000, "Class 7")
         .otherwise("Class 8 – Heavy Duty"))
    .withColumn("is_electric",      F.col("fuel_type").isin(["Electric","Hybrid"]))
    .withColumn("is_commercial",    F.col("commercial_vehicle") == "true")
    .withColumn("is_expired",       F.col("expiration_date") < F.current_timestamp())
    .withColumn("reg_year",         F.year("registration_date"))
    .dropDuplicates(["vin"])
    .withColumn("silver_timestamp", F.current_timestamp())
)

df_veh_silver.write.format("delta").mode("overwrite").option("overwriteSchema","true") \
    .partitionBy("state_code","vehicle_class") \
    .save(f"{SILVER_PATH}/vehicle_registrations")

print(f"✓ Silver vehicle_registrations → {df_veh_silver.count():,} rows")

# ─────────────────────────────────────────────────────────────────────────────
# SILVER 4 – PAVEMENT CONDITIONS
# ─────────────────────────────────────────────────────────────────────────────
df_pav_raw = spark.read.format("delta").load(f"{BRONZE_PATH}/pavement_conditions")

run_dq_checks(df_pav_raw, "pavement_conditions", [
    {"name": "segment_id not null",      "expr": "segment_id IS NULL",         "threshold": 0.00},
    {"name": "iri non-negative",         "expr": "iri < 0",                    "threshold": 0.00},
    {"name": "cracking_percent 0-100",   "expr": "cracking_percent < 0 OR cracking_percent > 100", "threshold": 0.01},
])

df_pav_silver = (
    df_pav_raw
    .withColumn("state_code",    F.upper(F.col("state_code")))
    .withColumn("pavement_age",  F.lit(2024) - F.col("year_constructed"))
    # IRI category (lower = smoother)
    .withColumn("iri_category",
        F.when(F.col("iri") <= 60,  "Very Good")
         .when(F.col("iri") <= 95,  "Good")
         .when(F.col("iri") <= 170, "Fair")
         .when(F.col("iri") <= 220, "Mediocre")
         .otherwise("Poor"))
    # Distress composite index
    .withColumn("distress_index",
        F.round(F.col("cracking_percent") * 0.4 + F.col("rutting_in") * 5 +
                (100 - F.col("pcr")) * 0.6, 1))
    .withColumn("maintenance_priority",
        F.when(F.col("distress_index") >= 60, "Immediate Rehab")
         .when(F.col("distress_index") >= 40, "Preventive Treatment")
         .when(F.col("distress_index") >= 20, "Monitor")
         .otherwise("No Action"))
    # Traffic load category
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

print(f"✓ Silver pavement_conditions → {df_pav_silver.count():,} rows")

# ─────────────────────────────────────────────────────────────────────────────
# Register Silver tables in Hive Metastore (or Unity Catalog)
# ─────────────────────────────────────────────────────────────────────────────
spark.sql("CREATE DATABASE IF NOT EXISTS main.dot_silver")

for tbl in ["traffic_incidents","bridge_inspections","vehicle_registrations","pavement_conditions"]:
    spark.sql(f"""
        CREATE OR REPLACE TABLE main.dot_silver.{tbl}
        AS SELECT * FROM delta.`{SILVER_PATH}/{tbl}`
    """)
    print(f"  Registered: main.dot_silver.{tbl}")

print("\n✅  Silver layer complete.")
