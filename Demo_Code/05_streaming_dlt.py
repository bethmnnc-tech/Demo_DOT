# =============================================================================
# DOT Streaming Ingestion — Spark Declarative Pipeline (DLT)
# Language: Python
# Description:
#   Bronze/Silver/Gold pipeline for incidents and bridge inspections.
#   Reads base_path from pipeline configuration (set per target in databricks.yml).
#
# NOTE: Part A (standalone Structured Streaming with writeStream/start) was
#       removed — those APIs are forbidden in DLT. Move Part A to a separate
#       file if standalone streaming is needed outside the pipeline.
# =============================================================================

import dlt
from pyspark.sql import functions as F

# ── Read base_path from pipeline configuration ───────────────────────────────
# Set via databricks.yml → resources.pipelines.configuration.base_path
#   dev  → /Volumes/main/default/dot_lakehouse
#   prod → /Volumes/main_prod/default/dot_lakehouse
BASE_PATH = spark.conf.get("base_path", "/Volumes/main/default/dot_lakehouse")

# =============================================================================
# BRONZE LAYER
# =============================================================================

# ── Bronze: Raw Incident Ingestion ───────────────────────────────────────────
@dlt.table(
    name="bronze_incident_raw",
    comment="Traffic incident records streamed from bronze Delta tables",
    table_properties={"quality": "bronze", "pipelines.autoOptimize.managed": "true"},
)
def bronze_incident_raw():
    return (
        spark.readStream
        .format("delta")
        .load(f"{BASE_PATH}/bronze/traffic_incidents")
        .withColumn("_ingest_time", F.current_timestamp())
    )

# ── Bronze: Raw Bridge Condition Feed ────────────────────────────────────────
@dlt.table(
    name="bronze_bridge_raw",
    comment="NBI bridge inspection records streamed from bronze Delta tables",
    table_properties={"quality": "bronze"},
)
def bronze_bridge_raw():
    return (
        spark.readStream
        .format("delta")
        .load(f"{BASE_PATH}/bronze/bridge_inspections")
        .withColumn("_ingest_time", F.current_timestamp())
    )

# =============================================================================
# SILVER LAYER
# =============================================================================

# ── Silver: Validated Incidents ──────────────────────────────────────────────
@dlt.expect_or_drop("valid_incident_id", "incident_id IS NOT NULL")
@dlt.expect_or_drop("valid_datetime", "incident_datetime IS NOT NULL")
@dlt.expect("valid_severity",
    "severity IN ('Fatal','Serious Injury','Minor Injury','Property Damage Only','Unknown')")
@dlt.expect("valid_lat", "latitude BETWEEN -90 AND 90")
@dlt.expect_or_drop("valid_longitude", "longitude BETWEEN -180 AND 180")
@dlt.table(
    name="silver_incidents_validated",
    comment="Cleansed and validated incident records",
    table_properties={"quality": "silver"},
)
def silver_incidents_validated():
    return (
        dlt.read_stream("bronze_incident_raw")
        .withColumn("incident_type", F.upper(F.trim(F.col("incident_type"))))
        .withColumn("state_code", F.upper(F.col("state_code")))
        .withColumn("incident_year", F.year("incident_datetime"))
        .withColumn("incident_month", F.month("incident_datetime"))
        .withColumn("incident_hour", F.hour("incident_datetime"))
        .withColumn("is_weekend", F.dayofweek("incident_datetime").isin(1, 7))
        .withColumn("has_fatality", F.col("fatalities") > 0)
        .withColumn("total_casualties", F.col("fatalities") + F.col("injuries"))
        .dropDuplicates(["incident_id"])
    )

# ── Silver: Validated Bridge Records ─────────────────────────────────────────
@dlt.expect_or_drop("valid_bridge_id", "bridge_id IS NOT NULL")
@dlt.expect("plausible_year_built", "year_built BETWEEN 1800 AND 2025")
@dlt.expect("valid_sufficiency", "sufficiency_rating BETWEEN 0 AND 100")
@dlt.table(
    name="silver_bridges_validated",
    comment="Validated bridge inspection records with risk scores",
    table_properties={"quality": "silver"},
)
def silver_bridges_validated():
    return (
        dlt.read_stream("bronze_bridge_raw")
        .withColumn("bridge_age_years", F.lit(2024) - F.col("year_built"))
        .withColumn("days_since_inspection",
            F.datediff(F.current_date(), F.to_date("last_inspection_date")))
        .withColumn("inspection_overdue", F.col("days_since_inspection") > 730)
        .withColumn("risk_score",
            F.round(
                (100 - F.col("sufficiency_rating")) * 0.5 +
                (F.lit(2024) - F.col("year_built")).cast("double") * 0.3, 1))
        .dropDuplicates(["bridge_id"])
    )

# =============================================================================
# GOLD LAYER
# =============================================================================

# ── Gold: Monthly Incident Rollup ────────────────────────────────────────────
@dlt.table(
    name="gold_incident_monthly_rollup",
    comment="Monthly incident summary for dashboard and reporting",
    table_properties={"quality": "gold"},
)
def gold_incident_monthly_rollup():
    return (
        dlt.read("silver_incidents_validated")
        .groupBy("state_code", "incident_year", "incident_month", "incident_type")
        .agg(
            F.count("incident_id").alias("total_incidents"),
            F.sum("fatalities").alias("total_fatalities"),
            F.sum("injuries").alias("total_injuries"),
            F.sum(F.when(F.col("has_fatality"), 1).otherwise(0)).alias("fatal_count"),
            F.avg("vehicles_involved").alias("avg_vehicles"),
        )
    )
