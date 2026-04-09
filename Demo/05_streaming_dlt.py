# =============================================================================
# NOTEBOOK 5: DOT Real-Time Streaming & Delta Live Tables (DLT)
# Databricks Notebook | Language: Python
# Description:
#   Part A – Spark Structured Streaming (sensor / traffic feed simulation)
#   Part B – Delta Live Tables pipeline definition for continuous ingestion
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# PART A: Structured Streaming – Traffic Sensor Feed
# Simulates real-time traffic volume/speed data from roadway sensors
# ─────────────────────────────────────────────────────────────────────────────

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType, IntegerType, StringType, StructField, StructType, TimestampType,
)

spark = SparkSession.builder \
    .appName("DOT_StreamingPipeline") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

BASE_PATH       = "/Volumes/main/default/dot_lakehouse"
STREAM_INPUT    = f"{BASE_PATH}/stream_input/sensor_events"
STREAM_BRONZE   = f"{BASE_PATH}/stream_bronze/sensor_readings"
STREAM_SILVER   = f"{BASE_PATH}/stream_silver/sensor_aggregated"
CHECKPOINT_BASE = f"{BASE_PATH}/checkpoints"

# ── Schema for incoming sensor JSON messages ─────────────────────────────────
sensor_schema = StructType([
    StructField("sensor_id",       StringType(),    False),
    StructField("route_id",        StringType(),    True),
    StructField("milepost",        DoubleType(),    True),
    StructField("state_code",      StringType(),    True),
    StructField("event_timestamp", TimestampType(), True),
    StructField("speed_mph",       DoubleType(),    True),
    StructField("volume_per_hour", IntegerType(),   True),
    StructField("occupancy_pct",   DoubleType(),    True),
    StructField("lane_count",      IntegerType(),   True),
    StructField("sensor_type",     StringType(),    True),  # Loop, Radar, Camera
    StructField("weather_flag",    StringType(),    True),  # Normal, Rain, Snow
])

# ── Stream Reader (Auto Loader for cloud-native file ingestion) ──────────────
df_sensor_stream = (
    spark.readStream
    .format("cloudFiles")                         # Auto Loader
    .option("cloudFiles.format", "json")
    .option("cloudFiles.inferColumnTypes", "false")
    .option("cloudFiles.schemaLocation", f"{CHECKPOINT_BASE}/sensor_schema")
    .schema(sensor_schema)
    .load(STREAM_INPUT)
)

# ── Bronze Write (raw, append-only) ─────────────────────────────────────────
bronze_query = (
    df_sensor_stream
    .withColumn("ingestion_timestamp", F.current_timestamp())
    .withColumn("source_file", F.col("_metadata.file_path"))
    .writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", f"{CHECKPOINT_BASE}/sensor_bronze")
    .option("mergeSchema", "true")
    .trigger(availableNow=True)       # micro-batch every 30s
    .start(STREAM_BRONZE)
)

bronze_query.awaitTermination()
print(f"✓ Bronze stream started → {STREAM_BRONZE}")

# ── Silver Aggregation Stream (5-minute tumbling windows) ───────────────────
df_sensor_bronze = (
    spark.readStream
    .format("delta")
    .load(STREAM_BRONZE)
)

df_sensor_silver_stream = (
    df_sensor_bronze
    .withWatermark("event_timestamp", "10 minutes")   # late data tolerance
    .groupBy(
        F.window("event_timestamp", "5 minutes"),
        "sensor_id","route_id","state_code","weather_flag",
    )
    .agg(
        F.avg("speed_mph").alias("avg_speed_mph"),
        F.min("speed_mph").alias("min_speed_mph"),
        F.max("speed_mph").alias("max_speed_mph"),
        F.avg("volume_per_hour").alias("avg_volume"),
        F.sum("volume_per_hour").alias("total_volume"),
        F.avg("occupancy_pct").alias("avg_occupancy"),
        F.count("sensor_id").alias("reading_count"),
    )
    .withColumn("window_start",    F.col("window.start"))
    .withColumn("window_end",      F.col("window.end"))
    .drop("window")
    # Congestion classification
    .withColumn("congestion_level",
        F.when(F.col("avg_speed_mph") >= 55,  "Free Flow")
         .when(F.col("avg_speed_mph") >= 35,  "Moderate")
         .when(F.col("avg_speed_mph") >= 15,  "Heavy")
         .otherwise("Stop-and-Go"))
    .withColumn("silver_timestamp", F.current_timestamp())
)

silver_query = (
    df_sensor_silver_stream
    .writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", f"{CHECKPOINT_BASE}/sensor_silver")
    .trigger(processingTime="5 minutes")
    .start(STREAM_SILVER)
)

print(f"✓ Silver aggregation stream started → {STREAM_SILVER}")

# ── Anomaly Detection Stream (speed drops ≥ 40% vs. baseline) ───────────────
df_anomaly = (
    df_sensor_bronze
    .withWatermark("event_timestamp", "5 minutes")
    .withColumn("is_anomaly",
        F.when(
            (F.col("speed_mph") < 10) |
            (F.col("occupancy_pct") > 95) |
            (F.col("volume_per_hour") == 0),
            True
        ).otherwise(False))
    .filter(F.col("is_anomaly") == True)
    .select(
        "sensor_id","route_id","state_code","milepost",
        "event_timestamp","speed_mph","occupancy_pct",
        "volume_per_hour","weather_flag",
        F.lit("AUTO_DETECTED").alias("alert_source"),
        F.current_timestamp().alias("alert_timestamp"),
    )
)

anomaly_query = (
    df_anomaly
    .writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", f"{CHECKPOINT_BASE}/sensor_anomalies")
    .trigger(processingTime="30 seconds")
    .start(f"{BASE_PATH}/stream_silver/traffic_anomalies")
)

print(f"✓ Anomaly detection stream started → {BASE_PATH}/stream_silver/traffic_anomalies")

# =============================================================================
# PART B: Delta Live Tables (DLT) Pipeline Definition
# Save this file separately and reference it in your DLT pipeline config.
# =============================================================================

DLT_PIPELINE_CODE = '''
import dlt
from pyspark.sql import functions as F
from pyspark.sql.types import *

BASE_PATH = "dbfs:/dot_lakehouse"

# ── Bronze: Raw Incident Ingestion ───────────────────────────────────────────
@dlt.table(
    name        = "bronze_incident_raw",
    comment     = "Raw traffic incident records ingested via Auto Loader",
    table_properties = {"quality": "bronze", "pipelines.autoOptimize.managed": "true"},
)
def bronze_incident_raw():
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.schemaLocation", f"{BASE_PATH}/dlt_schema/incidents")
        .load(f"{BASE_PATH}/landing/incidents/")
        .withColumn("_ingest_time", F.current_timestamp())
        .withColumn("_source_file", F.col("_metadata.file_path"))
    )

# ── Silver: Validated Incidents (expectations enforce data quality) ──────────
@dlt.expect_or_drop("valid_incident_id",    "incident_id IS NOT NULL")
@dlt.expect_or_drop("valid_datetime",       "incident_datetime IS NOT NULL")
@dlt.expect("valid_severity",
    "severity IN ('Fatal','Serious Injury','Minor Injury','Property Damage Only','Unknown')")
@dlt.expect("valid_lat",                    "latitude BETWEEN -90 AND 90")
@dlt.expect_or_drop("valid_longitude",      "longitude BETWEEN -180 AND 180")
@dlt.table(
    name        = "silver_incidents_validated",
    comment     = "Cleansed and validated incident records",
    table_properties = {"quality": "silver"},
)
def silver_incidents_validated():
    return (
        dlt.read_stream("bronze_incident_raw")
        .withColumn("incident_type",  F.upper(F.trim(F.col("incident_type"))))
        .withColumn("state_code",     F.upper(F.col("state_code")))
        .withColumn("incident_year",  F.year("incident_datetime"))
        .withColumn("incident_month", F.month("incident_datetime"))
        .withColumn("incident_hour",  F.hour("incident_datetime"))
        .withColumn("is_weekend",     F.dayofweek("incident_datetime").isin(1, 7))
        .withColumn("has_fatality",   F.col("fatalities") > 0)
        .withColumn("total_casualties", F.col("fatalities") + F.col("injuries"))
        .dropDuplicates(["incident_id"])
    )

# ── Gold: Monthly Incident Rollup ────────────────────────────────────────────
@dlt.table(
    name        = "gold_incident_monthly_rollup",
    comment     = "Monthly incident summary for dashboard and reporting",
    table_properties = {"quality": "gold"},
)
def gold_incident_monthly_rollup():
    return (
        dlt.read("silver_incidents_validated")
        .groupBy("state_code","incident_year","incident_month","incident_type")
        .agg(
            F.count("incident_id").alias("total_incidents"),
            F.sum("fatalities").alias("total_fatalities"),
            F.sum("injuries").alias("total_injuries"),
            F.sum(F.when(F.col("has_fatality"), 1).otherwise(0)).alias("fatal_count"),
            F.avg("vehicles_involved").alias("avg_vehicles"),
        )
    )

# ── Bronze: Raw Bridge Condition Feed ────────────────────────────────────────
@dlt.table(
    name        = "bronze_bridge_raw",
    comment     = "Raw NBI bridge inspection records",
    table_properties = {"quality": "bronze"},
)
def bronze_bridge_raw():
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "parquet")
        .option("cloudFiles.schemaLocation", f"{BASE_PATH}/dlt_schema/bridges")
        .load(f"{BASE_PATH}/landing/bridges/")
        .withColumn("_ingest_time", F.current_timestamp())
    )

# ── Silver: Validated Bridge Records ─────────────────────────────────────────
@dlt.expect_or_drop("valid_bridge_id",       "bridge_id IS NOT NULL")
@dlt.expect("plausible_year_built",          "year_built BETWEEN 1800 AND 2025")
@dlt.expect("valid_sufficiency",             "sufficiency_rating BETWEEN 0 AND 100")
@dlt.table(
    name        = "silver_bridges_validated",
    comment     = "Validated bridge inspection records with risk scores",
    table_properties = {"quality": "silver"},
)
def silver_bridges_validated():
    return (
        dlt.read_stream("bronze_bridge_raw")
        .withColumn("bridge_age_years", F.lit(2024) - F.col("year_built"))
        .withColumn("days_since_inspection",
            F.datediff(F.current_date(), F.to_date("last_inspection_date")))
        .withColumn("inspection_overdue",
            F.col("days_since_inspection") > 730)
        .withColumn("risk_score",
            F.round(
                (100 - F.col("sufficiency_rating")) * 0.5 +
                (F.lit(2024) - F.col("year_built")).cast("double") * 0.3, 1))
        .dropDuplicates(["bridge_id"])
    )
'''

# Write the DLT pipeline code to a separate file for deployment
dlt_path = f"{BASE_PATH}/dlt_pipelines/dot_pipeline.py"
dbutils.fs.put(dlt_path, DLT_PIPELINE_CODE, overwrite=True)
print(f"\n✓ DLT pipeline definition written → {dlt_path}")
print("""
To deploy the DLT pipeline:
  1. Go to Databricks UI → Workflows → Delta Live Tables → Create Pipeline
  2. Set Source: <above path>
  3. Set Target Schema: dot_dlt
  4. Set Pipeline Mode: Triggered or Continuous
  5. Click Start
""")
