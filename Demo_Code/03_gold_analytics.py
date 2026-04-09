# =============================================================================
# NOTEBOOK 3: DOT Gold Layer – Analytical Aggregations & Data Mart s
# Databricks Notebook | Language: PySpark + SQL
# Description: Builds Gold-layer aggregated tables and views used by
#              dashboards, ML models, and reporting teams.
# =============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("main.dot_goldAnalytics").getOrCreate()

BASE_PATH   = "/Volumes/main/default/dot_lakehouse"
SILVER_PATH = f"{BASE_PATH}/silver"
GOLD_PATH   = f"{BASE_PATH}/gold"

spark.sql("CREATE DATABASE IF NOT EXISTS main.dot_gold")

# ─────────────────────────────────────────────────────────────────────────────
# GOLD 1 – Incident Summary by Route / Year / Month
# Used by: Traffic Safety Dashboard, FHWA reporting
# ─────────────────────────────────────────────────────────────────────────────
df_inc = spark.read.format("delta").load(f"{SILVER_PATH}/traffic_incidents")

df_incident_summary = (
    df_inc
    .groupBy("state_code","route_id","incident_year","incident_month","incident_type","severity")
    .agg(
        F.count("incident_id").alias("total_incidents"),
        F.sum("fatalities").alias("total_fatalities"),
        F.sum("injuries").alias("total_injuries"),
        F.sum("total_casualties").alias("total_casualties"),
        F.avg("vehicles_involved").alias("avg_vehicles_involved"),
        F.countDistinct("county_code").alias("counties_affected"),
        F.sum(F.when(F.col("has_fatality"), 1).otherwise(0)).alias("fatal_incidents"),
        F.sum(F.when(F.col("is_weekend"), 1).otherwise(0)).alias("weekend_incidents"),
        # Peak hour distribution
        F.sum(F.when(F.col("incident_hour").between(6, 9), 1).otherwise(0)).alias("morning_rush_incidents"),
        F.sum(F.when(F.col("incident_hour").between(15, 19), 1).otherwise(0)).alias("evening_rush_incidents"),
        F.sum(F.when(F.col("incident_hour").between(22, 23) | F.col("incident_hour").between(0, 5), 1).otherwise(0)).alias("overnight_incidents"),
    )
    .withColumn("fatality_rate",    F.round(F.col("total_fatalities") / F.col("total_incidents"), 4))
    .withColumn("injury_rate",      F.round(F.col("total_injuries")   / F.col("total_incidents"), 4))
    .withColumn("gold_timestamp",   F.current_timestamp())
)

df_incident_summary.write.format("delta").mode("overwrite").option("overwriteSchema","true") \
    .partitionBy("state_code","incident_year") \
    .save(f"{GOLD_PATH}/incident_summary")

spark.sql(f"""
    CREATE OR REPLACE TABLE main.dot_gold.incident_summary
    AS SELECT * FROM delta.`{GOLD_PATH}/incident_summary`
""")
print(f"✓ Gold incident_summary → {df_incident_summary.count():,} rows")

# ─────────────────────────────────────────────────────────────────────────────
# GOLD 2 – High-Risk Corridors
# Identifies route segments with disproportionate fatality concentration
# ─────────────────────────────────────────────────────────────────────────────
windowSpec = Window.partitionBy("state_code").orderBy(F.desc("total_fatalities"))

df_high_risk = (
    df_inc
    .groupBy("state_code","route_id","county_code")
    .agg(
        F.count("incident_id").alias("incident_count"),
        F.sum("fatalities").alias("total_fatalities"),
        F.sum("injuries").alias("total_injuries"),
        F.countDistinct("incident_type").alias("incident_type_diversity"),
        F.avg("severity_score").alias("avg_severity_score"),
        F.max("incident_datetime").alias("most_recent_incident"),
    )
    .withColumn("fatality_concentration", F.round(F.col("total_fatalities") / F.col("incident_count"), 3))
    .withColumn("state_rank", F.rank().over(windowSpec))
    .withColumn("risk_tier",
        F.when(F.col("state_rank") <= 10, "Top 10 – Critical")
         .when(F.col("state_rank") <= 25, "Top 25 – High Risk")
         .when(F.col("state_rank") <= 50, "Top 50 – Elevated")
         .otherwise("Standard"))
    .withColumn("gold_timestamp", F.current_timestamp())
)

df_high_risk.write.format("delta").mode("overwrite").option("overwriteSchema","true") \
    .partitionBy("state_code") \
    .save(f"{GOLD_PATH}/high_risk_corridors")

spark.sql(f"""
    CREATE OR REPLACE TABLE main.dot_gold.high_risk_corridors
    AS SELECT * FROM delta.`{GOLD_PATH}/high_risk_corridors`
""")
print(f"✓ Gold high_risk_corridors → {df_high_risk.count():,} rows")

# ─────────────────────────────────────────────────────────────────────────────
# GOLD 3 – Bridge Infrastructure Health Summary
# Used by: Asset Management, Capital Planning, FHWA NBI Reporting
# ─────────────────────────────────────────────────────────────────────────────
df_brg = spark.read.format("delta").load(f"{SILVER_PATH}/bridge_inspections")

df_bridge_health = (
    df_brg
    .groupBy("state_code","county_code","bridge_type","age_category","priority_tier")
    .agg(
        F.count("bridge_id").alias("bridge_count"),
        F.avg("sufficiency_rating").alias("avg_sufficiency_rating"),
        F.min("sufficiency_rating").alias("min_sufficiency_rating"),
        F.avg("bridge_age_years").alias("avg_bridge_age"),
        F.avg("avg_daily_traffic").alias("avg_daily_traffic"),
        F.sum("avg_daily_traffic").alias("total_daily_traffic_exposed"),
        F.sum(F.when(F.col("structurally_deficient") == "true", 1).otherwise(0)).alias("structurally_deficient_count"),
        F.sum(F.when(F.col("functionally_obsolete") == "true", 1).otherwise(0)).alias("functionally_obsolete_count"),
        F.sum(F.when(F.col("inspection_overdue") == True, 1).otherwise(0)).alias("overdue_inspections"),
        F.sum("estimated_repair_cost_k").alias("total_estimated_repair_cost_k"),
        F.avg("risk_score").alias("avg_risk_score"),
        F.max("risk_score").alias("max_risk_score"),
    )
    .withColumn("deficiency_rate",
        F.round(F.col("structurally_deficient_count") / F.col("bridge_count"), 3))
    .withColumn("total_estimated_repair_cost_m",
        F.round(F.col("total_estimated_repair_cost_k") / 1000, 2))
    .withColumn("gold_timestamp", F.current_timestamp())
)

df_bridge_health.write.format("delta").mode("overwrite").option("overwriteSchema","true") \
    .partitionBy("state_code") \
    .save(f"{GOLD_PATH}/bridge_health_summary")

spark.sql(f"""
    CREATE OR REPLACE TABLE main.dot_gold.bridge_health_summary
    AS SELECT * FROM delta.`{GOLD_PATH}/bridge_health_summary`
""")
print(f"✓ Gold bridge_health_summary → {df_bridge_health.count():,} rows")

# ─────────────────────────────────────────────────────────────────────────────
# GOLD 4 – Fleet & Emissions Snapshot
# Used by: Environmental, Commercial Vehicle Operations
# ─────────────────────────────────────────────────────────────────────────────
df_veh = spark.read.format("delta").load(f"{SILVER_PATH}/vehicle_registrations")

df_fleet = (
    df_veh
    .groupBy("state_code","vehicle_class","weight_class","fuel_type","model_year")
    .agg(
        F.count("vin").alias("registered_vehicles"),
        F.avg("vehicle_age").alias("avg_vehicle_age"),
        F.avg("gvwr_lbs").alias("avg_gvwr_lbs"),
        F.sum(F.when(F.col("is_electric"), 1).otherwise(0)).alias("electric_count"),
        F.sum(F.when(F.col("is_commercial"), 1).otherwise(0)).alias("commercial_count"),
        F.sum(F.when(F.col("hazmat_certified") == "true", 1).otherwise(0)).alias("hazmat_count"),
        F.sum(F.when(F.col("is_expired"), 1).otherwise(0)).alias("expired_registrations"),
    )
    .withColumn("electric_share",  F.round(F.col("electric_count")   / F.col("registered_vehicles"), 3))
    .withColumn("commercial_share",F.round(F.col("commercial_count") / F.col("registered_vehicles"), 3))
    .withColumn("gold_timestamp",  F.current_timestamp())
)

df_fleet.write.format("delta").mode("overwrite").option("overwriteSchema","true") \
    .partitionBy("state_code","vehicle_class") \
    .save(f"{GOLD_PATH}/fleet_snapshot")

spark.sql(f"""
    CREATE OR REPLACE TABLE main.dot_gold.fleet_snapshot
    AS SELECT * FROM delta.`{GOLD_PATH}/fleet_snapshot`
""")
print(f"✓ Gold fleet_snapshot → {df_fleet.count():,} rows")

# ─────────────────────────────────────────────────────────────────────────────
# GOLD 5 – Pavement Needs Assessment
# Used by: Asset Management, Budget / Capital Improvement Plan
# ─────────────────────────────────────────────────────────────────────────────
df_pav = spark.read.format("delta").load(f"{SILVER_PATH}/pavement_conditions")

# Unit cost assumptions ($/lane-mile) by treatment type
treatment_costs = {
    "Immediate Rehab":       800_000,
    "Preventive Treatment":  250_000,
    "Monitor":               10_000,
    "No Action":             0,
}
cost_map_expr = (
    F.when(F.col("maintenance_priority") == "Immediate Rehab",      800000)
     .when(F.col("maintenance_priority") == "Preventive Treatment", 250000)
     .when(F.col("maintenance_priority") == "Monitor",              10000)
     .otherwise(0)
)

df_pavement_needs = (
    df_pav
    .withColumn("estimated_cost_per_mile", cost_map_expr)
    .withColumn("total_segment_cost",
        F.col("estimated_cost_per_mile") * F.col("segment_length_mi"))
    .groupBy("state_code","county_code","functional_class","pavement_type","maintenance_priority","iri_category")
    .agg(
        F.count("segment_id").alias("segment_count"),
        F.sum("segment_length_mi").alias("total_lane_miles"),
        F.avg("pcr").alias("avg_pcr"),
        F.avg("iri").alias("avg_iri"),
        F.avg("distress_index").alias("avg_distress_index"),
        F.avg("pavement_age").alias("avg_pavement_age"),
        F.sum("total_segment_cost").alias("total_estimated_cost_usd"),
        F.sum("aadt").alias("total_aadt"),
    )
    .withColumn("cost_per_aadt",
        F.when(F.col("total_aadt") > 0,
               F.round(F.col("total_estimated_cost_usd") / F.col("total_aadt"), 2))
         .otherwise(0))
    .withColumn("gold_timestamp", F.current_timestamp())
)

df_pavement_needs.write.format("delta").mode("overwrite").option("overwriteSchema","true") \
    .partitionBy("state_code","functional_class") \
    .save(f"{GOLD_PATH}/pavement_needs_assessment")

spark.sql(f"""
    CREATE OR REPLACE TABLE main.dot_gold.pavement_needs_assessment
    AS SELECT * FROM delta.`{GOLD_PATH}/pavement_needs_assessment`
""")
print(f"✓ Gold pavement_needs_assessment → {df_pavement_needs.count():,} rows")

# ─────────────────────────────────────────────────────────────────────────────
# GOLD 6 – Executive KPI Scorecard (single-row-per-state summary)
# ─────────────────────────────────────────────────────────────────────────────
inc_kpi = (
    df_inc.groupBy("state_code")
    .agg(
        F.count("incident_id").alias("total_incidents"),
        F.sum("fatalities").alias("total_fatalities"),
        F.sum("injuries").alias("total_injuries"),
    )
)

brg_kpi = (
    df_brg.groupBy("state_code")
    .agg(
        F.count("bridge_id").alias("total_bridges"),
        F.avg("sufficiency_rating").alias("avg_bridge_sufficiency"),
        F.sum(F.when(F.col("structurally_deficient") == "true", 1).otherwise(0)).alias("deficient_bridges"),
    )
)

pav_kpi = (
    df_pav.groupBy("state_code")
    .agg(
        F.sum("segment_length_mi").alias("total_lane_miles"),
        F.avg("pcr").alias("avg_pcr"),
        F.sum(F.when(F.col("condition_rating").isin("Poor","Very Poor"), F.col("segment_length_mi")).otherwise(0))
         .alias("poor_pavement_miles"),
    )
)

veh_kpi = (
    df_veh.groupBy("state_code")
    .agg(
        F.count("vin").alias("total_registered_vehicles"),
        F.sum(F.when(F.col("is_electric"), 1).otherwise(0)).alias("electric_vehicles"),
    )
)

df_scorecard = (
    inc_kpi
    .join(brg_kpi, "state_code", "left")
    .join(pav_kpi, "state_code", "left")
    .join(veh_kpi, "state_code", "left")
    .withColumn("fatality_per_100k_vehicles",
        F.round(F.col("total_fatalities") / F.col("total_registered_vehicles") * 100000, 2))
    .withColumn("poor_pavement_pct",
        F.round(F.col("poor_pavement_miles") / F.col("total_lane_miles") * 100, 1))
    .withColumn("bridge_deficiency_pct",
        F.round(F.col("deficient_bridges") / F.col("total_bridges") * 100, 1))
    .withColumn("ev_adoption_pct",
        F.round(F.col("electric_vehicles") / F.col("total_registered_vehicles") * 100, 1))
    .withColumn("gold_timestamp", F.current_timestamp())
)

df_scorecard.write.format("delta").mode("overwrite").option("overwriteSchema","true") \
    .save(f"{GOLD_PATH}/executive_kpi_scorecard")

spark.sql(f"""
    CREATE OR REPLACE TABLE main.dot_gold.executive_kpi_scorecard
    AS SELECT * FROM delta.`{GOLD_PATH}/executive_kpi_scorecard`
""")
print(f"✓ Gold executive_kpi_scorecard → {df_scorecard.count():,} rows")

# ─── Quick preview ──────────────────────────────────────────────────────────
print("\n── Executive KPI Scorecard Preview ──")
spark.sql("""
SELECT state_code,
       total_incidents,
       total_fatalities,
       total_bridges,
       ROUND(avg_bridge_sufficiency,1)  AS avg_bridge_sufficiency,
       ROUND(total_lane_miles,0)        AS total_lane_miles,
       poor_pavement_pct,
       total_registered_vehicles,
       ev_adoption_pct
FROM main.dot_gold.executive_kpi_scorecard
ORDER BY total_fatalities DESC
""").show(15)

print("\n✅  Gold layer complete.")
