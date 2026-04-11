# =============================================================================
# NOTEBOOK 7b: DOT Geospatial Analytics – Spatial Joins, H3 Grid & Hotspots
# Databricks Notebook | DBR 14.3 LTS ML
#
# Description:
#   1. Enrich incident / bridge records with H3 cell + nearest road segment
#   2. H3 hexagonal grid aggregation for density mapping
#   3. Spatial hotspot detection (incident clustering)
#   4. Corridor buffer analysis (incidents within 0.5 mi of a corridor)
#   5. Isochrone / travel-time zone simulation
#   6. Write enriched geospatial Gold tables
# =============================================================================

import subprocess
subprocess.check_call(["pip", "install", "-q", "h3", "shapely"])

import math
import h3
import pandas as pd
import numpy as np
from typing import List

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, DoubleType, IntegerType, ArrayType

import sys

# ── Configuration ────────────────────────────────────────────────────────────
# When run by a job: parameters arrive via sys.argv
# When run interactively: dbutils widgets provide a UI with dev defaults
if len(sys.argv) >= 3 and sys.argv[1].startswith("/"):
    BASE_PATH = sys.argv[1]
    CATALOG   = sys.argv[2]
else:
    dbutils.widgets.text("base_path", "/Volumes/main/default/dot_lakehouse")
    dbutils.widgets.text("catalog", "main")
    BASE_PATH = dbutils.widgets.get("base_path")
    CATALOG   = dbutils.widgets.get("catalog")

print(f"  BASE_PATH = {BASE_PATH}")
print(f"  CATALOG   = {CATALOG}")

spark = SparkSession.builder.appName("DOT_GeospatialAnalytics").getOrCreate()

GEO_PATH   = f"{BASE_PATH}/bronze/geospatial"
SILVER_PATH= f"{BASE_PATH}/silver"
GOLD_PATH  = f"{BASE_PATH}/gold/geospatial"

spark.sql(f"CREATE DATABASE IF NOT EXISTS {CATALOG}.dot_geo")

# ─────────────────────────────────────────────────────────────────────────────
# Driver-side H3 helpers
# On serverless (Spark Connect), h3 is only available on the driver — not
# inside UDFs which execute on the server.  We collect lat/lon to pandas,
# compute H3 indices on the driver, then join the result back to Spark.
# ─────────────────────────────────────────────────────────────────────────────

def _add_h3_columns(df_spark, id_col, lat_col="latitude", lon_col="longitude",
                    resolutions=None):
    """Compute H3 indices on the driver and join back as new columns."""
    if resolutions is None:
        resolutions = [6, 7, 8]
    pdf = df_spark.select(id_col, lat_col, lon_col).toPandas()
    for res in resolutions:
        col_name = f"h3_index_r{res}"
        pdf[col_name] = pdf.apply(
            lambda r, _r=res: h3.latlng_to_cell(float(r[lat_col]), float(r[lon_col]), _r)
            if pd.notna(r[lat_col]) and pd.notna(r[lon_col]) else None,
            axis=1,
        )
    h3_cols = [id_col] + [f"h3_index_r{r}" for r in resolutions]
    df_h3 = spark.createDataFrame(pdf[h3_cols])
    return df_spark.join(df_h3, on=id_col, how="left")


def _add_h3_disk(df_spark, id_col, lat_col="latitude", lon_col="longitude",
                 k_rings=2, resolution=8):
    """Compute H3 k-ring disk on driver, return (id, h3_neighbor) rows."""
    pdf = df_spark.select(id_col, lat_col, lon_col).toPandas()
    rows = []
    for _, r in pdf.iterrows():
        if pd.notna(r[lat_col]) and pd.notna(r[lon_col]):
            center = h3.latlng_to_cell(float(r[lat_col]), float(r[lon_col]), resolution)
            for cell in h3.grid_disk(center, k_rings):
                rows.append((r[id_col], cell))
    return spark.createDataFrame(rows, [id_col, "h3_neighbor"])


def _haversine_expr(lat1, lon1, lat2, lon2):
    """Spark-native haversine distance in miles (no UDF)."""
    R = F.lit(3958.8)
    phi1 = F.radians(lat1)
    phi2 = F.radians(lat2)
    dphi = F.radians(lat2 - lat1)
    dlam = F.radians(lon2 - lon1)
    a = (F.sin(dphi / 2) ** 2 +
         F.cos(phi1) * F.cos(phi2) * F.sin(dlam / 2) ** 2)
    return F.round(R * F.lit(2) * F.atan2(F.sqrt(a), F.sqrt(F.lit(1) - a)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 – Enrich Silver Incidents with Geospatial Attributes
# ─────────────────────────────────────────────────────────────────────────────
print("Step 1: Enriching incidents with H3 + geometry …")

df_inc = spark.read.format("delta").load(f"{SILVER_PATH}/traffic_incidents")
df_roads = spark.read.format("delta").load(f"{GEO_PATH}/road_segments")

# Compute H3 on driver, join back
df_inc_h3 = _add_h3_columns(df_inc, "incident_id", resolutions=[6, 7, 8])

df_inc_geo = (
    df_inc_h3
    .withColumn("geometry_wkt",
        F.concat(F.lit("POINT ("), F.col("longitude"), F.lit(" "), F.col("latitude"), F.lit(")")))
    # Cardinal direction from Charlotte (reference point)
    .withColumn("bearing_from_charlotte",
        F.degrees(F.atan2(
            F.col("longitude") - F.lit(-80.8431),
            F.col("latitude")  - F.lit(35.2271)
        ))
    )
    # Urban/rural proxy: within 30 miles of a major city (Spark-native haversine)
    .withColumn("urban_proximity_score",
        F.least(
            _haversine_expr(F.col("latitude"), F.col("longitude"),
                            F.lit(35.2271), F.lit(-80.8431)),   # Charlotte
            _haversine_expr(F.col("latitude"), F.col("longitude"),
                            F.lit(35.7796), F.lit(-78.6382)),   # Raleigh
            _haversine_expr(F.col("latitude"), F.col("longitude"),
                            F.lit(36.0726), F.lit(-79.7920)),   # Greensboro
        )
    )
    .withColumn("area_type",
        F.when(F.col("urban_proximity_score") <= 10, "Urban")
         .when(F.col("urban_proximity_score") <= 30, "Suburban")
         .otherwise("Rural"))
)

df_inc_geo.write.format("delta").mode("overwrite").option("overwriteSchema","true") \
    .partitionBy("state_code","incident_year") \
    .save(f"{GOLD_PATH}/incidents_geo_enriched")

spark.sql(f"""
    CREATE OR REPLACE TABLE {CATALOG}.dot_geo.incidents_geo_enriched
    AS SELECT * FROM delta.`{GOLD_PATH}/incidents_geo_enriched`
""")
print(f"  ✓ incidents_geo_enriched → {df_inc_geo.count():,} rows")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 – H3 Hexagonal Grid Aggregation (Incident Density Map)
# Resolution 8 ≈ 0.7 km² avg – ideal for city-level safety mapping
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 2: H3 hex grid aggregation …")

df_h3_grid = (
    df_inc_geo
    .filter(F.col("h3_index_r8").isNotNull())
    .groupBy("h3_index_r8","h3_index_r6","state_code")
    .agg(
        F.count("incident_id").alias("incident_count"),
        F.sum("fatalities").alias("total_fatalities"),
        F.sum("injuries").alias("total_injuries"),
        F.avg("severity_score").alias("avg_severity_score"),
        F.sum(F.when(F.col("has_fatality"), 1).otherwise(0)).alias("fatal_incident_count"),
        F.sum(F.when(F.col("incident_type") == "COLLISION", 1).otherwise(0)).alias("collision_count"),
        F.sum(F.when(F.col("incident_type") == "DUI", 1).otherwise(0)).alias("dui_count"),
        F.sum(F.when(F.col("incident_type") == "PEDESTRIAN", 1).otherwise(0)).alias("pedestrian_count"),
        F.sum(F.when(F.col("is_weekend"), 1).otherwise(0)).alias("weekend_incident_count"),
        F.sum(F.when(F.col("incident_hour").between(22, 23) | F.col("incident_hour").between(0, 5), 1).otherwise(0))
          .alias("night_incident_count"),
        F.countDistinct("route_id").alias("routes_in_cell"),
        F.min("incident_datetime").alias("first_incident"),
        F.max("incident_datetime").alias("latest_incident"),
    )
    # Density index: incidents per km² (H3 r8 avg area = 0.7374 km²)
    .withColumn("incident_density_per_km2",
        F.round(F.col("incident_count") / F.lit(0.7374), 2))
    # Severity weighted score
    .withColumn("severity_density",
        F.round(F.col("avg_severity_score") * F.col("incident_count"), 2))
    # Risk tier for choropleth maps
    .withColumn("risk_tier",
        F.when(F.col("total_fatalities") >= 5, "Critical – 5+ fatalities")
         .when(F.col("total_fatalities") >= 2, "High – 2-4 fatalities")
         .when(F.col("incident_count") >= 20,  "Elevated – high volume")
         .when(F.col("incident_count") >= 10,  "Moderate")
         .otherwise("Low"))
    .withColumn("gold_timestamp", F.current_timestamp())
)

df_h3_grid.write.format("delta").mode("overwrite").option("overwriteSchema","true") \
    .partitionBy("state_code") \
    .save(f"{GOLD_PATH}/h3_incident_density")

spark.sql(f"""
    CREATE OR REPLACE TABLE {CATALOG}.dot_geo.h3_incident_density
    AS SELECT * FROM delta.`{GOLD_PATH}/h3_incident_density`
""")
print(f"  ✓ h3_incident_density → {df_h3_grid.count():,} H3 cells")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 – Corridor-Level Spatial Aggregation
# Join incidents → road segments via H3 index proximity
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 3: Corridor spatial join (H3 proximity) …")

# Compute k-ring neighbors on driver (h3 not available in serverless UDFs)
df_inc_disk = _add_h3_disk(df_inc_geo, "incident_id")
df_inc_kring = (
    df_inc_geo.select("incident_id","route_id","severity_score",
                      "fatalities","injuries","incident_type","has_fatality")
    .join(df_inc_disk, on="incident_id", how="inner")
)

# Road segments with their H3 index
df_roads_h3 = df_roads.select("segment_id","route_id","h3_index_r8",
                               "aadt","area_type","functional_class")

# Join: incident's neighbor ring ∩ road segment H3 cell
df_corridor_incidents = (
    df_inc_kring
    .join(
        df_roads_h3.withColumnRenamed("route_id","seg_route_id"),
        df_inc_kring["h3_neighbor"] == df_roads_h3["h3_index_r8"],
        "inner"
    )
    .groupBy("seg_route_id","functional_class","area_type")
    .agg(
        F.countDistinct("incident_id").alias("incidents_on_corridor"),
        F.sum("fatalities").alias("fatalities_on_corridor"),
        F.sum("injuries").alias("injuries_on_corridor"),
        F.avg("severity_score").alias("avg_severity"),
        F.sum(F.when(F.col("has_fatality"), 1).otherwise(0)).alias("fatal_incidents"),
        F.avg("aadt").alias("reference_aadt"),
    )
    .withColumn("incidents_per_100m_vmt",
        F.when(F.col("reference_aadt") > 0,
            F.round(F.col("incidents_on_corridor") /
                    (F.col("reference_aadt") * F.lit(365) / F.lit(1e8)), 4))
         .otherwise(None))
    .withColumn("gold_timestamp", F.current_timestamp())
)

df_corridor_incidents.write.format("delta").mode("overwrite").option("overwriteSchema","true") \
    .save(f"{GOLD_PATH}/corridor_safety_rates")

spark.sql(f"""
    CREATE OR REPLACE TABLE {CATALOG}.dot_geo.corridor_safety_rates
    AS SELECT * FROM delta.`{GOLD_PATH}/corridor_safety_rates`
""")
print(f"  ✓ corridor_safety_rates → {df_corridor_incidents.count():,} corridor segments")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 – Spatial Hotspot Detection (simple density-based clustering)
# Identifies H3 cells that are statistical outliers in incident density
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 4: Hotspot detection …")

from pyspark.sql.window import Window

# Compute z-score of incident density per state
window_state = Window.partitionBy("state_code")

df_hotspots = (
    df_h3_grid
    .withColumn("mean_density",   F.avg("incident_density_per_km2").over(window_state))
    .withColumn("stddev_density", F.stddev("incident_density_per_km2").over(window_state))
    .withColumn("z_score",
        F.when(
            F.col("stddev_density") > 0,
            F.round((F.col("incident_density_per_km2") - F.col("mean_density")) /
                     F.col("stddev_density"), 3)
        ).otherwise(0))
    .withColumn("is_hotspot",     F.col("z_score") >= 1.5)
    .withColumn("is_cold_spot",   F.col("z_score") <= -1.5)
    .withColumn("hotspot_class",
        F.when(F.col("z_score") >= 3.0, "Hot Spot 99% Confidence")
         .when(F.col("z_score") >= 2.0, "Hot Spot 95% Confidence")
         .when(F.col("z_score") >= 1.5, "Hot Spot 90% Confidence")
         .when(F.col("z_score") <= -1.5,"Cold Spot (low activity)")
         .otherwise("Not Significant"))
    .withColumn("gold_timestamp", F.current_timestamp())
)

df_hotspots.write.format("delta").mode("overwrite").option("overwriteSchema","true") \
    .partitionBy("state_code") \
    .save(f"{GOLD_PATH}/incident_hotspots")

spark.sql(f"""
    CREATE OR REPLACE TABLE {CATALOG}.dot_geo.incident_hotspots
    AS SELECT * FROM delta.`{GOLD_PATH}/incident_hotspots`
""")

hotspot_count = df_hotspots.filter(F.col("is_hotspot")).count()
print(f"  ✓ incident_hotspots → {df_hotspots.count():,} cells | {hotspot_count} hotspots identified")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 – Bridge + TAZ Spatial Overlay
# Assign each bridge to a Traffic Analysis Zone using H3 proximity
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 5: Bridge–TAZ overlay …")

# The new 07a writes verified_bridges with pre-computed H3 columns.
# Fall back to Silver + driver-side H3 if that table doesn't exist.
try:
    df_brg = spark.read.format("delta").load(f"{GEO_PATH}/verified_bridges")
    print("  Using verified_bridges from 07a geo bronze layer")
except Exception:
    df_brg_raw = spark.read.format("delta").load(f"{SILVER_PATH}/bridge_inspections")
    df_brg = _add_h3_columns(df_brg_raw, "bridge_id", resolutions=[7])
    df_brg = df_brg.withColumn("geometry_wkt",
        F.concat(F.lit("POINT ("), F.col("longitude"), F.lit(" "), F.col("latitude"), F.lit(")")))
    print("  Using bridge_inspections from Silver layer (07a geo table not found)")

# Ensure h3_index_r7 and geometry_wkt exist regardless of source
if "h3_index_r7" not in df_brg.columns:
    df_brg_tmp = _add_h3_columns(df_brg, "bridge_id", resolutions=[7])
    df_brg = df_brg_tmp
if "geometry_wkt" not in df_brg.columns:
    df_brg = df_brg.withColumn("geometry_wkt",
        F.concat(F.lit("POINT ("), F.col("longitude"), F.lit(" "), F.col("latitude"), F.lit(")")))

df_taz = spark.read.format("delta").load(f"{GEO_PATH}/traffic_analysis_zones")

# New 07a writes `county` (was `county_code`) — handle both for backwards compat
taz_county_col = "county" if "county" in df_taz.columns else "county_code"

df_taz_sel = (
    df_taz.select("taz_id","h3_index_r7","zone_type","population",
                  "employment","vehicle_trips_daily")
          .withColumnRenamed("h3_index_r7","taz_h3")
)

df_brg_taz = (
    df_brg
    .join(
        df_taz_sel,
        df_brg["h3_index_r7"] == df_taz_sel["taz_h3"],
        "left"
    )
    .select(
        "bridge_id","state_code","bridge_type",
        F.col("age_category")    if "age_category"    in df_brg.columns else F.lit(None).alias("age_category"),
        F.col("worst_condition") if "worst_condition" in df_brg.columns else F.lit(None).alias("worst_condition"),
        F.col("risk_score")      if "risk_score"      in df_brg.columns else F.lit(0.0).alias("risk_score"),
        F.col("priority_tier")   if "priority_tier"   in df_brg.columns else F.lit(None).alias("priority_tier"),
        "sufficiency_rating",
        F.col("avg_daily_traffic") if "avg_daily_traffic" in df_brg.columns
            else F.col("aadt").alias("avg_daily_traffic"),
        "latitude","longitude","geometry_wkt","h3_index_r7",
        "taz_id","zone_type",
        F.col("population").alias("taz_population"),
        F.col("employment").alias("taz_employment"),
        F.col("vehicle_trips_daily").alias("taz_vehicle_trips"),
        F.round(
            F.col("taz_population") * (F.col("risk_score") / F.lit(100)), 0
        ).alias("population_at_risk_score"),
    )
    .withColumn("gold_timestamp", F.current_timestamp())
)

df_brg_taz.write.format("delta").mode("overwrite").option("overwriteSchema","true") \
    .partitionBy("state_code") \
    .save(f"{GOLD_PATH}/bridges_with_taz")

spark.sql(f"""
    CREATE OR REPLACE TABLE {CATALOG}.dot_geo.bridges_with_taz
    AS SELECT * FROM delta.`{GOLD_PATH}/bridges_with_taz`
""")
print(f"  ✓ bridges_with_taz → {df_brg_taz.count():,} rows")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 – Work Zone Conflict Analysis
# Identify incidents that occurred inside or adjacent to active work zones
# (using H3 cell overlap as proxy for geometric intersection)
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 6: Work zone conflict analysis …")

df_wz = (
    spark.read.format("delta").load(f"{GEO_PATH}/work_zones")
    .filter(F.col("status") == "Active")
    .select("work_zone_id","route_id","h3_index_r8",
            "zone_type","start_date","end_date","speed_limit_in_zone")
)

df_wz_renamed = (
    df_wz.withColumnRenamed("h3_index_r8","wz_h3")
         .withColumnRenamed("route_id","wz_route")
)

df_wz_incidents = (
    df_inc_geo
    .join(
        df_wz_renamed,
        (df_inc_geo["h3_index_r8"] == df_wz_renamed["wz_h3"]) &
        (df_inc_geo["incident_datetime"] >= df_wz_renamed["start_date"]) &
        (df_inc_geo["incident_datetime"] <= df_wz_renamed["end_date"]),
        "inner"
    )
    .groupBy("work_zone_id","zone_type","wz_route","speed_limit_in_zone")
    .agg(
        F.count("incident_id").alias("wz_incident_count"),
        F.sum("fatalities").alias("wz_fatalities"),
        F.sum("injuries").alias("wz_injuries"),
        F.avg("severity_score").alias("wz_avg_severity"),
    )
    .withColumn("gold_timestamp", F.current_timestamp())
)

df_wz_incidents.write.format("delta").mode("overwrite").option("overwriteSchema","true") \
    .save(f"{GOLD_PATH}/work_zone_incident_conflicts")

spark.sql(f"""
    CREATE OR REPLACE TABLE {CATALOG}.dot_geo.work_zone_incident_conflicts
    AS SELECT * FROM delta.`{GOLD_PATH}/work_zone_incident_conflicts`
""")
print(f"  ✓ work_zone_incident_conflicts → {df_wz_incidents.count():,} work zones with incidents")

print("\n✅  Geospatial analytics complete.")
