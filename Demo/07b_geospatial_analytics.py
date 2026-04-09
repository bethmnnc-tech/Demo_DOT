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
import numpy as np
from typing import List

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, DoubleType, IntegerType, ArrayType

spark = SparkSession.builder.appName("DOT_GeospatialAnalytics").getOrCreate()

BASE_PATH  = "/Volumes/main/default/dot_lakehouse"
GEO_PATH   = f"{BASE_PATH}/bronze/geospatial"
SILVER_PATH= f"{BASE_PATH}/silver"
GOLD_PATH  = f"{BASE_PATH}/gold/geospatial"

spark.sql("CREATE DATABASE IF NOT EXISTS dot_geo")

# ─────────────────────────────────────────────────────────────────────────────
# UDFs – H3 spatial indexing
# ─────────────────────────────────────────────────────────────────────────────

@F.udf(StringType())
def udf_h3_r8(lat, lon):
    """Assign H3 cell at resolution 8 (~0.7 km² avg area)."""
    if lat is None or lon is None:
        return None
    try:
        return h3.latlng_to_cell(float(lat), float(lon), 8)
    except Exception:
        return None

@F.udf(StringType())
def udf_h3_r6(lat, lon):
    """Assign H3 cell at resolution 6 (~36 km² avg area — county-ish)."""
    if lat is None or lon is None:
        return None
    try:
        return h3.latlng_to_cell(float(lat), float(lon), 6)
    except Exception:
        return None

@F.udf(StringType())
def udf_h3_r7(lat, lon):
    """Assign H3 cell at resolution 7 (~5 km² avg area — sub-district)."""
    if lat is None or lon is None:
        return None
    try:
        return h3.latlng_to_cell(float(lat), float(lon), 7)
    except Exception:
        return None

@F.udf(DoubleType())
def udf_haversine(lat1, lon1, lat2, lon2):
    """Haversine distance in miles between two lat/lon pairs."""
    if any(v is None for v in [lat1, lon1, lat2, lon2]):
        return None
    R = 3958.8
    phi1, phi2 = math.radians(float(lat1)), math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlam = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return round(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a)), 4)

@F.udf(ArrayType(StringType()))
def udf_h3_disk(lat, lon, k_rings=2):
    """Return the H3 cells within k rings of a point (for proximity lookups)."""
    if lat is None or lon is None:
        return []
    try:
        center = h3.latlng_to_cell(float(lat), float(lon), 8)
        return list(h3.grid_disk(center, k_rings))
    except Exception:
        return []

@F.udf(StringType())
def udf_point_wkt(lat, lon):
    if lat is None or lon is None:
        return None
    return f"POINT ({lon} {lat})"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 – Enrich Silver Incidents with Geospatial Attributes
# ─────────────────────────────────────────────────────────────────────────────
print("Step 1: Enriching incidents with H3 + geometry …")

df_inc = spark.read.format("delta").load(f"{SILVER_PATH}/traffic_incidents")
df_roads = spark.read.format("delta").load(f"{GEO_PATH}/road_segments")

df_inc_geo = (
    df_inc
    .withColumn("h3_index_r8", udf_h3_r8(F.col("latitude"), F.col("longitude")))
    .withColumn("h3_index_r6", udf_h3_r6(F.col("latitude"), F.col("longitude")))
    .withColumn("h3_index_r7", udf_h3_r7(F.col("latitude"), F.col("longitude")))
    .withColumn("geometry_wkt", udf_point_wkt(F.col("latitude"), F.col("longitude")))
    # Cardinal direction from Charlotte (reference point)
    .withColumn("bearing_from_charlotte",
        F.degrees(F.atan2(
            F.col("longitude") - F.lit(-80.8431),
            F.col("latitude")  - F.lit(35.2271)
        ))
    )
    # Urban/rural proxy: within 30 miles of a major city
    .withColumn("urban_proximity_score",
        F.least(
            udf_haversine(F.col("latitude"), F.col("longitude"),
                          F.lit(35.2271), F.lit(-80.8431)),   # Charlotte
            udf_haversine(F.col("latitude"), F.col("longitude"),
                          F.lit(35.7796), F.lit(-78.6382)),   # Raleigh
            udf_haversine(F.col("latitude"), F.col("longitude"),
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
    CREATE OR REPLACE TABLE dot_geo.incidents_geo_enriched
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
    CREATE OR REPLACE TABLE dot_geo.h3_incident_density
    AS SELECT * FROM delta.`{GOLD_PATH}/h3_incident_density`
""")
print(f"  ✓ h3_incident_density → {df_h3_grid.count():,} H3 cells")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 – Corridor-Level Spatial Aggregation
# Join incidents → road segments via H3 index proximity
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 3: Corridor spatial join (H3 proximity) …")

# Explode k-ring neighbors for proximity matching
df_inc_kring = (
    df_inc_geo
    .withColumn("nearby_h3_cells", udf_h3_disk(F.col("latitude"), F.col("longitude")))
    .withColumn("h3_neighbor", F.explode("nearby_h3_cells"))
    .select("incident_id","route_id","h3_neighbor","severity_score",
            "fatalities","injuries","incident_type","has_fatality")
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
    CREATE OR REPLACE TABLE dot_geo.corridor_safety_rates
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
    # Gi* statistic approximation using ring neighbors (spatial lag)
    .withColumn("gold_timestamp", F.current_timestamp())
)

df_hotspots.write.format("delta").mode("overwrite").option("overwriteSchema","true") \
    .partitionBy("state_code") \
    .save(f"{GOLD_PATH}/incident_hotspots")

spark.sql(f"""
    CREATE OR REPLACE TABLE dot_geo.incident_hotspots
    AS SELECT * FROM delta.`{GOLD_PATH}/incident_hotspots`
""")

hotspot_count = df_hotspots.filter(F.col("is_hotspot")).count()
print(f"  ✓ incident_hotspots → {df_hotspots.count():,} cells | {hotspot_count} hotspots identified")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 – Bridge + TAZ Spatial Overlay
# Assign each bridge to a Traffic Analysis Zone using H3 proximity
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 5: Bridge–TAZ overlay …")

df_brg = (
    spark.read.format("delta").load(f"{SILVER_PATH}/bridge_inspections")
    .withColumn("h3_index_r7", udf_h3_r7(F.col("latitude"), F.col("longitude")))
    .withColumn("geometry_wkt", udf_point_wkt(F.col("latitude"), F.col("longitude")))
)

df_taz = spark.read.format("delta").load(f"{GEO_PATH}/traffic_analysis_zones")

df_taz_renamed = (
    df_taz.select("taz_id","h3_index_r7","zone_type","population","employment","vehicle_trips_daily")
          .withColumnRenamed("h3_index_r7","taz_h3")
)

df_brg_taz = (
    df_brg
    .join(
        df_taz_renamed,
        df_brg["h3_index_r7"] == df_taz_renamed["taz_h3"],
        "left"
    )
    .select(
        "bridge_id","state_code","bridge_type","age_category","worst_condition",
        "risk_score","priority_tier","sufficiency_rating","avg_daily_traffic",
        "latitude","longitude","geometry_wkt","h3_index_r7",
        "taz_id","zone_type",
        F.col("population").alias("taz_population"),
        F.col("employment").alias("taz_employment"),
        F.col("vehicle_trips_daily").alias("taz_vehicle_trips"),
        # Population at risk (weighted by bridge ADT and risk score)
        F.round(
            F.col("population") * (F.col("risk_score") / F.lit(100)), 0
        ).alias("population_at_risk_score"),
    )
    .withColumn("gold_timestamp", F.current_timestamp())
)

df_brg_taz.write.format("delta").mode("overwrite").option("overwriteSchema","true") \
    .partitionBy("state_code") \
    .save(f"{GOLD_PATH}/bridges_with_taz")

spark.sql(f"""
    CREATE OR REPLACE TABLE dot_geo.bridges_with_taz
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
    .select("work_zone_id","route_id","h3_index_r8","zone_type","start_date","end_date","speed_limit_in_zone")
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
    CREATE OR REPLACE TABLE dot_geo.work_zone_incident_conflicts
    AS SELECT * FROM delta.`{GOLD_PATH}/work_zone_incident_conflicts`
""")
print(f"  ✓ work_zone_incident_conflicts → {df_wz_incidents.count():,} work zones with incidents")

print("\n✅  Geospatial analytics complete.")
