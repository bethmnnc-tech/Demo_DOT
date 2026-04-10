# =============================================================================
# NOTEBOOK 7: DOT Geospatial Data – Generation, Processing & Spatial Analytics
# Databricks Notebook | DBR 14.3 LTS ML | Language: Python
#
# Libraries used:
#   pip install shapely geopandas h3 keplergl folium --break-system-packages
#   (or pre-installed on DBR ML runtimes via %pip install)
#
# Description:
#   1. Generate realistic geospatial DOT datasets (roads, intersections,
#      traffic zones, bridge locations, sensor corridors)
#   2. Enrich existing incident / bridge / pavement data with spatial context
#   3. Spatial joins, buffer analysis, hexagonal grid aggregation (H3)
#   4. Hotspot detection using spatial clustering
#   5. Write to Delta with geometry stored as WKT and H3 index columns
# =============================================================================

import subprocess
subprocess.check_call(["pip", "install", "-q", "h3", "shapely"])

import json
import math
import random
import uuid
from datetime import datetime, timedelta
from typing import List, Tuple

import h3
import numpy as np
from shapely.geometry import (
    LineString, MultiLineString, Point, Polygon,
    mapping, shape,
)
from shapely.ops import unary_union

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType, DoubleType, IntegerType, StringType,
    StructField, StructType, TimestampType,
)

import sys

# ── Configuration ────────────────────────────────────────────────────────────
# When run by a job: parameters arrive via sys.argv
# When run interactively: dbutils widgets provide a UI with dev defaults
if len(sys.argv) >= 3:
    BASE_PATH = sys.argv[1]
    CATALOG   = sys.argv[2]
else:
    dbutils.widgets.text("base_path", "/Volumes/main/default/dot_lakehouse")
    dbutils.widgets.text("catalog", "main")
    BASE_PATH = dbutils.widgets.get("base_path")
    CATALOG   = dbutils.widgets.get("catalog")

print(f"  BASE_PATH = {BASE_PATH}")
print(f"  CATALOG   = {CATALOG}")

spark = SparkSession.builder \
    .appName("DOT_GeospatialData") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

GEO_PATH   = f"{BASE_PATH}/bronze/geospatial"
random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    """Return distance in miles between two lat/lon points."""
    R = 3958.8  # Earth radius in miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def bearing(lat1, lon1, lat2, lon2) -> float:
    """Compass bearing in degrees from point1 to point2."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(dlam)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def point_along_bearing(lat, lon, bearing_deg, dist_miles) -> Tuple[float, float]:
    """Move from (lat,lon) by dist_miles in bearing_deg direction."""
    R = 3958.8
    d = dist_miles / R
    b = math.radians(bearing_deg)
    phi1 = math.radians(lat)
    lam1 = math.radians(lon)
    phi2 = math.asin(math.sin(phi1)*math.cos(d) + math.cos(phi1)*math.sin(d)*math.cos(b))
    lam2 = lam1 + math.atan2(math.sin(b)*math.sin(d)*math.cos(phi1),
                              math.cos(d) - math.sin(phi1)*math.sin(phi2))
    return math.degrees(phi2), math.degrees(lam2)

def linestring_to_wkt(coords: List[Tuple]) -> str:
    return LineString(coords).wkt

def polygon_to_wkt(coords: List[Tuple]) -> str:
    return Polygon(coords).wkt

def lat_lon_to_h3(lat, lon, resolution=8) -> str:
    """Convert lat/lon to H3 hexagonal cell index."""
    return h3.latlng_to_cell(lat, lon, resolution)

# ─────────────────────────────────────────────────────────────────────────────
# NORTH CAROLINA ROAD NETWORK – Major corridors (real approximate routes)
# ─────────────────────────────────────────────────────────────────────────────
NC_CORRIDORS = {
    # (route_id, name, start_lat, start_lon, end_lat, end_lon, functional_class, lanes)
    "I-40":     ("I-40",    "Interstate 40 E-W",          35.1054,-84.0037,35.9049,-76.1518, "Interstate", 4),
    "I-85":     ("I-85",    "Interstate 85 NE-SW",         35.0146,-82.0104,36.4547,-79.8197, "Interstate", 4),
    "I-77":     ("I-77",    "Interstate 77 N-S",           35.2271,-80.8431,36.5818,-80.8742, "Interstate", 4),
    "I-26":     ("I-26",    "Interstate 26 (Asheville)",   35.4676,-82.5515,35.5984,-82.0271, "Interstate", 4),
    "US-74":    ("US-74",   "US 74 Charlotte-Wilmington",  35.2271,-80.8431,34.2257,-77.9447, "US Route",   2),
    "US-64":    ("US-64",   "US 64 Raleigh-Murphy",        35.7796,-78.6382,35.1554,-83.9810, "US Route",   2),
    "US-421":   ("US-421",  "US 421 Wilmington-Boone",     34.2257,-77.9447,36.2168,-81.6746, "US Route",   2),
    "NC-12":    ("NC-12",   "NC 12 Outer Banks",           35.9049,-75.6724,36.3404,-75.8328, "State Route",2),
    "NC-54":    ("NC-54",   "NC 54 Chapel Hill Corridor",  35.9132,-79.0558,35.8210,-78.6420, "State Route",2),
    "NC-147":   ("NC-147",  "Durham Freeway",              35.9940,-78.9462,35.9799,-78.8987, "State Route",4),
}

# ─────────────────────────────────────────────────────────────────────────────
# DATASET 1 – ROAD SEGMENT NETWORK
# Each major corridor is divided into ~2-mile segments with rich attributes
# ─────────────────────────────────────────────────────────────────────────────
print("Building road segment network …")

road_segments = []
seg_counter = 1

for route_key, (route_id, route_name, lat1, lon1, lat2, lon2, func_class, lanes) in NC_CORRIDORS.items():

    total_dist = haversine_distance(lat1, lon1, lat2, lon2)
    brng       = bearing(lat1, lon1, lat2, lon2)
    seg_len    = random.uniform(1.5, 3.0)      # segment length in miles
    n_segs     = max(1, int(total_dist / seg_len))

    prev_lat, prev_lon = lat1, lon1
    step = total_dist / n_segs

    for i in range(n_segs):
        seg_lat1, seg_lon1 = prev_lat, prev_lon
        seg_lat2, seg_lon2 = point_along_bearing(seg_lat1, seg_lon1, brng, step)

        # Add slight sinuosity to simulate real roads
        mid_lat = (seg_lat1 + seg_lat2) / 2 + random.uniform(-0.005, 0.005)
        mid_lon = (seg_lon1 + seg_lon2) / 2 + random.uniform(-0.005, 0.005)

        wkt = linestring_to_wkt([(seg_lon1,seg_lat1),(mid_lon,mid_lat),(seg_lon2,seg_lat2)])
        mid_h3 = lat_lon_to_h3(mid_lat, mid_lon, resolution=8)

        road_segments.append((
            f"SEG-{seg_counter:06d}",              # segment_id
            route_id,
            route_name,
            "NC",                                   # state_code
            func_class,
            lanes,
            i + 1,                                  # sequence number
            round(i * step, 2),                     # begin_milepost
            round((i + 1) * step, 2),               # end_milepost
            round(step, 3),                         # length_miles
            round(seg_lat1, 6), round(seg_lon1, 6), # start point
            round(seg_lat2, 6), round(seg_lon2, 6), # end point
            round(mid_lat, 6),  round(mid_lon, 6),  # midpoint
            wkt,                                    # geometry_wkt (LineString)
            mid_h3,                                 # h3_index_r8
            lat_lon_to_h3(mid_lat, mid_lon, 6),     # h3_index_r6 (county-ish)
            random.randint(1, 4),                   # speed_limit_mph × 10 = 10-40 → ×1.25 = 15-50
            random.choice(["Asphalt","Concrete","Composite"]),
            random.choice(["Good","Fair","Poor"]),
            random.randint(5000, 120000),            # aadt
            random.randint(0, 30),                  # truck_pct
            random.choice(["Urban","Suburban","Rural"]),
            datetime.now(),
        ))
        prev_lat, prev_lon = seg_lat2, seg_lon2
        seg_counter += 1

road_schema = StructType([
    StructField("segment_id",         StringType(),    False),
    StructField("route_id",           StringType(),    True),
    StructField("route_name",         StringType(),    True),
    StructField("state_code",         StringType(),    True),
    StructField("functional_class",   StringType(),    True),
    StructField("lanes",              IntegerType(),   True),
    StructField("sequence_num",       IntegerType(),   True),
    StructField("begin_milepost",     DoubleType(),    True),
    StructField("end_milepost",       DoubleType(),    True),
    StructField("length_miles",       DoubleType(),    True),
    StructField("start_lat",          DoubleType(),    True),
    StructField("start_lon",          DoubleType(),    True),
    StructField("end_lat",            DoubleType(),    True),
    StructField("end_lon",            DoubleType(),    True),
    StructField("midpoint_lat",       DoubleType(),    True),
    StructField("midpoint_lon",       DoubleType(),    True),
    StructField("geometry_wkt",       StringType(),    True),  # LineString WKT
    StructField("h3_index_r8",        StringType(),    True),  # ~0.7 km² cell
    StructField("h3_index_r6",        StringType(),    True),  # ~36 km² cell
    StructField("speed_limit_factor", IntegerType(),   True),
    StructField("surface_type",       StringType(),    True),
    StructField("pavement_condition", StringType(),    True),
    StructField("aadt",               IntegerType(),   True),
    StructField("truck_pct",          IntegerType(),   True),
    StructField("area_type",          StringType(),    True),
    StructField("ingestion_timestamp",TimestampType(), False),
])

df_roads = spark.createDataFrame(road_segments, schema=road_schema)
df_roads.write.format("delta").mode("overwrite").option("overwriteSchema","true") \
    .partitionBy("state_code","functional_class") \
    .save(f"{GEO_PATH}/road_segments")
print(f"  ✓ road_segments → {df_roads.count():,} segments across {len(NC_CORRIDORS)} corridors")

# ─────────────────────────────────────────────────────────────────────────────
# DATASET 2 – MAJOR INTERSECTIONS
# Simulates signalized and at-grade intersections along NC corridors
# ─────────────────────────────────────────────────────────────────────────────
print("Building intersections …")

NC_CITIES = [
    ("Charlotte",     35.2271, -80.8431),
    ("Raleigh",       35.7796, -78.6382),
    ("Durham",        35.9940, -78.8987),
    ("Greensboro",    36.0726, -79.7920),
    ("Winston-Salem", 36.0999, -80.2442),
    ("Fayetteville",  35.0527, -78.8784),
    ("Cary",          35.7915, -78.7811),
    ("Wilmington",    34.2257, -77.9447),
    ("Asheville",     35.5951, -82.5515),
    ("High Point",    35.9557, -80.0053),
    ("Concord",       35.4088, -80.5796),
    ("Gastonia",      35.2621, -81.1873),
    ("Chapel Hill",   35.9132, -79.0558),
    ("Boone",         36.2168, -81.6746),
    ("Outer Banks",   35.5585, -75.4665),
]

intersection_types  = ["Signalized","Stop Sign","Roundabout","Interchange","At-Grade Railroad"]
control_types       = ["Traffic Signal","Stop Sign","Yield","Roundabout","Uncontrolled"]

intersections = []
for i, (city, city_lat, city_lon) in enumerate(NC_CITIES):
    n = random.randint(8, 25)   # intersections per city
    for j in range(n):
        lat = city_lat + random.gauss(0, 0.04)
        lon = city_lon + random.gauss(0, 0.06)
        h3_r8 = lat_lon_to_h3(lat, lon, 8)
        intersections.append((
            f"INT-{i*30+j+1:05d}",
            city,
            "NC",
            round(lat, 6),
            round(lon, 6),
            Point(lon, lat).wkt,                    # geometry_wkt (Point)
            h3_r8,
            lat_lon_to_h3(lat, lon, 6),
            random.choice(intersection_types),
            random.choice(control_types),
            random.randint(2, 8),                   # num_approaches
            random.choice([True, False]),           # has_turn_lane
            random.choice([True, False]),           # has_pedestrian_signal
            random.randint(500, 80000),             # entering_volume_daily
            random.randint(0, 50),                  # crash_count_5yr
            random.randint(0, 5),                   # fatal_crash_count_5yr
            random.choice(["High","Moderate","Low"]),  # congestion_level
            datetime.now(),
        ))

intersection_schema = StructType([
    StructField("intersection_id",        StringType(),    False),
    StructField("nearest_city",           StringType(),    True),
    StructField("state_code",             StringType(),    True),
    StructField("latitude",               DoubleType(),    True),
    StructField("longitude",              DoubleType(),    True),
    StructField("geometry_wkt",           StringType(),    True),  # Point WKT
    StructField("h3_index_r8",            StringType(),    True),
    StructField("h3_index_r6",            StringType(),    True),
    StructField("intersection_type",      StringType(),    True),
    StructField("control_type",           StringType(),    True),
    StructField("num_approaches",         IntegerType(),   True),
    StructField("has_turn_lane",          StringType(),    True),
    StructField("has_pedestrian_signal",  StringType(),    True),
    StructField("entering_volume_daily",  IntegerType(),   True),
    StructField("crash_count_5yr",        IntegerType(),   True),
    StructField("fatal_crash_count_5yr",  IntegerType(),   True),
    StructField("congestion_level",       StringType(),    True),
    StructField("ingestion_timestamp",    TimestampType(), False),
])

df_intersections = spark.createDataFrame(intersections, schema=intersection_schema)
df_intersections.write.format("delta").mode("overwrite").option("overwriteSchema","true") \
    .partitionBy("state_code") \
    .save(f"{GEO_PATH}/intersections")
print(f"  ✓ intersections → {df_intersections.count():,} intersections")

# ─────────────────────────────────────────────────────────────────────────────
# DATASET 3 – TRAFFIC ANALYSIS ZONES (TAZ)
# Polygonal zones used in transportation demand modeling
# ─────────────────────────────────────────────────────────────────────────────
print("Building Traffic Analysis Zones …")

def random_polygon_around(lat, lon, radius_deg=0.04, n_vertices=6) -> str:
    """Create a roughly circular polygon (TAZ boundary)."""
    angles = sorted([random.uniform(0, 2*math.pi) for _ in range(n_vertices)])
    coords = []
    for a in angles:
        r = radius_deg * random.uniform(0.6, 1.0)
        coords.append((lon + r*math.cos(a), lat + r*math.sin(a)))
    coords.append(coords[0])  # close ring
    return Polygon(coords).wkt

taz_types    = ["CBD","Urban Residential","Suburban Residential","Industrial","Commercial","Mixed Use","Rural","University","Airport"]

taz_records = []
for i, (city, city_lat, city_lon) in enumerate(NC_CITIES):
    n_zones = random.randint(4, 12)
    for j in range(n_zones):
        lat = city_lat + random.gauss(0, 0.06)
        lon = city_lon + random.gauss(0, 0.08)
        poly_wkt = random_polygon_around(lat, lon, radius_deg=random.uniform(0.02, 0.06))
        taz_records.append((
            f"TAZ-{i*15+j+1:04d}",
            city,
            "NC",
            f"COUNTY_{random.randint(1,100):03d}",
            round(lat, 6),
            round(lon, 6),                          # centroid
            poly_wkt,                               # polygon boundary WKT
            lat_lon_to_h3(lat, lon, 7),             # H3 at resolution 7 (~5 km²)
            random.choice(taz_types),
            round(random.uniform(0.1, 25), 2),      # area_sq_miles
            random.randint(100, 80000),              # population
            random.randint(50, 40000),               # employment
            random.randint(10, 20000),               # households
            random.randint(200, 150000),             # vehicle_trips_daily
            random.randint(50, 50000),               # transit_trips_daily
            round(random.uniform(0.5, 5.0), 2),     # avg_trip_length_miles
            round(random.uniform(0, 100), 1),       # pct_zero_car_households
            datetime.now(),
        ))

taz_schema = StructType([
    StructField("taz_id",                 StringType(),    False),
    StructField("city",                   StringType(),    True),
    StructField("state_code",             StringType(),    True),
    StructField("county_code",            StringType(),    True),
    StructField("centroid_lat",           DoubleType(),    True),
    StructField("centroid_lon",           DoubleType(),    True),
    StructField("geometry_wkt",           StringType(),    True),  # Polygon WKT
    StructField("h3_index_r7",            StringType(),    True),
    StructField("zone_type",              StringType(),    True),
    StructField("area_sq_miles",          DoubleType(),    True),
    StructField("population",             IntegerType(),   True),
    StructField("employment",             IntegerType(),   True),
    StructField("households",             IntegerType(),   True),
    StructField("vehicle_trips_daily",    IntegerType(),   True),
    StructField("transit_trips_daily",    IntegerType(),   True),
    StructField("avg_trip_length_miles",  DoubleType(),    True),
    StructField("pct_zero_car_households",DoubleType(),    True),
    StructField("ingestion_timestamp",    TimestampType(), False),
])

df_taz = spark.createDataFrame(taz_records, schema=taz_schema)
df_taz.write.format("delta").mode("overwrite").option("overwriteSchema","true") \
    .partitionBy("state_code") \
    .save(f"{GEO_PATH}/traffic_analysis_zones")
print(f"  ✓ traffic_analysis_zones → {df_taz.count():,} zones")

# ─────────────────────────────────────────────────────────────────────────────
# DATASET 4 – TRAFFIC SENSOR LOCATIONS
# Fixed-point sensors: loop detectors, radar, CCTV, weigh-in-motion
# ─────────────────────────────────────────────────────────────────────────────
print("Building sensor locations …")

sensor_types = ["Inductive Loop","Radar","Video/CCTV","Weigh-in-Motion","Bluetooth","Lidar","Pneumatic Tube"]
sensor_status = ["Active","Offline","Maintenance","Decommissioned"]

sensors = []
for seg_id, route_id, lat, lon, h3_idx, aadt in (
    [(r[0], r[1], r[14], r[15], r[17], r[22]) for r in road_segments]
):
    if random.random() > 0.3:    # ~70% of segments have a sensor
        s_type = random.choice(sensor_types)
        sensors.append((
            f"SEN-{uuid.uuid4().hex[:8].upper()}",
            seg_id,
            route_id,
            "NC",
            round(lat + random.uniform(-0.001, 0.001), 6),
            round(lon + random.uniform(-0.001, 0.001), 6),
            Point(lon, lat).wkt,
            h3_idx,
            s_type,
            random.choice(sensor_status),
            random.randint(2010, 2024),              # install_year
            random.randint(1, 6),                   # num_lanes_monitored
            random.choice([True, False]),            # bidirectional
            round(random.uniform(0.85, 1.0), 3),    # uptime_pct (last 30d)
            aadt,
            random.randint(1, 60),                  # data_interval_sec
            random.choice(["5G","LTE","Fiber","WiFi","Cellular"]),  # comms_type
            datetime.now(),
        ))

sensor_schema = StructType([
    StructField("sensor_id",           StringType(),    False),
    StructField("segment_id",          StringType(),    True),
    StructField("route_id",            StringType(),    True),
    StructField("state_code",          StringType(),    True),
    StructField("latitude",            DoubleType(),    True),
    StructField("longitude",           DoubleType(),    True),
    StructField("geometry_wkt",        StringType(),    True),
    StructField("h3_index_r8",         StringType(),    True),
    StructField("sensor_type",         StringType(),    True),
    StructField("operational_status",  StringType(),    True),
    StructField("install_year",        IntegerType(),   True),
    StructField("num_lanes_monitored", IntegerType(),   True),
    StructField("bidirectional",       StringType(),    True),
    StructField("uptime_pct",          DoubleType(),    True),
    StructField("reference_aadt",      IntegerType(),   True),
    StructField("data_interval_sec",   IntegerType(),   True),
    StructField("comms_type",          StringType(),    True),
    StructField("ingestion_timestamp", TimestampType(), False),
])

df_sensors = spark.createDataFrame(sensors, schema=sensor_schema)
df_sensors.write.format("delta").mode("overwrite").option("overwriteSchema","true") \
    .partitionBy("state_code","sensor_type") \
    .save(f"{GEO_PATH}/sensor_locations")
print(f"  ✓ sensor_locations → {df_sensors.count():,} sensors")

# ─────────────────────────────────────────────────────────────────────────────
# DATASET 5 – WORK ZONES (active construction polygons)
# ─────────────────────────────────────────────────────────────────────────────
print("Building work zones …")

wz_types    = ["Road Closure","Lane Reduction","Shoulder Closure","Full Closure","Detour"]
wz_status   = ["Active","Planned","Completed"]

work_zones = []
for i in range(120):
    city_lat, city_lon = random.choice([(c[1], c[2]) for c in NC_CITIES])
    lat = city_lat + random.gauss(0, 0.1)
    lon = city_lon + random.gauss(0, 0.15)
    start = datetime(2024,1,1) + timedelta(days=random.randint(0,364))
    end   = start + timedelta(days=random.randint(7, 180))
    poly_wkt = random_polygon_around(lat, lon, radius_deg=0.01, n_vertices=5)
    work_zones.append((
        f"WZ-{i+1:04d}",
        "NC",
        f"COUNTY_{random.randint(1,100):03d}",
        random.choice(list(NC_CORRIDORS.keys())),
        round(lat, 6), round(lon, 6),
        poly_wkt,
        lat_lon_to_h3(lat, lon, 8),
        random.choice(wz_types),
        random.choice(wz_status),
        start,
        end,
        random.randint(1, 4),                       # lanes_closed
        round(random.uniform(0.1, 5.0), 2),         # zone_length_miles
        random.randint(5, 65),                      # speed_limit_in_zone
        random.choice([True, False]),               # flagging_required
        round(random.uniform(50000, 20000000), 0),  # project_cost_usd
        random.choice(["NCDOT","Contractor","County","Utility"]),
        datetime.now(),
    ))

wz_schema = StructType([
    StructField("work_zone_id",          StringType(),    False),
    StructField("state_code",            StringType(),    True),
    StructField("county_code",           StringType(),    True),
    StructField("route_id",              StringType(),    True),
    StructField("centroid_lat",          DoubleType(),    True),
    StructField("centroid_lon",          DoubleType(),    True),
    StructField("geometry_wkt",          StringType(),    True),  # Polygon WKT
    StructField("h3_index_r8",           StringType(),    True),
    StructField("zone_type",             StringType(),    True),
    StructField("status",                StringType(),    True),
    StructField("start_date",            TimestampType(), True),
    StructField("end_date",              TimestampType(), True),
    StructField("lanes_closed",          IntegerType(),   True),
    StructField("zone_length_miles",     DoubleType(),    True),
    StructField("speed_limit_in_zone",   IntegerType(),   True),
    StructField("flagging_required",     StringType(),    True),
    StructField("project_cost_usd",      DoubleType(),    True),
    StructField("responsible_agency",    StringType(),    True),
    StructField("ingestion_timestamp",   TimestampType(), False),
])

df_wz = spark.createDataFrame(work_zones, schema=wz_schema)
df_wz.write.format("delta").mode("overwrite").option("overwriteSchema","true") \
    .partitionBy("state_code") \
    .save(f"{GEO_PATH}/work_zones")
print(f"  ✓ work_zones → {df_wz.count():,} zones")

print("\n✅  Geospatial Bronze datasets written.")
print(f"   Path: {GEO_PATH}/")
