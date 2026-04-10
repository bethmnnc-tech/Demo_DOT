# =============================================================================
# NOTEBOOK 1: DOT Sample Data Generator
# Databricks Notebook | Cluster: Standard | Language: Python
# Description: Generates realistic Department of Transportation sample datasets
#              and saves them to the Databricks DBFS / Delta Lake Bronze layer.
# =============================================================================

# ── Imports ──────────────────────────────────────────────────────────────────
import random
import sys
import uuid
from datetime import datetime, timedelta

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType, IntegerType, StringType, StructField, StructType, TimestampType,
)

spark = SparkSession.builder.appName("DOT_SampleDataGenerator").getOrCreate()

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

NUM_RECORDS = 50_000                         # scale up as needed
SEED        = 42
random.seed(SEED)

print(f"Generating {NUM_RECORDS:,} records per domain …")
print(f"  BASE_PATH = {BASE_PATH}")
print(f"  CATALOG   = {CATALOG}")

# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN 1 – TRAFFIC INCIDENTS
# Source analog: FHWA / state DOT crash reporting systems
# ─────────────────────────────────────────────────────────────────────────────
incident_types   = ["Collision","Pedestrian","Bicycle","Weather","Road Hazard","DUI","Hit-and-Run","Work Zone"]
severity_levels  = ["Fatal","Serious Injury","Minor Injury","Property Damage Only","Unknown"]
road_conditions  = ["Dry","Wet","Snow/Ice","Fog","Construction","Flooded"]
states           = ["NC","VA","SC","TN","GA","FL","TX","CA","NY","IL"]
weather_types    = ["Clear","Rain","Snow","Fog","Wind","Hail"]

def rand_coord_nc():
    """Return a lat/lon roughly within North Carolina."""
    return round(random.uniform(33.8, 36.6), 6), round(random.uniform(-84.3, -75.4), 6)

def rand_date(start_year=2020, end_year=2024):
    start = datetime(start_year, 1, 1)
    delta = (datetime(end_year, 12, 31) - start).days
    return start + timedelta(days=random.randint(0, delta),
                             hours=random.randint(0, 23),
                             minutes=random.randint(0, 59))

incidents = []
for _ in range(NUM_RECORDS):
    lat, lon = rand_coord_nc()
    dt = rand_date()
    incidents.append((
        str(uuid.uuid4()),                          # incident_id
        random.choice(incident_types),              # incident_type
        random.choice(severity_levels),             # severity
        dt,                                         # incident_datetime
        random.choice(states),                      # state_code
        f"COUNTY_{random.randint(1,100):03d}",      # county_code
        f"ROUTE_{random.randint(1,500)}",           # route_id
        float(round(random.uniform(0, 100), 2)),    # milepost
        lat, lon,                                   # latitude, longitude
        random.choice(road_conditions),             # road_condition
        random.choice(weather_types),               # weather_condition
        random.randint(0, 6),                       # fatalities
        random.randint(0, 15),                      # injuries
        random.randint(1, 8),                       # vehicles_involved
        random.choice(["Reported","Under Review","Closed"]),  # status
        datetime.now(),                             # ingestion_timestamp
    ))

incident_schema = StructType([
    StructField("incident_id",         StringType(),    False),
    StructField("incident_type",       StringType(),    True),
    StructField("severity",            StringType(),    True),
    StructField("incident_datetime",   TimestampType(), True),
    StructField("state_code",          StringType(),    True),
    StructField("county_code",         StringType(),    True),
    StructField("route_id",            StringType(),    True),
    StructField("milepost",            DoubleType(),    True),
    StructField("latitude",            DoubleType(),    True),
    StructField("longitude",           DoubleType(),    True),
    StructField("road_condition",      StringType(),    True),
    StructField("weather_condition",   StringType(),    True),
    StructField("fatalities",          IntegerType(),   True),
    StructField("injuries",            IntegerType(),   True),
    StructField("vehicles_involved",   IntegerType(),   True),
    StructField("status",              StringType(),    True),
    StructField("ingestion_timestamp", TimestampType(), False),
])

df_incidents = spark.createDataFrame(incidents, schema=incident_schema)
df_incidents.write.format("delta").mode("overwrite").option("overwriteSchema","true") \
    .partitionBy("state_code","incident_type") \
    .save(f"{BASE_PATH}/bronze/traffic_incidents")

print(f"  ✓ traffic_incidents  → {df_incidents.count():,} rows")

# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN 2 – BRIDGE INSPECTIONS
# Source analog: NBI (National Bridge Inventory)
# ─────────────────────────────────────────────────────────────────────────────
bridge_types     = ["Girder","Truss","Arch","Suspension","Cable-Stayed","Culvert","Box Beam","Moveable"]
material_types   = ["Concrete","Steel","Prestressed Concrete","Timber","Masonry","Aluminum"]
owner_types      = ["State DOT","Federal","County","City/Municipality","Railroad","Other"]
condition_codes  = ["N - Not Applicable","G - Good","F - Fair","P - Poor","C - Critical"]

bridges = []
for i in range(10_000):
    lat, lon = rand_coord_nc()
    year_built = random.randint(1930, 2020)
    last_insp  = rand_date(2018, 2024)
    bridges.append((
        f"BRG-{i+1:06d}",                          # bridge_id
        f"Bridge over {random.choice(['River','Creek','Highway','Railroad'])} {i}",
        random.choice(states),
        f"COUNTY_{random.randint(1,100):03d}",
        f"ROUTE_{random.randint(1,500)}",
        float(round(random.uniform(0, 100), 2)),
        lat, lon,
        random.choice(bridge_types),
        random.choice(material_types),
        random.choice(owner_types),
        year_built,
        round(random.uniform(20, 600), 1),          # span_length_ft
        round(random.uniform(10, 80), 1),           # deck_width_ft
        round(random.uniform(5, 80), 1),            # vertical_clearance_ft
        random.randint(500, 80_000),                # avg_daily_traffic
        last_insp,
        random.choice(condition_codes),             # deck_condition
        random.choice(condition_codes),             # superstructure_condition
        random.choice(condition_codes),             # substructure_condition
        round(random.uniform(0, 100), 1),           # sufficiency_rating (0-100)
        random.choice([True, False]),               # structurally_deficient
        random.choice([True, False]),               # functionally_obsolete
        round(random.uniform(0, 500), 0),           # estimated_repair_cost_k
        datetime.now(),
    ))

bridge_schema = StructType([
    StructField("bridge_id",                  StringType(),    False),
    StructField("bridge_name",                StringType(),    True),
    StructField("state_code",                 StringType(),    True),
    StructField("county_code",                StringType(),    True),
    StructField("route_id",                   StringType(),    True),
    StructField("milepost",                   DoubleType(),    True),
    StructField("latitude",                   DoubleType(),    True),
    StructField("longitude",                  DoubleType(),    True),
    StructField("bridge_type",                StringType(),    True),
    StructField("material_type",              StringType(),    True),
    StructField("owner_type",                 StringType(),    True),
    StructField("year_built",                 IntegerType(),   True),
    StructField("span_length_ft",             DoubleType(),    True),
    StructField("deck_width_ft",              DoubleType(),    True),
    StructField("vertical_clearance_ft",      DoubleType(),    True),
    StructField("avg_daily_traffic",          IntegerType(),   True),
    StructField("last_inspection_date",       TimestampType(), True),
    StructField("deck_condition",             StringType(),    True),
    StructField("superstructure_condition",   StringType(),    True),
    StructField("substructure_condition",     StringType(),    True),
    StructField("sufficiency_rating",         DoubleType(),    True),
    StructField("structurally_deficient",     StringType(),    True),
    StructField("functionally_obsolete",      StringType(),    True),
    StructField("estimated_repair_cost_k",    DoubleType(),    True),
    StructField("ingestion_timestamp",        TimestampType(), False),
])

df_bridges = spark.createDataFrame(bridges, schema=bridge_schema)
df_bridges.write.format("delta").mode("overwrite").option("overwriteSchema","true") \
    .partitionBy("state_code") \
    .save(f"{BASE_PATH}/bronze/bridge_inspections")

print(f"  ✓ bridge_inspections → {df_bridges.count():,} rows")

# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN 3 – VEHICLE REGISTRATION & FLEET
# Source analog: FMCSA / state DMV
# ─────────────────────────────────────────────────────────────────────────────
vehicle_classes = ["Passenger Car","Light Truck","Medium Truck","Heavy Truck",
                   "Bus","Motorcycle","Trailer","Special Equipment"]
fuel_types      = ["Gasoline","Diesel","Electric","Hybrid","CNG","Hydrogen","LPG"]
makes           = ["Ford","Chevrolet","Toyota","Honda","Freightliner","Peterbilt",
                   "Kenworth","Volvo","Mercedes","RAM","Tesla","Rivian"]

vehicles = []
for i in range(NUM_RECORDS):
    reg_date = rand_date(2015, 2024)
    vehicles.append((
        f"VIN{random.randint(10**16, 10**17-1)}",  # vin
        random.choice(states),
        f"COUNTY_{random.randint(1,100):03d}",
        random.choice(vehicle_classes),
        random.choice(makes),
        random.randint(2000, 2024),                 # model_year
        random.choice(fuel_types),
        round(random.uniform(8000, 80000), 0),      # gvwr_lbs
        round(random.uniform(1, 300), 1),           # horsepower
        reg_date,
        reg_date + timedelta(days=365),             # expiration_date
        random.choice(["Active","Expired","Suspended","Revoked"]),
        random.choice([True, False]),               # commercial_vehicle
        random.choice([True, False]),               # hazmat_certified
        datetime.now(),
    ))

vehicle_schema = StructType([
    StructField("vin",                StringType(),    False),
    StructField("state_code",         StringType(),    True),
    StructField("county_code",        StringType(),    True),
    StructField("vehicle_class",      StringType(),    True),
    StructField("make",               StringType(),    True),
    StructField("model_year",         IntegerType(),   True),
    StructField("fuel_type",          StringType(),    True),
    StructField("gvwr_lbs",           DoubleType(),    True),
    StructField("horsepower",         DoubleType(),    True),
    StructField("registration_date",  TimestampType(), True),
    StructField("expiration_date",    TimestampType(), True),
    StructField("registration_status",StringType(),    True),
    StructField("commercial_vehicle", StringType(),    True),
    StructField("hazmat_certified",   StringType(),    True),
    StructField("ingestion_timestamp",TimestampType(), False),
])

df_vehicles = spark.createDataFrame(vehicles, schema=vehicle_schema)
df_vehicles.write.format("delta").mode("overwrite").option("overwriteSchema","true") \
    .partitionBy("state_code","vehicle_class") \
    .save(f"{BASE_PATH}/bronze/vehicle_registrations")

print(f"  ✓ vehicle_registrations → {df_vehicles.count():,} rows")

# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN 4 – ROAD PAVEMENT CONDITIONS
# Source analog: FHWA HPMS (Highway Performance Monitoring System)
# ─────────────────────────────────────────────────────────────────────────────
pavement_types   = ["Asphalt","Concrete","Composite","Gravel","Chip Seal"]
functional_class = ["Interstate","US Route","State Route","County Road","Local Road","Ramp"]

pavements = []
for i in range(20_000):
    insp_date = rand_date(2019, 2024)
    pavements.append((
        f"SEG-{i+1:07d}",
        random.choice(states),
        f"COUNTY_{random.randint(1,100):03d}",
        f"ROUTE_{random.randint(1,500)}",
        round(random.uniform(0, 100), 2),           # begin_milepost
        round(random.uniform(0, 5), 2),             # segment_length_mi
        random.choice(functional_class),
        random.choice(pavement_types),
        random.randint(1960, 2023),                 # year_constructed
        random.randint(0, 100),                     # IRI (International Roughness Index)
        round(random.uniform(0, 100), 1),           # PSI (Present Serviceability Index)
        round(random.uniform(0, 100), 1),           # PCR (Pavement Condition Rating)
        round(random.uniform(0, 30), 1),            # cracking_percent
        round(random.uniform(0, 10), 2),            # rutting_in
        random.choice(["Good","Fair","Poor","Very Poor"]),
        random.randint(500, 120_000),               # aadt (Annual Average Daily Traffic)
        random.randint(0, 50),                      # truck_percent
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

print(f"  ✓ pavement_conditions → {df_pavements.count():,} rows")

print("\n✅  Bronze layer generation complete.")
print(f"   Root path: {BASE_PATH}/bronze/")
