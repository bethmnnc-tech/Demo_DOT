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

# ── NC boundary polygon for point-in-polygon check ──────────────────────────
_NC_POLY = [
    (-84.322,35.216),(-84.29,35.226),(-84.09,35.247),(-83.873,35.243),
    (-83.671,35.278),(-83.499,35.564),(-83.253,35.672),(-82.994,35.773),
    (-82.775,35.998),(-82.595,36.062),(-82.409,36.084),(-82.032,36.12),
    (-81.909,36.304),(-81.723,36.354),(-81.677,36.589),(-80.838,36.562),
    (-80.613,36.557),(-80.295,36.544),(-79.892,36.541),(-79.511,36.541),
    (-78.91,36.541),(-78.324,36.544),(-77.768,36.544),(-77.176,36.547),
    (-76.562,36.55),(-76.033,36.55),(-75.868,36.551),(-75.867,36.008),
    (-75.745,35.868),(-75.641,35.559),(-75.484,35.261),(-75.528,35.186),
    (-75.722,35.065),(-75.94,34.807),(-76.06,34.654),(-76.471,34.541),
    (-76.683,34.558),(-76.944,34.499),(-77.21,34.601),(-77.518,34.439),
    (-77.751,34.285),(-77.872,34.14),(-77.923,33.939),(-78.01,33.858),
    (-78.554,33.861),(-79.072,34.3),(-79.458,34.631),(-79.667,34.801),
    (-79.928,34.808),(-80.321,34.814),(-80.561,34.817),(-80.797,34.82),
    (-80.782,34.936),(-80.935,35.108),(-81.044,35.149),(-81.043,35.265),
    (-81.058,35.363),(-82.354,35.199),(-82.897,35.058),(-83.109,35.001),
    (-83.618,34.987),(-84.322,35.216),
]

def _point_in_nc(lat, lon):
    """Ray-casting point-in-polygon test."""
    x, y = lon, lat
    n = len(_NC_POLY)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = _NC_POLY[i]
        xj, yj = _NC_POLY[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

# ── City/highway cluster definitions ────────────────────────────────────────
# (name, lat, lon, weight, spread_deg) — weight controls % of points
_NC_CLUSTERS = [
    # Major metros (60% of incidents)
    ("Charlotte",       35.227, -80.843, 0.18, 0.25),
    ("Raleigh",         35.780, -78.640, 0.14, 0.20),
    ("Greensboro",      36.073, -79.792, 0.08, 0.15),
    ("Durham",          35.994, -78.899, 0.06, 0.12),
    ("Winston-Salem",   36.100, -80.244, 0.05, 0.12),
    ("Fayetteville",    35.053, -78.878, 0.05, 0.15),
    ("Wilmington",      34.226, -77.945, 0.04, 0.15),
    # Mid-size cities (15%)
    ("Asheville",       35.595, -82.551, 0.04, 0.18),
    ("Gastonia",        35.262, -81.187, 0.02, 0.10),
    ("Jacksonville",    34.754, -77.430, 0.02, 0.10),
    ("Greenville",      35.613, -77.366, 0.02, 0.12),
    ("Hickory",         35.733, -81.341, 0.01, 0.10),
    # Highway corridors (25%)
    ("I-40 Piedmont",   35.85,  -79.20,  0.07, 0.40),   # long E-W spread
    ("I-85 Corridor",   35.55,  -80.20,  0.06, 0.35),
    ("I-95 Eastern",    35.30,  -77.90,  0.05, 0.50),
    ("US-74 Southern",  35.10,  -79.50,  0.04, 0.45),
    ("I-77 North",      35.60,  -80.85,  0.03, 0.25),
    ("I-26 Mountains",  35.40,  -82.50,  0.04, 0.30),
]

# Precompute cumulative weights
_cum_weights = []
_cum = 0.0
for c in _NC_CLUSTERS:
    _cum += c[3]
    _cum_weights.append(_cum)

def rand_coord_nc():
    """Return a lat/lon clustered around NC cities/highways, guaranteed inside NC."""
    for _ in range(50):  # retry until inside NC
        # Pick a cluster by weight
        r = random.random() * _cum_weights[-1]
        idx = 0
        for i, cw in enumerate(_cum_weights):
            if r <= cw:
                idx = i
                break
        name, clat, clon, w, spread = _NC_CLUSTERS[idx]
        lat = round(random.gauss(clat, spread * 0.4), 6)
        lon = round(random.gauss(clon, spread * 0.6), 6)
        if _point_in_nc(lat, lon):
            return lat, lon
    # Fallback: Charlotte center
    return 35.227, -80.843

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
        "NC",                                       # state_code (all incidents at NC coords)
        random.choice(["Mecklenburg","Wake","Guilford","Durham","Forsyth","Cumberland",
                       "Buncombe","New Hanover","Gaston","Onslow","Pitt","Catawba",
                       "Cabarrus","Union","Johnston","Randolph","Davidson","Rowan",
                       "Iredell","Alamance","Robeson","Wayne","Harnett","Craven",
                       "Moore","Lee","Dare","Brunswick","Henderson","Surry"]),
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
