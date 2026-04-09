import os
from flask import Flask, jsonify, send_from_directory

app = Flask(__name__, static_folder="static")

# ── Serve the map dashboard ──────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "dot_geospatial_nc_demo.html")

# ── Health check (Databricks Apps requires this) ─────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok"})

# ── Live data API endpoints (optional – connect to Delta tables) ─────────────
# Uncomment these once your cluster is attached and Gold tables are available.

# from pyspark.sql import SparkSession
# spark = SparkSession.builder.getOrCreate()

# @app.route("/api/hotspots")
# def hotspots():
#     df = spark.sql("""
#         SELECT h3_index_r8, incident_count, total_fatalities,
#                avg_severity_score, hotspot_class
#         FROM dot_geo.incident_hotspots
#         WHERE is_hotspot = TRUE
#         LIMIT 2000
#     """)
#     return jsonify(df.toPandas().to_dict(orient="records"))

# @app.route("/api/bridges")
# def bridges():
#     df = spark.sql("""
#         SELECT latitude, longitude, risk_score,
#                priority_tier, year_built
#         FROM dot_silver.bridge_inspections
#     """)
#     return jsonify(df.toPandas().to_dict(orient="records"))

# @app.route("/api/sensors")
# def sensors():
#     df = spark.sql("""
#         SELECT latitude, longitude, sensor_type,
#                uptime_pct, route_id
#         FROM dot_geo.sensor_locations
#         WHERE operational_status = 'Active'
#     """)
#     return jsonify(df.toPandas().to_dict(orient="records"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
