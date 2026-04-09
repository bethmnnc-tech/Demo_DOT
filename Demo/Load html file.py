# Databricks notebook source
with open("/Volumes/main/default/dot_lakehouse/exports/dot_geospatial_nc_demo.html", "r") as f:
    html_content = f.read()

displayHTML(html_content)

# COMMAND ----------

# In a Databricks notebook, run after your Gold layer job

# Pull live data from Gold tables
df_hotspots = spark.sql("""
    SELECT h3_index_r8, incident_count, total_fatalities, 
           avg_severity_score, hotspot_class, z_score
    FROM workspace.dot_geo.incident_hotspots
    WHERE incident_count > 0
""").toPandas()

df_bridges = spark.sql("""
    SELECT latitude, longitude, risk_score, year_built, 
           priority_tier, state_code
    FROM workspace.dot_silver.bridge_inspections
""").toPandas()

# NOTE: sensor_locations table does not exist in the workspace.
# Uncomment and update once the table is available.
# df_sensors = spark.sql("""
#     SELECT latitude, longitude, sensor_type, 
#            operational_status, uptime_pct, route_id
#     FROM dot_geo.sensor_locations
#     WHERE operational_status = 'Active'
# """).toPandas()

import json

# Convert to JSON for injection into the HTML template
hotspots_json = df_hotspots.to_json(orient='records')
bridges_json  = df_bridges.to_json(orient='records')
# sensors_json  = df_sensors.to_json(orient='records')

# Read the HTML template
with open("/Volumes/main/default/dot_lakehouse/exports/dot_geospatial_nc_demo.html", "r") as f:
    template = f.read()

# Replace placeholder comments with live data
output = template \
    .replace("/* INJECT_HOTSPOTS */", f"const HOTSPOTS_DATA = {hotspots_json};") \
    .replace("/* INJECT_BRIDGES */",  f"const BRIDGES_DATA  = {bridges_json};")
    # .replace("/* INJECT_SENSORS */",  f"const SENSORS_DATA  = {sensors_json};")

# Write the live version
with open("/Volumes/main/default/dot_lakehouse/exports/dot_geospatial_nc_demo.html", "w") as f:
    f.write(output)

print("✅ Live HTML regenerated from Delta tables")
