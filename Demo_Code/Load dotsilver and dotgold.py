# Databricks notebook source
dbutils.widgets.text("base_path", "/Volumes/main/default/dot_lakehouse")
dbutils.widgets.text("catalog", "main")
BASE_PATH = dbutils.widgets.get("base_path")
CATALOG   = dbutils.widgets.get("catalog")
print(f"BASE_PATH={BASE_PATH}, CATALOG={CATALOG}")

# COMMAND ----------

df = spark.read.format('delta').load(f'{BASE_PATH}/gold/bridge_health_summary/')
df.write.format('delta').mode('overwrite').saveAsTable(f'{CATALOG}.dot_gold.bridge_health_summary')

# COMMAND ----------

df = spark.read.format('delta').load(f'{BASE_PATH}/gold/executive_kpi_scorecard/')
df.write.format('delta').mode('overwrite').saveAsTable(f'{CATALOG}.dot_gold.executive_kpi_scorecard')

# COMMAND ----------

df = spark.read.format('delta').load(f'{BASE_PATH}/gold/fleet_snapshot/')
df.write.format('delta').mode('overwrite').saveAsTable(f'{CATALOG}.dot_gold.fleet_snapshot')

# COMMAND ----------

df = spark.read.format('delta').load(f'{BASE_PATH}/gold/high_risk_corridors/')
df.write.format('delta').mode('overwrite').saveAsTable(f'{CATALOG}.dot_gold.high_risk_corridors')

# COMMAND ----------

df = spark.read.format('delta').load(f'{BASE_PATH}/gold/incident_summary/')
df.write.format('delta').mode('overwrite').saveAsTable(f'{CATALOG}.dot_gold.incident_summary')

# COMMAND ----------

df = spark.read.format('delta').load(f'{BASE_PATH}/gold/pavement_needs_assessment/')
df.write.format('delta').mode('overwrite').saveAsTable(f'{CATALOG}.dot_gold.pavement_needs_assessment')

# COMMAND ----------

df = spark.read.format('delta').load(f'{BASE_PATH}/silver/bridge_inspections/')
df.write.format('delta').mode('overwrite').saveAsTable(f'{CATALOG}.dot_silver.bridge_inspections')

# COMMAND ----------

df = spark.read.format('delta').load(f'{BASE_PATH}/silver/pavement_conditions/')
df.write.format('delta').mode('overwrite').saveAsTable(f'{CATALOG}.dot_silver.pavement_conditions')

# COMMAND ----------

df = spark.read.format('delta').load(f'{BASE_PATH}/silver/traffic_incidents/')
df.write.format('delta').mode('overwrite').saveAsTable(f'{CATALOG}.dot_silver.traffic_incidents')

# COMMAND ----------

df = spark.read.format('delta').load(f'{BASE_PATH}/silver/vehicle_registrations/')
df.write.format('delta').mode('overwrite').saveAsTable(f'{CATALOG}.dot_silver.vehicle_registrations')
