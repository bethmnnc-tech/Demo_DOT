# Databricks notebook source
df = spark.read.format('delta').load('/Volumes/main/default/dot_lakehouse/gold/bridge_health_summary/')
df.write.format('delta').mode('overwrite').saveAsTable('main.dot_gold.bridge_health_summary')

# COMMAND ----------

df = spark.read.format('delta').load('/Volumes/main/default/dot_lakehouse/gold/executive_kpi_scorecard/')
df.write.format('delta').mode('overwrite').saveAsTable('main.dot_gold.executive_kpi_scorecard')

# COMMAND ----------

df = spark.read.format('delta').load('/Volumes/main/default/dot_lakehouse/gold/fleet_snapshot/')
df.write.format('delta').mode('overwrite').saveAsTable('main.dot_gold.fleet_snapshot')

# COMMAND ----------

df = spark.read.format('delta').load('/Volumes/main/default/dot_lakehouse/gold/high_risk_corridors/')
df.write.format('delta').mode('overwrite').saveAsTable('main.dot_gold.high_risk_corridors')

# COMMAND ----------

df = spark.read.format('delta').load('/Volumes/main/default/dot_lakehouse/gold/incident_summary/')
df.write.format('delta').mode('overwrite').saveAsTable('main.dot_gold.incident_summary')

# COMMAND ----------

df = spark.read.format('delta').load('/Volumes/main/default/dot_lakehouse/gold/pavement_needs_assessment/')
df.write.format('delta').mode('overwrite').saveAsTable('main.dot_gold.pavement_needs_assessment')

# COMMAND ----------

df = spark.read.format('delta').load('/Volumes/main/default/dot_lakehouse/silver/bridge_inspections/')
df.write.format('delta').mode('overwrite').saveAsTable('main.dot_silver.bridge_inspections')

# COMMAND ----------

df = spark.read.format('delta').load('/Volumes/main/default/dot_lakehouse/silver/pavement_conditions/')
df.write.format('delta').mode('overwrite').saveAsTable('main.dot_silver.pavement_conditions')

# COMMAND ----------

df = spark.read.format('delta').load('/Volumes/main/default/dot_lakehouse/silver/traffic_incidents/')
df.write.format('delta').mode('overwrite').saveAsTable('main.dot_silver.traffic_incidents')

# COMMAND ----------

df = spark.read.format('delta').load('/Volumes/main/default/dot_lakehouse/silver/vehicle_registrations/')
df.write.format('delta').mode('overwrite').saveAsTable('main.dot_silver.vehicle_registrations')
