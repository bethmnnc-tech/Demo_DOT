CREATE SCHEMA IF NOT EXISTS main.bronze;

CREATE OR REPLACE TABLE main.bronze.traffic_incidents
USING DELTA
LOCATION '/Volumes/main/default/dot_lakehouse/bronze/traffic_incidents';

CREATE OR REPLACE TABLE main.bronze.bridge_inspections
USING DELTA
LOCATION '/Volumes/main/default/dot_lakehouse/bronze/bridge_inspections';

