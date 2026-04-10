# Databricks notebook source
# DBTITLE 1,Setup Lakehouse Monitor and refresh metric tables
dbutils.widgets.text("base_path", "/Volumes/main/default/dot_lakehouse")
dbutils.widgets.text("catalog", "main")
BASE_PATH = dbutils.widgets.get("base_path")
CATALOG   = dbutils.widgets.get("catalog")

import json, requests
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Build auth headers from SDK config
auth_headers = w.config._header_factory()
auth_headers["Content-Type"] = "application/json"

TABLE = f"{CATALOG}.dot_gold.bridge_health_summary"
BASE = w.config.host + "/api/2.1/unity-catalog/tables/" + TABLE + "/monitor"

# Create snapshot monitor (current state only, no time series)
resp = requests.post(
    BASE,
    headers=auth_headers,
    json={
        "output_schema_name": f"{CATALOG}.dot_gold",
        f"assets_dir": f"/Workspace/Repos/Beth_Development/Demo_DOT/databricks_lakehouse_monitoring/{CATALOG}.dot_gold.bridge_health_summary",
        "snapshot": {},
    },
)
print("Create monitor: " + str(resp.status_code))
if resp.status_code == 200:
    info = resp.json()
    print("Profile metrics: " + info.get("profile_metrics_table_name", "N/A"))
    print("Drift metrics: " + info.get("drift_metrics_table_name", "N/A"))
else:
    print(resp.text[:500])

# COMMAND ----------

# DBTITLE 1,Run first refresh and verify metric tables
import time, requests
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
auth_headers = w.config._header_factory()
auth_headers["Content-Type"] = "application/json"

TABLE = f"{CATALOG}.dot_gold.bridge_health_summary"
BASE = w.config.host + "/api/2.1/unity-catalog/tables/" + TABLE + "/monitor"

# Trigger refresh
resp = requests.post(BASE + "/refreshes", headers=auth_headers)
print("Refresh: " + str(resp.status_code))
if resp.status_code == 200:
    refresh_id = resp.json()["refresh_id"]
    print("Refresh started (ID: " + str(refresh_id) + "), polling...")

    while True:
        time.sleep(15)
        resp = requests.get(BASE + "/refreshes", headers=auth_headers)
        refreshes = resp.json().get("refreshes", [])
        current = [r for r in refreshes if r["refresh_id"] == refresh_id]
        if current:
            state = current[0]["state"]
            print("  State: " + state)
            if state in ("SUCCESS", "FAILED", "CANCELED"):
                if current[0].get("message"):
                    print("  Message: " + current[0]["message"])
                break

    # Verify metric tables exist
    print()
    for suffix in ["profile_metrics", "drift_metrics"]:
        try:
            cnt = spark.table(f"{CATALOG}.dot_gold.bridge_health_summary_" + suffix).count()
            print("bridge_health_summary_" + suffix + ": " + str(cnt) + " rows")
        except Exception as e:
            print("bridge_health_summary_" + suffix + ": " + str(e))
    print("\nDone! The monitoring dashboard should now work.")
else:
    print(resp.text[:500])
