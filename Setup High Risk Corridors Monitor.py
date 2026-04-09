# Databricks notebook source
# DBTITLE 1,Setup Lakehouse Monitor and refresh metric tables
import time, json, requests
from databricks.sdk.runtime import dbutils

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
host = ctx.apiUrl().get()
token = ctx.apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

TABLE = "workspace.dot_gold.high_risk_corridors"

# Get existing monitor
resp = requests.get(f"{host}/api/2.1/unity-catalog/tables/{TABLE}/monitor", headers=headers)
print(f"GET monitor: {resp.status_code}")
if resp.status_code == 200:
    info = resp.json()
    print(f"  Output schema: {info.get('output_schema_name')}")
    print(f"  Profile table: {info.get('profile_metrics_table_name')}")
    print(f"  Drift table:   {info.get('drift_metrics_table_name')}")
else:
    print(f"  Response: {resp.text[:300]}")

# Trigger refresh
resp = requests.post(f"{host}/api/2.1/unity-catalog/tables/{TABLE}/monitor/refreshes", headers=headers)
print(f"\nPOST refresh: {resp.status_code}")
if resp.status_code == 200:
    refresh = resp.json()
    refresh_id = refresh["refresh_id"]
    print(f"  Refresh started (ID: {refresh_id})")

    # Poll until complete
    while True:
        time.sleep(15)
        resp = requests.get(
            f"{host}/api/2.1/unity-catalog/tables/{TABLE}/monitor/refreshes",
            headers=headers
        )
        refreshes = resp.json().get("refreshes", [])
        current = [r for r in refreshes if r["refresh_id"] == refresh_id]
        if current:
            state = current[0]["state"]
            print(f"  State: {state}")
            if state in ("SUCCESS", "FAILED", "CANCELED"):
                if current[0].get("message"):
                    print(f"  Message: {current[0]['message']}")
                break

    # Verify metric tables exist
    print()
    for suffix in ["profile_metrics", "drift_metrics"]:
        try:
            cnt = spark.table(f"workspace.dot_gold.high_risk_corridors_{suffix}").count()
            print(f"✓ high_risk_corridors_{suffix}: {cnt} rows")
        except Exception as e:
            print(f"✗ high_risk_corridors_{suffix}: {e}")
else:
    print(f"  Response: {resp.text[:500]}")
