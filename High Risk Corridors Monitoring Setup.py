# Databricks notebook source
import requests, time
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
auth_headers = w.config._header_factory()
auth_headers["Content-Type"] = "application/json"

TABLE = "main.dot_gold.high_risk_corridors"
BASE = w.config.host + "/api/2.1/unity-catalog/tables/" + TABLE + "/monitor"

# Wait for monitor to become ACTIVE
print("Waiting for monitor to become ACTIVE...")
for i in range(40):
    resp = requests.get(BASE, headers=auth_headers)
    if resp.status_code == 200:
        status = resp.json().get("status", "UNKNOWN")
        print("  Monitor status: " + status)
        if status == "MONITOR_STATUS_ACTIVE":
            break
    else:
        print("  GET monitor: " + str(resp.status_code))
    time.sleep(15)

# Trigger refresh
resp = requests.post(BASE + "/refreshes", headers=auth_headers)
print("\nRefresh: " + str(resp.status_code))
if resp.status_code == 200:
    refresh_id = resp.json()["refresh_id"]
    print("Refresh ID: " + str(refresh_id) + " - polling...")
    while True:
        time.sleep(15)
        resp2 = requests.get(BASE + "/refreshes", headers=auth_headers)
        refreshes = resp2.json().get("refreshes", [])
        current = [r for r in refreshes if r["refresh_id"] == refresh_id]
        if current and current[0]["state"] in ("SUCCESS", "FAILED", "CANCELED"):
            print("Result: " + current[0]["state"])
            if current[0].get("message"):
                print("Message: " + current[0]["message"])
            break
        print("  ...running")

    # Verify
    for suffix in ["profile_metrics", "drift_metrics"]:
        cnt = spark.table("main.dot_gold.high_risk_corridors_" + suffix).count()
        print("high_risk_corridors_" + suffix + ": " + str(cnt) + " rows")
    print("\nDone! Metric tables are ready.")
else:
    print(resp.text[:500])
