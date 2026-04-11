# Databricks notebook source
# DBTITLE 1,Fetch cameras and incidents from NCDOT API
import requests, json, re, time
from datetime import datetime, timezone

NCDOT_API = "https://eapps.ncdot.gov/services/traffic-prod/v1"
TARGET_ROADS = ["I-40", "I-85", "I-77", "I-95", "US-74"]

# Get road lookup
roads = requests.get(f"{NCDOT_API}/roads", timeout=15).json()
road_lookup = {r["name"]: r["id"] for r in roads if r["name"] in TARGET_ROADS}
rid_to_name = {v: k for k, v in road_lookup.items()}

# Fetch all cameras
all_cams = requests.get(f"{NCDOT_API}/cameras", timeout=15).json()
cameras = []
for cam in all_cams:
    try:
        detail = requests.get(f"{NCDOT_API}/cameras/{cam['id']}", timeout=8).json()
    except Exception:
        continue
    if detail.get("roadId") not in rid_to_name or detail.get("status") != "OK":
        continue
    cameras.append({
        "label": detail.get("locationName", ""),
        "route": rid_to_name[detail["roadId"]],
        "lat": detail["latitude"],
        "lon": detail["longitude"],
        "image_url": detail.get("imageURL", ""),
    })

# Fetch incidents
incidents = []
seen = set()
for road, rid in road_lookup.items():
    try:
        resp = requests.get(f"{NCDOT_API}/roads/{rid}/incidents?verbose=true&recent=true", timeout=15)
        for inc in resp.json():
            iid = inc.get("id")
            if iid in seen:
                continue
            seen.add(iid)
            incidents.append({
                "road": inc.get("road", road).strip(),
                "type": inc.get("incidentType", ""),
                "condition": inc.get("condition", ""),
                "location": inc.get("location", ""),
                "reason": inc.get("reason", ""),
                "county": inc.get("countyName", ""),
                "severity": inc.get("severity", 1),
                "lat": inc.get("latitude"),
                "lon": inc.get("longitude"),
            })
    except Exception:
        pass

timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
print(f"Fetched {len(cameras)} cameras, {len(incidents)} incidents at {timestamp}")

# COMMAND ----------

# DBTITLE 1,Save data.json
data = {"cameras": cameras, "incidents": incidents, "timestamp": timestamp}
app_dir = "/Workspace/Repos/Beth_Development/Demo_DOT/dot_app/ncdot-camera-viewer"

with open(f"{app_dir}/data.json", "w") as f:
    json.dump(data, f)

print(f"Saved data.json: {len(json.dumps(data)):,} bytes")

# COMMAND ----------

# DBTITLE 1,Rebuild index.html with fresh data
with open(f"{app_dir}/index.html") as f:
    html = f.read()

# Replace the JS data variables (use lambda to avoid re escape issues in URLs)
html = re.sub(r'var incidents = \[.*?\];', lambda m: 'var incidents = ' + json.dumps(incidents) + ';', html, flags=re.DOTALL)
html = re.sub(r'var cameras = \[.*?\];', lambda m: 'var cameras = ' + json.dumps(cameras) + ';', html, flags=re.DOTALL)

# Update header badges
html = re.sub(r'\d+ active incidents', f'{len(incidents)} active incidents', html)
html = re.sub(r'\d+ cameras', f'{len(cameras)} cameras', html)

# Update timestamp
html = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC', timestamp, html)

with open(f"{app_dir}/index.html", "w") as f:
    f.write(html)

print(f"Rebuilt index.html: {len(html):,} chars")

# COMMAND ----------

# DBTITLE 1,Redeploy ncdot-camera-viewer app
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
host = w.config.host
auth_headers = w.config.authenticate()
headers = {**auth_headers, "Content-Type": "application/json"}

resp = requests.post(f"{host}/api/2.0/apps/ncdot-camera-viewer/deployments", headers=headers,
    json={"source_code_path": "/Workspace/Repos/Beth_Development/Demo_DOT/dot_app/ncdot-camera-viewer", "mode": "SNAPSHOT"})
did = resp.json().get("deployment_id")
print(f"Deploying: {did}")

for i in range(18):
    time.sleep(10)
    r = requests.get(f"{host}/api/2.0/apps/ncdot-camera-viewer/deployments/{did}", headers=headers)
    state = r.json().get("status", {}).get("state", "?")
    print(f"  [{(i+1)*10}s] {state}")
    if state in ("SUCCEEDED", "FAILED"):
        break

print(f"\nRefresh complete: {len(cameras)} cameras, {len(incidents)} incidents deployed")
