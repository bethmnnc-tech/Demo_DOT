# Databricks notebook source
# MAGIC %md
# MAGIC # Michigan DOT Camera Data Refresh
# MAGIC Fetches live camera data from MDOT MiDrive API and rebuilds the static HTML app.

# COMMAND ----------

# DBTITLE 1,Fetch cameras, incidents, and construction from MDOT
import requests, json, re, os, builtins
from datetime import datetime, timezone

app_dir = "/Workspace/Repos/Beth_Development/Demo_DOT/dot_app/mdot-camera-viewer"
BASE = "https://mdotjboss.state.mi.us/MiDrive"

# Use a session — incident/construction endpoints require cookies
session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})

# ── 1. Fetch cameras ────────────────────────────────────────
print("Fetching cameras from MDOT MiDrive API...")
r = session.get(f"{BASE}/camera/list", timeout=30)
r.raise_for_status()
cameras_raw = r.json()
print(f"Raw cameras: {len(cameras_raw)}")

cameras = []
for c in cameras_raw:
    coord_match = re.search(r'lat=([\d.-]+)&lon=([\d.-]+).*?id=(\d+)', c.get("county", ""))
    if not coord_match:
        continue
    img_match = re.search(r'src="([^"]+)"', c.get("image", ""))
    if not img_match:
        continue
    county = re.sub(r'\s*<.*', '', c.get("county", "")).strip()
    cameras.append({
        "label": (c.get("location", "") or "").strip(),
        "route": (c.get("route", "") or "").strip(),
        "lat": float(coord_match.group(1)),
        "lon": float(coord_match.group(2)),
        "image_url": img_match.group(1),
        "county": county,
        "direction": (c.get("direction", "") or "").strip(),
        "id": coord_match.group(3),
    })
print(f"Parsed cameras: {len(cameras)}")

# ── 2. Fetch incidents (requires session cookie) ───────────
print("\nFetching incidents...")
session.get(f"{BASE}/map", timeout=15)  # establish session cookies

r_inc = session.get(f"{BASE}/incidents/AllForPage", timeout=15)
r_inc.raise_for_status()
inc_raw = r_inc.json()
print(f"Raw incidents: {len(inc_raw)}")

incidents = []
for inc in inc_raw:
    text = inc.get("incidentText", "")
    location = re.search(r'Location:\s*</strong>([^<]+)', text)
    event_type = re.search(r'Event Type:\s*</strong>\s*([^<]+)', text)
    lanes = re.search(r'Lanes Blocked:\s*</strong>\s*([^<]+)', text)
    county = re.search(r'County:\s*</strong>\s*([^<]+)', text)
    message = re.search(r'Event Message:</strong>\s*([^<]+)', text)

    icon = inc.get("iconURL", "")
    lanes_text = (lanes.group(1).strip() if lanes else "")
    if "All Lanes" in lanes_text:
        severity = 4  # Severe
    elif "red" in icon:
        severity = 3  # Major
    elif "yellow" in icon or "orange" in icon:
        severity = 2  # Moderate
    else:
        severity = 1  # Minor / cleared

    title = inc.get("incidentTitle", "")
    road_match = re.search(r'(I-\d+|M-\d+|US-\d+)', title)
    road = road_match.group(1) if road_match else ""

    incidents.append({
        "road": road,
        "type": (event_type.group(1).strip() if event_type else "Incident"),
        "condition": lanes_text or "Unknown",
        "location": (location.group(1).strip() if location else title),
        "reason": (message.group(1).strip() if message else ""),
        "county": (county.group(1).strip() if county else ""),
        "severity": severity,
        "lat": inc["latitude"],
        "lon": inc["longitude"],
    })

# ── 3. Fetch construction/work zones ───────────────────────
print("Fetching construction zones...")
session.get(f"{BASE}/construction", timeout=15)
r_con = session.get(f"{BASE}/construction/list/loadConstruction", timeout=15)
r_con.raise_for_status()
con_raw = r_con.json()
print(f"Raw construction: {len(con_raw)}")

for c in con_raw:
    desc = c.get("description", "")
    coord = re.search(r'lat=([\d.-]+)&lon=([\d.-]+)', desc)
    if not coord:
        continue
    clean_desc = re.sub(r'\s*<a\b.*', '', desc).strip()
    incidents.append({
        "road": re.sub(r'^(EB|WB|NB|SB)\s+', '', (c.get("route", "") or "")).strip(),
        "type": "Construction",
        "condition": (c.get("type", "") or "work zone").replace("-", " ").title(),
        "location": clean_desc,
        "reason": "",
        "county": (c.get("county", "") or "").strip(),
        "severity": 2,  # Moderate
        "lat": float(coord.group(1)),
        "lon": float(coord.group(2)),
    })

print(f"Total incidents + construction: {len(incidents)}")

# ── 4. Save data.json ──────────────────────────────────────
timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
data = {"cameras": cameras, "incidents": incidents, "timestamp": timestamp}
with builtins.open(f"{app_dir}/data.json", "w") as f:
    json.dump(data, f)
print(f"Saved data.json: {os.path.getsize(f'{app_dir}/data.json'):,} bytes")

# COMMAND ----------

# DBTITLE 1,Rebuild index.html with fresh data
# Rebuild index.html with fresh camera + incident + construction data
# Split incidents: real incidents vs construction zones
real_incidents = [i for i in incidents if i["type"] != "Construction"]
construction = [i for i in incidents if i["type"] == "Construction"]

cameras_json = json.dumps(cameras)
incidents_json = json.dumps(real_incidents)
construction_json = json.dumps(construction)

with builtins.open(f"{app_dir}/index.html") as f:
    html = f.read()

# Replace cameras data
html = re.sub(
    r'var cameras = \[.*?\];',
    lambda m: f'var cameras = {cameras_json};',
    html,
    flags=re.DOTALL
)

# Replace incidents data (just real incidents, not construction)
html = re.sub(
    r'var incidents = \[.*?\];',
    lambda m: f'var incidents = {incidents_json};',
    html,
    flags=re.DOTALL
)

# Replace construction data
html = re.sub(
    r'var construction = \[.*?\];',
    lambda m: f'var construction = {construction_json};',
    html,
    flags=re.DOTALL
)

# Update badge counts
html = re.sub(r'\d+ active incidents', f'{len(real_incidents)} active incidents', html, count=1)
html = re.sub(r'\d+ cameras', f'{len(cameras)} cameras', html, count=1)
html = re.sub(r'\d+ construction zones', f'{len(construction)} construction zones', html, count=1)
html = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC', timestamp, html)

# Update filter-counts default text
html = re.sub(
    r"\d+ incidents, \d+ construction, \d+ cameras",
    f"{len(real_incidents)} incidents, {len(construction)} construction, {len(cameras)} cameras",
    html, count=1
)

with builtins.open(f"{app_dir}/index.html", "w") as f:
    f.write(html)
print(f"Rebuilt index.html: {len(html):,} chars")
print(f"  {len(cameras)} cameras, {len(real_incidents)} incidents, {len(construction)} construction zones")

# COMMAND ----------

# Redeploy the app
import requests as req

host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

resp = req.post(
    f"{host}/api/2.0/apps/mdot-camera-viewer/deployments",
    headers=headers,
    json={"source_code_path": "/Workspace/Repos/Beth_Development/Demo_DOT/dot_app/mdot-camera-viewer"}
)
print(f"Deploy status: {resp.status_code}")
print(resp.json())
