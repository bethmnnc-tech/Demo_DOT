import os
import json
import requests as req
from flask import Flask, jsonify, send_from_directory

app = Flask(__name__, static_folder="static")

NCDOT_API = "https://eapps.ncdot.gov/services/traffic-prod/v1"

# ── Serve the map dashboard ──
@app.route("/")
def index():
    return send_from_directory("static", "dot_geospatial_nc_demo.html")

# ── Health check ──
@app.route("/health")
def health():
    return jsonify({"status": "ok"})

# ── Live camera feed from NCDOT API ──
@app.route("/api/cameras")
def cameras():
    try:
        roads = req.get(f"{NCDOT_API}/roads", timeout=15).json()
        rid_to_name = {r["id"]: r["name"] for r in roads}
        all_cams = req.get(f"{NCDOT_API}/cameras", timeout=15).json()
        cameras = []
        for cam in all_cams:
            try:
                detail = req.get(f"{NCDOT_API}/cameras/{cam['id']}", timeout=8).json()
            except Exception:
                continue
            if detail.get("status") != "OK":
                continue
            lat = detail.get("latitude")
            lon = detail.get("longitude")
            if not lat or not lon:
                continue
            cameras.append({
                "label": detail.get("locationName", ""),
                "route": rid_to_name.get(detail.get("roadId"), ""),
                "lat": lat,
                "lon": lon,
                "image_url": detail.get("imageURL", ""),
            })
        return jsonify(cameras)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
