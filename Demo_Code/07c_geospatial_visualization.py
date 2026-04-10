# =============================================================================
# NOTEBOOK 7c: DOT Geospatial Visualization & GeoJSON Export
# Databricks Notebook | DBR 14.3 LTS ML
#
# Description:
#   1. Export key geospatial tables to GeoJSON (for ArcGIS, QGIS, web maps)
#   2. Build interactive Folium map (renders in Databricks notebook output)
#   3. Build KeplerGl choropleth map (H3 hex layer)
#   4. GeoParquet export (columnar spatial format for large-scale tools)
# =============================================================================

import subprocess
subprocess.check_call(["pip", "install", "-q", "folium", "keplergl", "geopandas", "pyarrow", "h3", "shapely"])

import json
import os
import folium
import geopandas as gpd
import pandas as pd
from shapely import wkt as shapely_wkt
from keplergl import KeplerGl

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("DOT_GeoVisualization").getOrCreate()

# ── Configuration ──
# When run by a job: parameters arrive via sys.argv
# When run interactively: dbutils widgets provide defaults
import sys
if len(sys.argv) >= 3:
    BASE_PATH = sys.argv[1]
    CATALOG   = sys.argv[2]
else:
    dbutils.widgets.text("base_path", "/Volumes/main/default/dot_lakehouse")
    dbutils.widgets.text("catalog", "main")
    BASE_PATH = dbutils.widgets.get("base_path")
    CATALOG   = dbutils.widgets.get("catalog")
GEO_PATH  = f"{BASE_PATH}/gold/geospatial"
EXPORT_PATH = f"{BASE_PATH}/exports/geospatial"

# Ensure export directories exist
os.makedirs(EXPORT_PATH, exist_ok=True)
os.makedirs(f"{EXPORT_PATH}/geoparquet", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Spark DF → GeoPandas GeoDataFrame
# ─────────────────────────────────────────────────────────────────────────────
def to_geodataframe(df_spark, geom_col="geometry_wkt", crs="EPSG:4326") -> gpd.GeoDataFrame:
    """Convert a Spark DF with WKT geometry column to a GeoPandas GeoDataFrame."""
    pdf = df_spark.toPandas()
    pdf = pdf[pdf[geom_col].notna() & (pdf[geom_col] != "")]
    geom = gpd.GeoSeries.from_wkt(pdf[geom_col], crs=crs)
    gdf = gpd.GeoDataFrame(pdf.drop(columns=[geom_col]), geometry=geom)
    return gdf

# ─────────────────────────────────────────────────────────────────────────────
# MAP 1 – Incident Hotspot Map (Folium)
#   Layers: H3 hex cells (color by risk tier) + incident markers
# ─────────────────────────────────────────────────────────────────────────────
print("Building Folium incident hotspot map …")

# Pull hotspot cells with H3 boundaries
df_hot = (
    spark.read.format("delta").load(f"{GEO_PATH}/incident_hotspots")
    .filter(F.col("is_hotspot"))
    .select("h3_index_r8","incident_count","total_fatalities",
            "avg_severity_score","hotspot_class","risk_tier")
    .limit(500)   # limit for notebook rendering
)
pdf_hot = df_hot.toPandas()

# Pull top 200 incidents for marker layer
df_inc_sample = (
    spark.read.format("delta").load(f"{GEO_PATH}/incidents_geo_enriched")
    .filter(F.col("has_fatality") == True)
    .select("incident_id","latitude","longitude","incident_type",
            "severity","fatalities","injuries","incident_datetime")
    .limit(200)
)
pdf_inc = df_inc_sample.toPandas()

# Build Folium map centered on NC
m = folium.Map(
    location=[35.5, -79.5],
    zoom_start=7,
    tiles="CartoDB Positron",
    attr="DOT Safety Analytics | Data: NCDOT synthetic demo",
)

# H3 hexagon layer (polygon overlay)
import h3
def h3_to_polygon_coords(h3_index: str):
    """Return Folium-style [lat, lon] boundary coords for an H3 cell."""
    boundary = h3.cell_to_boundary(h3_index)  # list of (lat, lon)
    return [[lat, lon] for lat, lon in boundary]

risk_colors = {
    "Hot Spot 99% Confidence": "#d73027",
    "Hot Spot 95% Confidence": "#f46d43",
    "Hot Spot 90% Confidence": "#fdae61",
}

hex_layer = folium.FeatureGroup(name="Incident Hotspot Hexagons", show=True)
for _, row in pdf_hot.iterrows():
    try:
        coords = h3_to_polygon_coords(row["h3_index_r8"])
        color  = risk_colors.get(row["hotspot_class"], "#fee090")
        folium.Polygon(
            locations=coords,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.55,
            weight=0.8,
            tooltip=folium.Tooltip(
                f"<b>H3: {row['h3_index_r8']}</b><br>"
                f"Incidents: {row['incident_count']}<br>"
                f"Fatalities: {row['total_fatalities']}<br>"
                f"Class: {row['hotspot_class']}"
            ),
        ).add_to(hex_layer)
    except Exception:
        pass
hex_layer.add_to(m)

# Fatal incident markers
fatal_layer = folium.FeatureGroup(name="Fatal Incidents", show=True)
for _, row in pdf_inc.iterrows():
    if pd.notna(row["latitude"]) and pd.notna(row["longitude"]):
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5,
            color="#7b2d8b",
            fill=True,
            fill_color="#c084fc",
            fill_opacity=0.8,
            tooltip=folium.Tooltip(
                f"<b>{row['incident_type']}</b><br>"
                f"Severity: {row['severity']}<br>"
                f"Fatalities: {row['fatalities']}<br>"
                f"Date: {str(row['incident_datetime'])[:10]}"
            ),
        ).add_to(fatal_layer)
fatal_layer.add_to(m)

# Legend
legend_html = """
<div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
     padding:12px 16px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.2);
     font-family:sans-serif;font-size:12px">
  <b>Incident Hotspot Map</b><br><br>
  <span style="background:#d73027;padding:2px 8px;margin-right:6px">&nbsp;</span> 99% Confidence<br>
  <span style="background:#f46d43;padding:2px 8px;margin-right:6px">&nbsp;</span> 95% Confidence<br>
  <span style="background:#fdae61;padding:2px 8px;margin-right:6px">&nbsp;</span> 90% Confidence<br>
  <span style="background:#c084fc;padding:2px 8px;border-radius:50%;margin-right:6px">&nbsp;</span> Fatal Incident
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))
folium.LayerControl().add_to(m)

# Save map directly to Volumes path
map_path = f"{EXPORT_PATH}/dot_hotspot_map.html"
m.save(map_path)
print(f"  ✓ Folium map saved → {map_path}")

# Display in notebook
displayHTML(m._repr_html_())   # renders the interactive map inline

# ─────────────────────────────────────────────────────────────────────────────
# MAP 2 – KeplerGl Choropleth (H3 hex grid + road segments)
# ─────────────────────────────────────────────────────────────────────────────
print("\nBuilding KeplerGl H3 choropleth map …")

# H3 density data
df_density = spark.read.format("delta").load(f"{GEO_PATH}/h3_incident_density") \
    .select("h3_index_r8","incident_count","total_fatalities","avg_severity_score","risk_tier") \
    .limit(2000)
pdf_density = df_density.toPandas()

# Road segments (sample)
df_roads_sample = spark.read.format("delta").load(f"{BASE_PATH}/bronze/geospatial/road_segments") \
    .select("segment_id","route_name","geometry_wkt","aadt","pavement_condition") \
    .limit(300)
pdf_roads = df_roads_sample.toPandas()

# Sensor locations
df_sensors_sample = spark.read.format("delta").load(f"{BASE_PATH}/bronze/geospatial/sensor_locations") \
    .select("sensor_id","sensor_type","latitude","longitude","operational_status","uptime_pct") \
    .limit(300)
pdf_sensors = df_sensors_sample.toPandas()

# KeplerGl config (H3 hex layer)
kepler_config = {
    "version": "v1",
    "config": {
        "mapStyle": {"styleType": "dark"},
        "visState": {
            "layers": [
                {
                    "id": "h3_layer",
                    "type": "hexagonId",
                    "config": {
                        "dataId": "incident_density",
                        "label": "Incident Density",
                        "columns": {"hex_id": "h3_index_r8"},
                        "visConfig": {
                            "opacity": 0.8,
                            "colorRange": {
                                "colors": ["#feedde","#fdbe85","#fd8d3c","#e6550d","#a63603"]
                            },
                            "coverage": 0.9,
                            "enable3d": True,
                            "elevationScale": 5,
                        },
                        "colorField": {"name": "incident_count", "type": "integer"},
                        "heightField":{"name": "total_fatalities", "type": "integer"},
                    },
                    "visualChannels": {
                        "colorField":  {"name": "incident_count",    "type": "integer"},
                        "heightField": {"name": "total_fatalities",  "type": "integer"},
                    },
                },
                {
                    "id": "sensor_layer",
                    "type": "point",
                    "config": {
                        "dataId": "sensors",
                        "label": "Traffic Sensors",
                        "columns": {"lat": "latitude", "lng": "longitude"},
                        "visConfig": {
                            "radius": 6,
                            "colorRange": {"colors": ["#00b4d8","#0077b6"]},
                        },
                    },
                },
            ]
        },
        "mapState": {"latitude": 35.5, "longitude": -79.5, "zoom": 7},
    },
}

kepler_map = KeplerGl(height=600, config=kepler_config)
kepler_map.add_data(data=pdf_density,  name="incident_density")
kepler_map.add_data(data=pdf_sensors,  name="sensors")

kepler_path = f"{EXPORT_PATH}/dot_kepler_h3_map.html"
kepler_map.save_to_html(file_name=kepler_path)
print(f"  ✓ KeplerGl map saved → {kepler_path}")

kepler_map   # renders inline in notebook

# ─────────────────────────────────────────────────────────────────────────────
# EXPORT 1 – GeoJSON (universal format for ArcGIS, QGIS, web maps)
# ─────────────────────────────────────────────────────────────────────────────
print("\nExporting GeoJSON files …")

def export_geojson(df_spark, geom_col, output_path, limit=5000):
    gdf = to_geodataframe(df_spark.limit(limit), geom_col=geom_col)
    geojson_str = gdf.to_json()
    with open(output_path, "w") as f:
        f.write(geojson_str)
    print(f"  ✓ Exported {len(gdf)} features → {output_path}")
    return gdf

# Road segments as LineStrings
gdf_roads = export_geojson(
    spark.read.format("delta").load(f"{BASE_PATH}/bronze/geospatial/road_segments")
         .select("segment_id","route_name","functional_class","lanes","aadt",
                 "pavement_condition","area_type","geometry_wkt"),
    "geometry_wkt",
    f"{EXPORT_PATH}/road_segments.geojson",
)

# Intersections as Points
gdf_int = export_geojson(
    spark.read.format("delta").load(f"{BASE_PATH}/bronze/geospatial/intersections")
         .select("intersection_id","nearest_city","intersection_type",
                 "crash_count_5yr","fatal_crash_count_5yr","geometry_wkt"),
    "geometry_wkt",
    f"{EXPORT_PATH}/intersections.geojson",
)

# Traffic Analysis Zones as Polygons
gdf_taz = export_geojson(
    spark.read.format("delta").load(f"{BASE_PATH}/bronze/geospatial/traffic_analysis_zones")
         .select("taz_id","city","zone_type","population","employment",
                 "vehicle_trips_daily","geometry_wkt"),
    "geometry_wkt",
    f"{EXPORT_PATH}/traffic_analysis_zones.geojson",
)

# Work Zones as Polygons
gdf_wz = export_geojson(
    spark.read.format("delta").load(f"{BASE_PATH}/bronze/geospatial/work_zones")
         .filter(F.col("status") == "Active")
         .select("work_zone_id","zone_type","status","lanes_closed",
                 "zone_length_miles","project_cost_usd","geometry_wkt"),
    "geometry_wkt",
    f"{EXPORT_PATH}/active_work_zones.geojson",
)

# Incident hotspots (H3 cells → Polygon boundaries)
def h3_cells_to_geodataframe(pdf_h3: pd.DataFrame, h3_col: str) -> gpd.GeoDataFrame:
    """Convert H3 cell index column to GeoDataFrame with Polygon geometries."""
    from shapely.geometry import Polygon as ShapelyPolygon
    import h3
    data_rows = []
    geoms = []
    for _, row in pdf_h3.iterrows():
        try:
            boundary = h3.cell_to_boundary(row[h3_col])
            poly = ShapelyPolygon([(lon, lat) for lat, lon in boundary])
            data_rows.append(row.to_dict())
            geoms.append(poly)
        except Exception:
            pass
    if not data_rows:
        return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:4326"))
    pdf_result = pd.DataFrame(data_rows)
    return gpd.GeoDataFrame(pdf_result, geometry=gpd.GeoSeries(geoms, index=pdf_result.index, crs="EPSG:4326"))

pdf_hotspot = (
    spark.read.format("delta").load(f"{GEO_PATH}/incident_hotspots")
    .filter(F.col("is_hotspot"))
    .select("h3_index_r8","incident_count","total_fatalities",
            "avg_severity_score","hotspot_class","z_score")
    .limit(1000)
    .toPandas()
)
gdf_hotspot = h3_cells_to_geodataframe(pdf_hotspot, "h3_index_r8")
hotspot_path = f"{EXPORT_PATH}/incident_hotspots.geojson"
with open(hotspot_path, "w") as f:
    f.write(gdf_hotspot.to_json())
print(f"  ✓ Exported {len(gdf_hotspot)} hotspot cells → {hotspot_path}")

# ─────────────────────────────────────────────────────────────────────────────
# EXPORT 2 – GeoParquet (cloud-native columnar spatial format)
# Compatible with DuckDB, Sedona, BigQuery, Snowflake Iceberg
# ─────────────────────────────────────────────────────────────────────────────
print("\nExporting GeoParquet files …")

for gdf, name in [
    (gdf_roads,   "road_segments"),
    (gdf_int,     "intersections"),
    (gdf_taz,     "traffic_analysis_zones"),
    (gdf_hotspot, "incident_hotspots"),
]:
    pq_path = f"{EXPORT_PATH}/geoparquet/{name}.parquet"
    gdf.to_parquet(pq_path, engine="pyarrow", index=False)
    print(f"  ✓ GeoParquet: {name} ({len(gdf)} features)")

print(f"\n✅  All geospatial exports complete → {EXPORT_PATH}/")
print("""
Files exported:
  GeoJSON (for QGIS / ArcGIS / web maps):
    road_segments.geojson
    intersections.geojson
    traffic_analysis_zones.geojson
    active_work_zones.geojson
    incident_hotspots.geojson
  Interactive HTML maps:
    dot_hotspot_map.html   (Folium – incident layer)
    dot_kepler_h3_map.html (KeplerGl – H3 3D choropleth)
  GeoParquet (DuckDB / Sedona / Snowflake):
    geoparquet/road_segments.parquet
    geoparquet/intersections.parquet
    geoparquet/traffic_analysis_zones.parquet
    geoparquet/incident_hotspots.parquet
""")
