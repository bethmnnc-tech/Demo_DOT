# Databricks notebook source
# MAGIC %sql
# MAGIC -- =============================================================================
# MAGIC -- NOTEBOOK 7d: DOT Geospatial SQL Queries
# MAGIC -- Databricks SQL Notebook | Uses dot_geo database
# MAGIC -- Description: Spatial analytics queries using H3 functions and WKT geometry.
# MAGIC --              Databricks supports ST_* functions natively via Photon engine.
# MAGIC -- =============================================================================
# MAGIC
# MAGIC -- ─────────────────────────────────────────────────────────────────────────────
# MAGIC -- Q1: Incident Hotspot Summary – Top H3 Cells by Fatality Concentration
# MAGIC -- ─────────────────────────────────────────────────────────────────────────────
# MAGIC SELECT
# MAGIC     h.h3_index_r8,
# MAGIC     h.state_code,
# MAGIC     h.incident_count,
# MAGIC     h.total_fatalities,
# MAGIC     h.total_injuries,
# MAGIC     ROUND(h.avg_severity_score, 2)          AS avg_severity,
# MAGIC     h.fatal_incident_count,
# MAGIC     h.night_incident_count,
# MAGIC     h.weekend_incident_count,
# MAGIC     ROUND(h.incident_density_per_km2, 2)    AS density_per_km2,
# MAGIC     ROUND(h.z_score, 2)                     AS hotspot_z_score,
# MAGIC     h.hotspot_class,
# MAGIC     h.risk_tier
# MAGIC FROM dot_geo.incident_hotspots h
# MAGIC WHERE h.is_hotspot = TRUE
# MAGIC ORDER BY h.total_fatalities DESC, h.incident_count DESC
# MAGIC LIMIT 50;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC
# MAGIC -- ─────────────────────────────────────────────────────────────────────────────
# MAGIC -- Q2: Corridor Safety Rate (incidents per 100M VMT)
# MAGIC -- Industry standard: compares corridors on a traffic-normalized basis
# MAGIC -- ─────────────────────────────────────────────────────────────────────────────
# MAGIC SELECT
# MAGIC     c.seg_route_id                                    AS route_id,
# MAGIC     c.functional_class,
# MAGIC     c.area_type,
# MAGIC     c.incidents_on_corridor,
# MAGIC     c.fatalities_on_corridor,
# MAGIC     c.injuries_on_corridor,
# MAGIC     FORMAT_NUMBER(c.reference_aadt, 0)                AS reference_aadt,
# MAGIC     ROUND(c.avg_severity, 2)                          AS avg_severity_score,
# MAGIC     ROUND(c.incidents_per_100m_vmt, 4)                AS incidents_per_100m_vmt,
# MAGIC     RANK() OVER (
# MAGIC         PARTITION BY c.functional_class
# MAGIC         ORDER BY c.incidents_per_100m_vmt DESC
# MAGIC     )                                                 AS safety_rank_in_class
# MAGIC FROM dot_geo.corridor_safety_rates c
# MAGIC ORDER BY c.incidents_per_100m_vmt DESC
# MAGIC LIMIT 30;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC
# MAGIC -- ─────────────────────────────────────────────────────────────────────────────
# MAGIC -- Q3: Bridge Risk by Traffic Analysis Zone
# MAGIC -- Identifies TAZs where structurally deficient bridges expose most population
# MAGIC -- ─────────────────────────────────────────────────────────────────────────────
# MAGIC SELECT
# MAGIC     b.taz_id,
# MAGIC     b.zone_type,
# MAGIC     b.state_code,
# MAGIC     COUNT(b.bridge_id)                                AS bridge_count,
# MAGIC     SUM(CASE WHEN b.worst_condition = 'Poor' THEN 1 ELSE 0 END) AS poor_condition_bridges,
# MAGIC     ROUND(AVG(b.risk_score), 1)                       AS avg_risk_score,
# MAGIC     ROUND(AVG(b.sufficiency_rating), 1)               AS avg_sufficiency,
# MAGIC     MAX(b.taz_population)                             AS taz_population,
# MAGIC     MAX(b.taz_vehicle_trips)                          AS taz_vehicle_trips,
# MAGIC     ROUND(SUM(b.population_at_risk_score), 0)         AS total_pop_at_risk_score,
# MAGIC     SUM(b.avg_daily_traffic)                          AS total_bridge_aadt,
# MAGIC     COLLECT_LIST(CASE WHEN b.priority_tier = 'Priority 1 – Immediate' THEN b.bridge_id END)
# MAGIC                                                       AS p1_bridge_ids
# MAGIC FROM dot_geo.bridges_with_taz b
# MAGIC GROUP BY 1,2,3
# MAGIC HAVING COUNT(b.bridge_id) >= 2
# MAGIC ORDER BY total_pop_at_risk_score DESC
# MAGIC LIMIT 25;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC
# MAGIC -- ─────────────────────────────────────────────────────────────────────────────
# MAGIC -- Q4: H3 Spatial Roll-Up – County-Level Density (resolution 6)
# MAGIC -- ─────────────────────────────────────────────────────────────────────────────
# MAGIC SELECT
# MAGIC     h3_index_r6                                       AS h3_county_cell,
# MAGIC     state_code,
# MAGIC     COUNT(DISTINCT h3_index_r8)                       AS sub_cells_count,
# MAGIC     SUM(incident_count)                               AS total_incidents,
# MAGIC     SUM(total_fatalities)                             AS total_fatalities,
# MAGIC     SUM(fatal_incident_count)                         AS fatal_incidents,
# MAGIC     SUM(night_incident_count)                         AS night_incidents,
# MAGIC     SUM(pedestrian_count)                             AS pedestrian_incidents,
# MAGIC     SUM(dui_count)                                    AS dui_incidents,
# MAGIC     ROUND(SUM(incident_count) / COUNT(DISTINCT h3_index_r8), 1) AS avg_incidents_per_cell,
# MAGIC     ROUND(AVG(avg_severity_score), 2)                 AS avg_severity,
# MAGIC     SUM(CASE WHEN risk_tier LIKE '%Critical%' OR risk_tier LIKE '%High%' THEN 1 ELSE 0 END)
# MAGIC                                                       AS high_risk_sub_cells
# MAGIC FROM dot_geo.h3_incident_density
# MAGIC GROUP BY 1,2
# MAGIC ORDER BY total_fatalities DESC;
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC -- ─────────────────────────────────────────────────────────────────────────────
# MAGIC -- Q5: Work Zone Safety Analysis – Incidents inside active construction zones
# MAGIC -- ─────────────────────────────────────────────────────────────────────────────
# MAGIC SELECT
# MAGIC     w.work_zone_id,
# MAGIC     w.zone_type,
# MAGIC     w.wz_route                                        AS route_id,
# MAGIC     w.speed_limit_in_zone,
# MAGIC     w.wz_incident_count,
# MAGIC     w.wz_fatalities,
# MAGIC     w.wz_injuries,
# MAGIC     ROUND(w.wz_avg_severity, 2)                       AS avg_severity,
# MAGIC     CASE
# MAGIC         WHEN w.wz_fatalities >= 1 THEN 'Fatal – Immediate Review Required'
# MAGIC         WHEN w.wz_incident_count >= 5 THEN 'High Frequency – Safety Audit'
# MAGIC         ELSE 'Monitor'
# MAGIC     END                                               AS recommended_action
# MAGIC FROM dot_geo.work_zone_incident_conflicts w
# MAGIC ORDER BY w.wz_fatalities DESC, w.wz_incident_count DESC;
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC -- ─────────────────────────────────────────────────────────────────────────────
# MAGIC -- Q6: Sensor Coverage Gap Analysis
# MAGIC -- Identifies H3 cells with high incident density but no active sensors
# MAGIC -- ─────────────────────────────────────────────────────────────────────────────
# MAGIC WITH incident_cells AS (
# MAGIC     SELECT h3_index_r8, incident_count, total_fatalities, risk_tier
# MAGIC     FROM dot_geo.h3_incident_density
# MAGIC     WHERE incident_count >= 5
# MAGIC ),
# MAGIC sensor_cells AS (
# MAGIC     SELECT DISTINCT h3_index_r8 AS h3_sensor
# MAGIC     FROM dot_geo.incidents_geo_enriched inc
# MAGIC     INNER JOIN (
# MAGIC         SELECT h3_index_r8 AS h3s
# MAGIC         FROM delta.`/Volumes/main/default/dot_lakehouse/bronze/geospatial/sensor_locations`
# MAGIC         WHERE operational_status = 'Active'
# MAGIC     ) s ON inc.h3_index_r8 = s.h3s
# MAGIC )
# MAGIC SELECT
# MAGIC     ic.h3_index_r8,
# MAGIC     ic.incident_count,
# MAGIC     ic.total_fatalities,
# MAGIC     ic.risk_tier,
# MAGIC     CASE WHEN sc.h3_sensor IS NULL THEN 'No Coverage' ELSE 'Covered' END AS sensor_coverage,
# MAGIC     CASE WHEN sc.h3_sensor IS NULL AND ic.total_fatalities >= 1
# MAGIC          THEN 'Priority Sensor Deployment'
# MAGIC          WHEN sc.h3_sensor IS NULL
# MAGIC          THEN 'Consider Sensor Deployment'
# MAGIC          ELSE 'Adequate Coverage'
# MAGIC     END AS sensor_recommendation
# MAGIC FROM incident_cells ic
# MAGIC LEFT JOIN sensor_cells sc ON ic.h3_index_r8 = sc.h3_sensor
# MAGIC ORDER BY ic.total_fatalities DESC, ic.incident_count DESC
# MAGIC LIMIT 40;
# MAGIC
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC -- ─────────────────────────────────────────────────────────────────────────────
# MAGIC -- Q7: Urban vs Rural Safety Comparison using area_type enrichment
# MAGIC -- ─────────────────────────────────────────────────────────────────────────────
# MAGIC SELECT
# MAGIC     area_type,
# MAGIC     state_code,
# MAGIC     COUNT(*)                                          AS total_incidents,
# MAGIC     SUM(fatalities)                                   AS total_fatalities,
# MAGIC     SUM(injuries)                                     AS total_injuries,
# MAGIC     ROUND(AVG(severity_score), 2)                     AS avg_severity,
# MAGIC     SUM(CASE WHEN incident_type = 'PEDESTRIAN' THEN 1 ELSE 0 END)  AS pedestrian_incidents,
# MAGIC     SUM(CASE WHEN incident_type = 'DUI'         THEN 1 ELSE 0 END)  AS dui_incidents,
# MAGIC     SUM(CASE WHEN is_weekend THEN 1 ELSE 0 END)       AS weekend_incidents,
# MAGIC     ROUND(SUM(fatalities) / COUNT(*), 4)              AS fatality_rate,
# MAGIC     ROUND(
# MAGIC         SUM(CASE WHEN incident_type = 'PEDESTRIAN' THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
# MAGIC         2
# MAGIC     )                                                 AS pedestrian_pct
# MAGIC FROM dot_geo.incidents_geo_enriched
# MAGIC GROUP BY 1,2
# MAGIC ORDER BY 2, 1;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC -- ─────────────────────────────────────────────────────────────────────────────
# MAGIC -- Q8: Native Databricks ST_ Spatial Functions (Photon engine)
# MAGIC -- Requires Databricks Runtime 11.3+ with Photon enabled
# MAGIC -- ─────────────────────────────────────────────────────────────────────────────
# MAGIC
# MAGIC -- Measure road segment length in meters using ST_Length on WGS84 geometry
# MAGIC SELECT
# MAGIC     segment_id,
# MAGIC     route_name,
# MAGIC     functional_class,
# MAGIC     aadt,
# MAGIC     geometry_wkt,
# MAGIC     ROUND(ST_Length(ST_GeomFromWKT(geometry_wkt)) * 111139, 0) AS approx_length_m,
# MAGIC     ST_X(ST_StartPoint(ST_GeomFromWKT(geometry_wkt)))           AS start_lon,
# MAGIC     ST_Y(ST_StartPoint(ST_GeomFromWKT(geometry_wkt)))           AS start_lat
# MAGIC FROM delta.`/Volumes/main/default/dot_lakehouse/bronze/geospatial/road_segments`
# MAGIC LIMIT 20;
# MAGIC
# MAGIC -- Check if incidents fall within TAZ polygons (point-in-polygon)
# MAGIC SELECT
# MAGIC     inc.incident_id,
# MAGIC     inc.incident_type,
# MAGIC     inc.severity,
# MAGIC     inc.latitude,
# MAGIC     inc.longitude,
# MAGIC     taz.taz_id,
# MAGIC     taz.zone_type,
# MAGIC     taz.population
# MAGIC FROM dot_geo.incidents_geo_enriched inc
# MAGIC CROSS JOIN delta.`/Volumes/main/default/dot_lakehouse/bronze/geospatial/traffic_analysis_zones` taz
# MAGIC WHERE ST_Contains(
# MAGIC     ST_GeomFromWKT(taz.geometry_wkt),
# MAGIC     ST_Point(inc.longitude, inc.latitude)
# MAGIC )
# MAGIC LIMIT 100;
# MAGIC
# MAGIC -- Buffer analysis: incidents within 0.5 degrees (~34 miles) of I-40 midpoint
# MAGIC SELECT
# MAGIC     inc.incident_id,
# MAGIC     inc.incident_type,
# MAGIC     inc.severity,
# MAGIC     inc.latitude,
# MAGIC     inc.longitude,
# MAGIC     ST_Distance(
# MAGIC         ST_Point(inc.longitude, inc.latitude),
# MAGIC         ST_Point(-80.0, 35.5)   -- approximate I-40 midpoint
# MAGIC     ) * 111139 AS distance_m_from_i40_mid
# MAGIC FROM dot_geo.incidents_geo_enriched inc
# MAGIC WHERE ST_Distance(
# MAGIC     ST_Point(inc.longitude, inc.latitude),
# MAGIC     ST_Point(-80.0, 35.5)
# MAGIC ) <= 0.5
# MAGIC ORDER BY distance_m_from_i40_mid
# MAGIC LIMIT 50;
# MAGIC
