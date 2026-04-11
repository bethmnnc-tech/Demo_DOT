-- =============================================================================
-- NOTEBOOK 7d: DOT Geospatial SQL Queries
-- Databricks SQL Notebook | Uses ${catalog}.dot_geo database
-- Description: Spatial analytics queries using H3 functions and WKT geometry.
--              Databricks supports ST_* functions natively via Photon engine.
-- =============================================================================

-- Widget for catalog selection (dev=main, prod=main_prod)
CREATE WIDGET TEXT catalog DEFAULT "main";
CREATE WIDGET TEXT base_path DEFAULT "/Volumes/main/default/dot_lakehouse";

-- ─────────────────────────────────────────────────────────────────────────────
-- Q1: Incident Hotspot Summary – Top H3 Cells by Fatality Concentration
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    h.h3_index_r8,
    h.state_code,
    h.incident_count,
    h.total_fatalities,
    h.total_injuries,
    ROUND(h.avg_severity_score, 2)          AS avg_severity,
    h.fatal_incident_count,
    h.night_incident_count,
    h.weekend_incident_count,
    ROUND(h.incident_density_per_km2, 2)    AS density_per_km2,
    ROUND(h.z_score, 2)                     AS hotspot_z_score,
    h.hotspot_class,
    h.risk_tier
FROM ${catalog}.dot_geo.incident_hotspots h
WHERE h.is_hotspot = TRUE
ORDER BY h.total_fatalities DESC, h.incident_count DESC
LIMIT 50;


-- ─────────────────────────────────────────────────────────────────────────────
-- Q2: Corridor Safety Rate (incidents per 100M VMT)
-- Industry standard: compares corridors on a traffic-normalized basis
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    c.seg_route_id                                    AS route_id,
    c.functional_class,
    c.area_type,
    c.incidents_on_corridor,
    c.fatalities_on_corridor,
    c.injuries_on_corridor,
    FORMAT_NUMBER(c.reference_aadt, 0)                AS reference_aadt,
    ROUND(c.avg_severity, 2)                          AS avg_severity_score,
    ROUND(c.incidents_per_100m_vmt, 4)                AS incidents_per_100m_vmt,
    RANK() OVER (
        PARTITION BY c.functional_class
        ORDER BY c.incidents_per_100m_vmt DESC
    )                                                 AS safety_rank_in_class
FROM ${catalog}.dot_geo.corridor_safety_rates c
ORDER BY c.incidents_per_100m_vmt DESC
LIMIT 30;


-- ─────────────────────────────────────────────────────────────────────────────
-- Q3: Bridge Risk by Traffic Analysis Zone
-- Identifies TAZs where structurally deficient bridges expose most population
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    b.taz_id,
    b.zone_type,
    b.state_code,
    COUNT(b.bridge_id)                                AS bridge_count,
    SUM(CASE WHEN b.worst_condition = 'Poor' THEN 1 ELSE 0 END) AS poor_condition_bridges,
    ROUND(AVG(b.risk_score), 1)                       AS avg_risk_score,
    ROUND(AVG(b.sufficiency_rating), 1)               AS avg_sufficiency,
    MAX(b.taz_population)                             AS taz_population,
    MAX(b.taz_vehicle_trips)                          AS taz_vehicle_trips,
    ROUND(SUM(b.population_at_risk_score), 0)         AS total_pop_at_risk_score,
    SUM(b.avg_daily_traffic)                          AS total_bridge_aadt,
    COLLECT_LIST(CASE WHEN b.priority_tier = 'Priority 1 – Immediate' THEN b.bridge_id END)
                                                      AS p1_bridge_ids
FROM ${catalog}.dot_geo.bridges_with_taz b
GROUP BY 1,2,3
HAVING COUNT(b.bridge_id) >= 2
ORDER BY total_pop_at_risk_score DESC
LIMIT 25;


-- ─────────────────────────────────────────────────────────────────────────────
-- Q4: H3 Spatial Roll-Up – County-Level Density (resolution 6)
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    h3_index_r6                                       AS h3_county_cell,
    state_code,
    COUNT(DISTINCT h3_index_r8)                       AS sub_cells_count,
    SUM(incident_count)                               AS total_incidents,
    SUM(total_fatalities)                             AS total_fatalities,
    SUM(fatal_incident_count)                         AS fatal_incidents,
    SUM(night_incident_count)                         AS night_incidents,
    SUM(pedestrian_count)                             AS pedestrian_incidents,
    SUM(dui_count)                                    AS dui_incidents,
    ROUND(SUM(incident_count) / COUNT(DISTINCT h3_index_r8), 1) AS avg_incidents_per_cell,
    ROUND(AVG(avg_severity_score), 2)                 AS avg_severity,
    SUM(CASE WHEN risk_tier LIKE '%Critical%' OR risk_tier LIKE '%High%' THEN 1 ELSE 0 END)
                                                      AS high_risk_sub_cells
FROM ${catalog}.dot_geo.h3_incident_density
GROUP BY 1,2
ORDER BY total_fatalities DESC;


-- ─────────────────────────────────────────────────────────────────────────────
-- Q5: Work Zone Safety Analysis – Incidents inside active construction zones
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    w.work_zone_id,
    w.zone_type,
    w.wz_route                                        AS route_id,
    w.speed_limit_in_zone,
    w.wz_incident_count,
    w.wz_fatalities,
    w.wz_injuries,
    ROUND(w.wz_avg_severity, 2)                       AS avg_severity,
    CASE
        WHEN w.wz_fatalities >= 1 THEN 'Fatal – Immediate Review Required'
        WHEN w.wz_incident_count >= 5 THEN 'High Frequency – Safety Audit'
        ELSE 'Monitor'
    END                                               AS recommended_action
FROM ${catalog}.dot_geo.work_zone_incident_conflicts w
ORDER BY w.wz_fatalities DESC, w.wz_incident_count DESC;


-- ─────────────────────────────────────────────────────────────────────────────
-- Q6: Sensor Coverage Gap Analysis
-- Identifies H3 cells with high incident density but no active sensors
-- ─────────────────────────────────────────────────────────────────────────────
WITH incident_cells AS (
    SELECT h3_index_r8, incident_count, total_fatalities, risk_tier
    FROM ${catalog}.dot_geo.h3_incident_density
    WHERE incident_count >= 5
),
sensor_cells AS (
    SELECT DISTINCT h3_index_r8 AS h3_sensor
    FROM ${catalog}.dot_geo.incidents_geo_enriched inc
    INNER JOIN (
        SELECT h3_index_r8 AS h3s
        FROM delta.`${base_path}/bronze/geospatial/sensor_locations`
        WHERE operational_status = 'Active'
    ) s ON inc.h3_index_r8 = s.h3s
)
SELECT
    ic.h3_index_r8,
    ic.incident_count,
    ic.total_fatalities,
    ic.risk_tier,
    CASE WHEN sc.h3_sensor IS NULL THEN 'No Coverage' ELSE 'Covered' END AS sensor_coverage,
    CASE WHEN sc.h3_sensor IS NULL AND ic.total_fatalities >= 1
         THEN 'Priority Sensor Deployment'
         WHEN sc.h3_sensor IS NULL
         THEN 'Consider Sensor Deployment'
         ELSE 'Adequate Coverage'
    END AS sensor_recommendation
FROM incident_cells ic
LEFT JOIN sensor_cells sc ON ic.h3_index_r8 = sc.h3_sensor
ORDER BY ic.total_fatalities DESC, ic.incident_count DESC
LIMIT 40;


-- ─────────────────────────────────────────────────────────────────────────────
-- Q7: Urban vs Rural Safety Comparison using area_type enrichment
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    area_type,
    state_code,
    COUNT(*)                                          AS total_incidents,
    SUM(fatalities)                                   AS total_fatalities,
    SUM(injuries)                                     AS total_injuries,
    ROUND(AVG(severity_score), 2)                     AS avg_severity,
    SUM(CASE WHEN incident_type = 'PEDESTRIAN' THEN 1 ELSE 0 END)  AS pedestrian_incidents,
    SUM(CASE WHEN incident_type = 'DUI'         THEN 1 ELSE 0 END)  AS dui_incidents,
    SUM(CASE WHEN is_weekend THEN 1 ELSE 0 END)       AS weekend_incidents,
    ROUND(SUM(fatalities) / COUNT(*), 4)              AS fatality_rate,
    ROUND(
        SUM(CASE WHEN incident_type = 'PEDESTRIAN' THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
        2
    )                                                 AS pedestrian_pct
FROM ${catalog}.dot_geo.incidents_geo_enriched
GROUP BY 1,2
ORDER BY 2, 1;


-- ─────────────────────────────────────────────────────────────────────────────
-- Q8: Native Databricks ST_ Spatial Functions (Photon engine)
-- Requires Databricks Runtime 11.3+ with Photon enabled
-- ─────────────────────────────────────────────────────────────────────────────

-- Measure road segment length in meters using ST_Length on WGS84 geometry
SELECT
    segment_id,
    route_name,
    functional_class,
    aadt,
    speed_limit_mph,
    geometry_wkt,
    ROUND(ST_Length(ST_GeomFromWKT(geometry_wkt)) * 111139, 0) AS approx_length_m,
    ST_X(ST_StartPoint(ST_GeomFromWKT(geometry_wkt)))           AS start_lon,
    ST_Y(ST_StartPoint(ST_GeomFromWKT(geometry_wkt)))           AS start_lat
FROM delta.`${base_path}/bronze/geospatial/road_segments`
LIMIT 20;

-- Check if incidents fall within TAZ polygons (point-in-polygon)
SELECT
    inc.incident_id,
    inc.incident_type,
    inc.severity,
    inc.latitude,
    inc.longitude,
    taz.taz_id,
    taz.zone_type,
    taz.population
FROM ${catalog}.dot_geo.incidents_geo_enriched inc
CROSS JOIN delta.`${base_path}/bronze/geospatial/traffic_analysis_zones` taz
WHERE ST_Contains(
    ST_GeomFromWKT(taz.geometry_wkt),
    ST_Point(inc.longitude, inc.latitude)
)
LIMIT 100;

-- Buffer analysis: incidents within 0.5 degrees (~34 miles) of I-40 midpoint
SELECT
    inc.incident_id,
    inc.incident_type,
    inc.severity,
    inc.latitude,
    inc.longitude,
    ST_Distance(
        ST_Point(inc.longitude, inc.latitude),
        ST_Point(-80.0, 35.5)   -- approximate I-40 midpoint
    ) * 111139 AS distance_m_from_i40_mid
FROM ${catalog}.dot_geo.incidents_geo_enriched inc
WHERE ST_Distance(
    ST_Point(inc.longitude, inc.latitude),
    ST_Point(-80.0, 35.5)
) <= 0.5
ORDER BY distance_m_from_i40_mid
LIMIT 50;
