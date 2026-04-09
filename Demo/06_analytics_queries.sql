-- =============================================================================
-- NOTEBOOK 6: DOT Analytics SQL – Dashboard & Reporting Queries
-- Databricks SQL Notebook
-- Description: Production-ready SQL queries for common DOT reporting needs.
--              Register as Named Queries or use in Databricks SQL Dashboards.
-- =============================================================================

-- ─────────────────────────────────────────────────────────────────────────────
-- Q1: Statewide Safety Scorecard (Executive Dashboard)
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    s.state_code,
    s.total_incidents,
    s.total_fatalities,
    s.total_injuries,
    s.total_bridges,
    ROUND(s.avg_bridge_sufficiency, 1)         AS avg_bridge_sufficiency,
    s.deficient_bridges,
    s.bridge_deficiency_pct,
    ROUND(s.total_lane_miles, 0)               AS total_lane_miles,
    s.poor_pavement_pct,
    s.total_registered_vehicles,
    s.ev_adoption_pct,
    ROUND(s.fatality_per_100k_vehicles, 2)     AS fatality_per_100k_vehicles,
    CASE
        WHEN s.fatality_per_100k_vehicles > 15  THEN '🔴 High Risk'
        WHEN s.fatality_per_100k_vehicles > 8   THEN '🟡 Moderate Risk'
        ELSE                                         '🟢 Low Risk'
    END AS safety_grade,
    CASE
        WHEN s.poor_pavement_pct > 25 THEN '🔴 Critical'
        WHEN s.poor_pavement_pct > 15 THEN '🟡 Fair'
        ELSE                               '🟢 Good'
    END AS pavement_grade,
    CASE
        WHEN s.bridge_deficiency_pct > 20 THEN '🔴 Critical'
        WHEN s.bridge_deficiency_pct > 10 THEN '🟡 Fair'
        ELSE                                   '🟢 Good'
    END AS bridge_grade
FROM dot_gold.executive_kpi_scorecard s
ORDER BY s.total_fatalities DESC;


-- ─────────────────────────────────────────────────────────────────────────────
-- Q2: Top 20 High-Risk Road Corridors
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    h.state_code,
    h.county_code,
    h.route_id,
    h.incident_count,
    h.total_fatalities,
    h.total_injuries,
    ROUND(h.fatality_concentration, 3)   AS fatalities_per_incident,
    ROUND(h.avg_severity_score, 2)       AS avg_severity_score,
    h.risk_tier,
    h.state_rank,
    DATE_FORMAT(h.most_recent_incident, 'yyyy-MM-dd') AS last_incident_date,
    h.incident_type_diversity            AS num_incident_types
FROM dot_gold.high_risk_corridors h
WHERE h.risk_tier IN ('Top 10 – Critical', 'Top 25 – High Risk')
ORDER BY h.state_code, h.state_rank
LIMIT 20;


-- ─────────────────────────────────────────────────────────────────────────────
-- Q3: Incident Trend Analysis – Year-over-Year
-- ─────────────────────────────────────────────────────────────────────────────
WITH monthly_trend AS (
    SELECT
        state_code,
        incident_year,
        incident_month,
        CONCAT(incident_year, '-', LPAD(incident_month, 2, '0')) AS year_month,
        SUM(total_incidents)  AS total_incidents,
        SUM(total_fatalities) AS total_fatalities,
        SUM(total_injuries)   AS total_injuries,
        SUM(fatal_incidents)  AS fatal_incidents
    FROM dot_gold.incident_summary
    GROUP BY 1,2,3,4
),
with_lag AS (
    SELECT
        *,
        LAG(total_incidents, 12) OVER (PARTITION BY state_code ORDER BY incident_year, incident_month)  AS incidents_prior_year,
        LAG(total_fatalities, 12) OVER (PARTITION BY state_code ORDER BY incident_year, incident_month) AS fatalities_prior_year
    FROM monthly_trend
)
SELECT
    state_code,
    year_month,
    total_incidents,
    total_fatalities,
    incidents_prior_year,
    fatalities_prior_year,
    ROUND((total_incidents - incidents_prior_year) / NULLIF(incidents_prior_year, 0) * 100, 1)  AS incident_yoy_pct,
    ROUND((total_fatalities - fatalities_prior_year) / NULLIF(fatalities_prior_year, 0) * 100, 1) AS fatality_yoy_pct,
    CASE WHEN total_incidents > incidents_prior_year THEN '▲' ELSE '▼' END AS incident_trend
FROM with_lag
WHERE incident_year >= 2021
ORDER BY state_code, year_month;


-- ─────────────────────────────────────────────────────────────────────────────
-- Q4: Bridge Infrastructure Investment Prioritization
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    b.state_code,
    b.county_code,
    b.bridge_type,
    b.age_category,
    b.priority_tier,
    b.bridge_count,
    ROUND(b.avg_sufficiency_rating, 1)             AS avg_sufficiency,
    ROUND(b.avg_bridge_age, 0)                     AS avg_age_years,
    b.structurally_deficient_count,
    b.functionally_obsolete_count,
    b.overdue_inspections,
    FORMAT_NUMBER(b.total_daily_traffic_exposed, 0) AS total_daily_traffic,
    FORMAT_NUMBER(b.total_estimated_repair_cost_m, 1) AS estimated_repair_cost_m_usd,
    ROUND(b.avg_risk_score, 1)                     AS avg_risk_score,
    ROUND(b.deficiency_rate * 100, 1)              AS deficiency_rate_pct,
    RANK() OVER (PARTITION BY b.state_code ORDER BY b.total_estimated_repair_cost_m DESC) AS cost_rank
FROM dot_gold.bridge_health_summary b
WHERE b.priority_tier IN ('Priority 1 – Immediate','Priority 2 – Near Term')
ORDER BY b.state_code, b.total_estimated_repair_cost_m DESC;


-- ─────────────────────────────────────────────────────────────────────────────
-- Q5: Pavement Needs by Functional Class – Capital Program Input
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    p.state_code,
    p.functional_class,
    p.maintenance_priority,
    COUNT(*)                                          AS segment_groups,
    ROUND(SUM(p.total_lane_miles), 1)                 AS total_lane_miles,
    ROUND(AVG(p.avg_pcr), 1)                          AS avg_pcr,
    ROUND(AVG(p.avg_iri), 0)                          AS avg_iri,
    ROUND(SUM(p.total_estimated_cost_usd) / 1e6, 2)   AS total_cost_million_usd,
    ROUND(SUM(p.total_estimated_cost_usd) / NULLIF(SUM(p.total_lane_miles), 0) / 1000, 1) AS cost_per_lane_mile_k,
    FORMAT_NUMBER(SUM(p.total_aadt), 0)               AS total_aadt_exposed
FROM dot_gold.pavement_needs_assessment p
GROUP BY 1,2,3
ORDER BY p.state_code, p.functional_class,
    CASE p.maintenance_priority
        WHEN 'Immediate Rehab'       THEN 1
        WHEN 'Preventive Treatment'  THEN 2
        WHEN 'Monitor'               THEN 3
        ELSE 4
    END;


-- ─────────────────────────────────────────────────────────────────────────────
-- Q6: Fleet EV Adoption & Emissions Trend
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    f.state_code,
    f.model_year,
    f.vehicle_class,
    SUM(f.registered_vehicles)                        AS total_vehicles,
    SUM(f.electric_count)                             AS electric_vehicles,
    SUM(f.commercial_count)                           AS commercial_vehicles,
    SUM(f.hazmat_count)                               AS hazmat_vehicles,
    SUM(f.expired_registrations)                      AS expired_registrations,
    ROUND(SUM(f.electric_count) / SUM(f.registered_vehicles) * 100, 2) AS ev_share_pct,
    ROUND(SUM(f.expired_registrations) / SUM(f.registered_vehicles) * 100, 2) AS expiry_rate_pct
FROM dot_gold.fleet_snapshot f
WHERE f.model_year >= 2015
GROUP BY 1,2,3
ORDER BY f.state_code, f.model_year DESC, f.vehicle_class;


-- ─────────────────────────────────────────────────────────────────────────────
-- Q7: Incident Hot-Spot Heat Map (hourly × day-of-week)
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    i.day_of_week,
    CASE i.day_of_week
        WHEN 1 THEN 'Sunday'    WHEN 2 THEN 'Monday'  WHEN 3 THEN 'Tuesday'
        WHEN 4 THEN 'Wednesday' WHEN 5 THEN 'Thursday' WHEN 6 THEN 'Friday'
        ELSE 'Saturday'
    END AS day_name,
    i.incident_hour,
    COUNT(*)                  AS incident_count,
    SUM(i.fatalities)         AS total_fatalities,
    ROUND(AVG(i.severity_score), 2) AS avg_severity
FROM dot_silver.traffic_incidents i
GROUP BY 1,2,3
ORDER BY 1, 3;


-- ─────────────────────────────────────────────────────────────────────────────
-- Q8: Delta Table History & Data Lineage Audit
-- ─────────────────────────────────────────────────────────────────────────────
DESCRIBE HISTORY dot_silver.traffic_incidents;

-- Table statistics
ANALYZE TABLE dot_silver.traffic_incidents COMPUTE STATISTICS FOR ALL COLUMNS;

-- Vacuum old Delta files (retain 7 days of history)
-- VACUUM dot_silver.traffic_incidents RETAIN 168 HOURS;

-- Optimize for query performance
OPTIMIZE dot_silver.traffic_incidents ZORDER BY (state_code, incident_year, route_id);
OPTIMIZE dot_silver.bridge_inspections ZORDER BY (state_code, risk_score);
OPTIMIZE dot_silver.pavement_conditions ZORDER BY (state_code, functional_class, distress_index);
