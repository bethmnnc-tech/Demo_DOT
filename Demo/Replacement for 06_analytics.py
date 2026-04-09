# Databricks notebook source
# MAGIC %sql
# MAGIC -- =============================================================================
# MAGIC -- NOTEBOOK 6: DOT Analytics SQL – Dashboard & Reporting Queries
# MAGIC -- Databricks SQL Notebook
# MAGIC -- Description: Production-ready SQL queries for common DOT reporting needs.
# MAGIC --              Register as Named Queries or use in Databricks SQL Dashboards.
# MAGIC -- =============================================================================
# MAGIC
# MAGIC -- ─────────────────────────────────────────────────────────────────────────────
# MAGIC -- Q1: Statewide Safety Scorecard (Executive Dashboard)
# MAGIC -- ─────────────────────────────────────────────────────────────────────────────
# MAGIC SELECT
# MAGIC     s.state_code,
# MAGIC     s.total_incidents,
# MAGIC     s.total_fatalities,
# MAGIC     s.total_injuries,
# MAGIC     s.total_bridges,
# MAGIC     ROUND(s.avg_bridge_sufficiency, 1)         AS avg_bridge_sufficiency,
# MAGIC     s.deficient_bridges,
# MAGIC     s.bridge_deficiency_pct,
# MAGIC     ROUND(s.total_lane_miles, 0)               AS total_lane_miles,
# MAGIC     s.poor_pavement_pct,
# MAGIC     s.total_registered_vehicles,
# MAGIC     s.ev_adoption_pct,
# MAGIC     ROUND(s.fatality_per_100k_vehicles, 2)     AS fatality_per_100k_vehicles,
# MAGIC     CASE
# MAGIC         WHEN s.fatality_per_100k_vehicles > 15  THEN '🔴 High Risk'
# MAGIC         WHEN s.fatality_per_100k_vehicles > 8   THEN '🟡 Moderate Risk'
# MAGIC         ELSE                                         '🟢 Low Risk'
# MAGIC     END AS safety_grade,
# MAGIC     CASE
# MAGIC         WHEN s.poor_pavement_pct > 25 THEN '🔴 Critical'
# MAGIC         WHEN s.poor_pavement_pct > 15 THEN '🟡 Fair'
# MAGIC         ELSE                               '🟢 Good'
# MAGIC     END AS pavement_grade,
# MAGIC     CASE
# MAGIC         WHEN s.bridge_deficiency_pct > 20 THEN '🔴 Critical'
# MAGIC         WHEN s.bridge_deficiency_pct > 10 THEN '🟡 Fair'
# MAGIC         ELSE                                   '🟢 Good'
# MAGIC     END AS bridge_grade
# MAGIC FROM main.dot_gold.executive_kpi_scorecard s
# MAGIC ORDER BY s.total_fatalities DESC;
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC -- ─────────────────────────────────────────────────────────────────────────────
# MAGIC -- Q2: Top 20 High-Risk Road Corridors
# MAGIC -- ─────────────────────────────────────────────────────────────────────────────
# MAGIC SELECT
# MAGIC     h.state_code,
# MAGIC     h.county_code,
# MAGIC     h.route_id,
# MAGIC     h.incident_count,
# MAGIC     h.total_fatalities,
# MAGIC     h.total_injuries,
# MAGIC     ROUND(h.fatality_concentration, 3)   AS fatalities_per_incident,
# MAGIC     ROUND(h.avg_severity_score, 2)       AS avg_severity_score,
# MAGIC     h.risk_tier,
# MAGIC     h.state_rank,
# MAGIC     DATE_FORMAT(h.most_recent_incident, 'yyyy-MM-dd') AS last_incident_date,
# MAGIC     h.incident_type_diversity            AS num_incident_types
# MAGIC FROM main.dot_gold.high_risk_corridors h
# MAGIC WHERE h.risk_tier IN ('Top 10 – Critical', 'Top 25 – High Risk')
# MAGIC ORDER BY h.state_code, h.state_rank
# MAGIC LIMIT 20;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC
# MAGIC -- ─────────────────────────────────────────────────────────────────────────────
# MAGIC -- Q3: Incident Trend Analysis – Year-over-Year
# MAGIC -- ─────────────────────────────────────────────────────────────────────────────
# MAGIC WITH monthly_trend AS (
# MAGIC     SELECT
# MAGIC         state_code,
# MAGIC         incident_year,
# MAGIC         incident_month,
# MAGIC         CONCAT(incident_year, '-', LPAD(incident_month, 2, '0')) AS year_month,
# MAGIC         SUM(total_incidents)  AS total_incidents,
# MAGIC         SUM(total_fatalities) AS total_fatalities,
# MAGIC         SUM(total_injuries)   AS total_injuries,
# MAGIC         SUM(fatal_incidents)  AS fatal_incidents
# MAGIC     FROM main.dot_gold.incident_summary
# MAGIC     GROUP BY 1,2,3,4
# MAGIC ),
# MAGIC with_lag AS (
# MAGIC     SELECT
# MAGIC         *,
# MAGIC         LAG(total_incidents, 12) OVER (PARTITION BY state_code ORDER BY incident_year, incident_month)  AS incidents_prior_year,
# MAGIC         LAG(total_fatalities, 12) OVER (PARTITION BY state_code ORDER BY incident_year, incident_month) AS fatalities_prior_year
# MAGIC     FROM monthly_trend
# MAGIC )
# MAGIC SELECT
# MAGIC     state_code,
# MAGIC     year_month,
# MAGIC     total_incidents,
# MAGIC     total_fatalities,
# MAGIC     incidents_prior_year,
# MAGIC     fatalities_prior_year,
# MAGIC     ROUND((total_incidents - incidents_prior_year) / NULLIF(incidents_prior_year, 0) * 100, 1)  AS incident_yoy_pct,
# MAGIC     ROUND((total_fatalities - fatalities_prior_year) / NULLIF(fatalities_prior_year, 0) * 100, 1) AS fatality_yoy_pct,
# MAGIC     CASE WHEN total_incidents > incidents_prior_year THEN '▲' ELSE '▼' END AS incident_trend
# MAGIC FROM with_lag
# MAGIC WHERE incident_year >= 2021
# MAGIC ORDER BY state_code, year_month;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC
# MAGIC -- ─────────────────────────────────────────────────────────────────────────────
# MAGIC -- Q4: Bridge Infrastructure Investment Prioritization
# MAGIC -- ─────────────────────────────────────────────────────────────────────────────
# MAGIC SELECT
# MAGIC     b.state_code,
# MAGIC     b.county_code,
# MAGIC     b.bridge_type,
# MAGIC     b.age_category,
# MAGIC     b.priority_tier,
# MAGIC     b.bridge_count,
# MAGIC     ROUND(b.avg_sufficiency_rating, 1)             AS avg_sufficiency,
# MAGIC     ROUND(b.avg_bridge_age, 0)                     AS avg_age_years,
# MAGIC     b.structurally_deficient_count,
# MAGIC     b.functionally_obsolete_count,
# MAGIC     b.overdue_inspections,
# MAGIC     FORMAT_NUMBER(b.total_daily_traffic_exposed, 0) AS total_daily_traffic,
# MAGIC     FORMAT_NUMBER(b.total_estimated_repair_cost_m, 1) AS estimated_repair_cost_m_usd,
# MAGIC     ROUND(b.avg_risk_score, 1)                     AS avg_risk_score,
# MAGIC     ROUND(b.deficiency_rate * 100, 1)              AS deficiency_rate_pct,
# MAGIC     RANK() OVER (PARTITION BY b.state_code ORDER BY b.total_estimated_repair_cost_m DESC) AS cost_rank
# MAGIC FROM main.dot_gold.bridge_health_summary b
# MAGIC WHERE b.priority_tier IN ('Priority 1 – Immediate','Priority 2 – Near Term')
# MAGIC ORDER BY b.state_code, b.total_estimated_repair_cost_m DESC;
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC -- ─────────────────────────────────────────────────────────────────────────────
# MAGIC -- Q5: Pavement Needs by Functional Class – Capital Program Input
# MAGIC -- ─────────────────────────────────────────────────────────────────────────────
# MAGIC SELECT
# MAGIC     p.state_code,
# MAGIC     p.functional_class,
# MAGIC     p.maintenance_priority,
# MAGIC     COUNT(*)                                          AS segment_groups,
# MAGIC     ROUND(SUM(p.total_lane_miles), 1)                 AS total_lane_miles,
# MAGIC     ROUND(AVG(p.avg_pcr), 1)                          AS avg_pcr,
# MAGIC     ROUND(AVG(p.avg_iri), 0)                          AS avg_iri,
# MAGIC     ROUND(SUM(p.total_estimated_cost_usd) / 1e6, 2)   AS total_cost_million_usd,
# MAGIC     ROUND(SUM(p.total_estimated_cost_usd) / NULLIF(SUM(p.total_lane_miles), 0) / 1000, 1) AS cost_per_lane_mile_k,
# MAGIC     FORMAT_NUMBER(SUM(p.total_aadt), 0)               AS total_aadt_exposed
# MAGIC FROM main.dot_gold.pavement_needs_assessment p
# MAGIC GROUP BY 1,2,3
# MAGIC ORDER BY p.state_code, p.functional_class,
# MAGIC     CASE p.maintenance_priority
# MAGIC         WHEN 'Immediate Rehab'       THEN 1
# MAGIC         WHEN 'Preventive Treatment'  THEN 2
# MAGIC         WHEN 'Monitor'               THEN 3
# MAGIC         ELSE 4
# MAGIC     END;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC
# MAGIC -- ─────────────────────────────────────────────────────────────────────────────
# MAGIC -- Q6: Fleet EV Adoption & Emissions Trend
# MAGIC -- ─────────────────────────────────────────────────────────────────────────────
# MAGIC SELECT
# MAGIC     f.state_code,
# MAGIC     f.model_year,
# MAGIC     f.vehicle_class,
# MAGIC     SUM(f.registered_vehicles)                        AS total_vehicles,
# MAGIC     SUM(f.electric_count)                             AS electric_vehicles,
# MAGIC     SUM(f.commercial_count)                           AS commercial_vehicles,
# MAGIC     SUM(f.hazmat_count)                               AS hazmat_vehicles,
# MAGIC     SUM(f.expired_registrations)                      AS expired_registrations,
# MAGIC     ROUND(SUM(f.electric_count) / SUM(f.registered_vehicles) * 100, 2) AS ev_share_pct,
# MAGIC     ROUND(SUM(f.expired_registrations) / SUM(f.registered_vehicles) * 100, 2) AS expiry_rate_pct
# MAGIC FROM main.dot_gold.fleet_snapshot f
# MAGIC WHERE f.model_year >= 2015
# MAGIC GROUP BY 1,2,3
# MAGIC ORDER BY f.state_code, f.model_year DESC, f.vehicle_class;
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC -- ─────────────────────────────────────────────────────────────────────────────
# MAGIC -- Q7: Incident Hot-Spot Heat Map (hourly × day-of-week)
# MAGIC -- ─────────────────────────────────────────────────────────────────────────────
# MAGIC SELECT
# MAGIC     i.day_of_week,
# MAGIC     CASE i.day_of_week
# MAGIC         WHEN 1 THEN 'Sunday'    WHEN 2 THEN 'Monday'  WHEN 3 THEN 'Tuesday'
# MAGIC         WHEN 4 THEN 'Wednesday' WHEN 5 THEN 'Thursday' WHEN 6 THEN 'Friday'
# MAGIC         ELSE 'Saturday'
# MAGIC     END AS day_name,
# MAGIC     i.incident_hour,
# MAGIC     COUNT(*)                  AS incident_count,
# MAGIC     SUM(i.fatalities)         AS total_fatalities,
# MAGIC     ROUND(AVG(i.severity_score), 2) AS avg_severity
# MAGIC FROM main.dot_silver.traffic_incidents i
# MAGIC GROUP BY 1,2,3
# MAGIC ORDER BY 1, 3;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC
# MAGIC -- ─────────────────────────────────────────────────────────────────────────────
# MAGIC -- Q8: Delta Table History & Data Lineage Audit
# MAGIC -- ─────────────────────────────────────────────────────────────────────────────
# MAGIC DESCRIBE HISTORY main.dot_silver.traffic_incidents;
# MAGIC
# MAGIC -- Table statistics
# MAGIC ANALYZE TABLE main.dot_silver.traffic_incidents COMPUTE STATISTICS FOR ALL COLUMNS;
# MAGIC
# MAGIC -- Vacuum old Delta files (retain 7 days of history)
# MAGIC -- VACUUM main.dot_silver.traffic_incidents RETAIN 168 HOURS;
# MAGIC
# MAGIC -- Optimize for query performance
# MAGIC OPTIMIZE main.dot_silver.traffic_incidents ZORDER BY (state_code, incident_year, route_id);
# MAGIC OPTIMIZE main.dot_silver.bridge_inspections ZORDER BY (state_code, risk_score);
# MAGIC OPTIMIZE main.dot_silver.pavement_conditions ZORDER BY (state_code, functional_class, distress_index);
