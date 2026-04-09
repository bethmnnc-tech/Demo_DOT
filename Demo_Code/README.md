# DOT Transportation Data Platform – Databricks Reference Architecture

## Overview
A complete Lakehouse data engineering solution for Department of Transportation
data management, analytics, and predictive modeling built on Databricks + Delta Lake.

---

## Architecture: Medallion Lakehouse

```
Landing Zone          Bronze Layer          Silver Layer          Gold Layer
─────────────         ────────────          ────────────          ──────────
Raw JSON/CSV/  ──▶   Raw Delta tables  ──▶  Validated +     ──▶  Aggregated
Parquet files         (append-only)         Enriched tables       KPI tables
from sensors,
APIs, batch
uploads
```

---

## Data Domains Covered

| Domain               | Records    | Source Analog            | Primary Use Case                    |
|----------------------|------------|--------------------------|-------------------------------------|
| Traffic Incidents    | 50,000     | FHWA / State DOT Crash   | Safety analysis, trend reporting    |
| Bridge Inspections   | 10,000     | NBI (National Bridge Inv)| Asset management, capital planning  |
| Vehicle Registrations| 50,000     | FMCSA / DMV              | Fleet analytics, emissions tracking |
| Pavement Conditions  | 20,000     | FHWA HPMS                | Maintenance planning, CIP           |
| Sensor Stream        | Continuous | Traffic Management Center| Real-time congestion monitoring     |

---

## Notebook Execution Order

```
01_generate_sample_data.py      → Creates all Bronze Delta tables
02_silver_transformations.py    → DQ checks + enrichment → Silver
03_gold_analytics.py            → Aggregations + KPI tables → Gold
04_ml_models.py                 → MLflow model training & logging
05_streaming_dlt.py             → Structured Streaming + DLT pipeline
06_analytics_queries.sql        → Dashboard & reporting SQL
```

---

## Delta Table Storage Layout

```
dbfs:/dot_lakehouse/
├── bronze/
│   ├── traffic_incidents/          (partitioned by state_code, incident_type)
│   ├── bridge_inspections/         (partitioned by state_code)
│   ├── vehicle_registrations/      (partitioned by state_code, vehicle_class)
│   └── pavement_conditions/        (partitioned by state_code, functional_class)
├── silver/
│   ├── traffic_incidents/          (partitioned by state_code, year, month)
│   ├── bridge_inspections/         (partitioned by state_code)
│   ├── vehicle_registrations/      (partitioned by state_code, vehicle_class)
│   └── pavement_conditions/        (partitioned by state_code, functional_class)
├── gold/
│   ├── incident_summary/
│   ├── high_risk_corridors/
│   ├── bridge_health_summary/
│   ├── fleet_snapshot/
│   ├── pavement_needs_assessment/
│   └── executive_kpi_scorecard/
├── stream_bronze/
│   └── sensor_readings/
├── stream_silver/
│   ├── sensor_aggregated/
│   └── traffic_anomalies/
└── models/
    ├── incident_severity_gbt/
    ├── bridge_risk_gbt/
    └── pavement_deterioration_rf/
```

---

## Hive Metastore / Unity Catalog Databases

| Database      | Layer  | Description                              |
|---------------|--------|------------------------------------------|
| dot_silver    | Silver | Validated, enriched operational tables   |
| dot_gold      | Gold   | Aggregated marts for dashboards/reports  |
| dot_dlt       | All    | Delta Live Tables managed tables         |

---

## ML Models (MLflow)

| Model                          | Algorithm    | Target Variable    | Key Metrics         |
|--------------------------------|--------------|--------------------|---------------------|
| Incident Severity Classifier   | GBT Classify | Severity class     | Accuracy, F1        |
| Bridge Risk Score Regressor    | GBT Regressor| risk_score (0-100) | RMSE, R², MAE       |
| Pavement Deterioration Predict | Random Forest| Poor condition flag| AUC-ROC, AUC-PR     |

---

## Key Derived Fields (Silver Layer)

### Traffic Incidents
- `incident_year / month / hour` – temporal decomposition
- `is_weekend` – weekend flag
- `has_fatality` – boolean fatal incident
- `total_casualties` – fatalities + injuries
- `severity_score` – numeric 1–5 ordinal

### Bridge Inspections  
- `bridge_age_years` – 2024 − year_built
- `age_category` – New / Mature / Aging / Old
- `worst_condition` – derived from deck + super + sub
- `risk_score` – composite scoring formula
- `priority_tier` – Priority 1–4 investment tier
- `days_since_inspection` – freshness indicator
- `inspection_overdue` – > 730 days = overdue

### Vehicle Registrations
- `vehicle_age` – 2024 − model_year
- `weight_class` – FHWA Class 1–8
- `is_electric` – EV/Hybrid flag
- `is_expired` – registration currency

### Pavement Conditions
- `pavement_age` – 2024 − year_constructed
- `iri_category` – Very Good → Poor (IRI thresholds)
- `distress_index` – composite: cracking + rutting + PCR
- `maintenance_priority` – Immediate Rehab → No Action
- `traffic_category` – High / Medium / Low volume

---

## Data Quality Framework

Each Silver transformation includes DQ checks with configurable thresholds:
- **Null checks** (threshold = 0%): Primary keys, timestamps
- **Range checks** (threshold = 1%): Coordinates, ratings, scores
- **Referential checks**: Code lists validation
- Results are printed as a DQ report; extend to log to a `dot_dq_log` Delta table

---

## Streaming Architecture

```
External Sensors / TMC APIs
         │
         ▼
 Auto Loader (cloudFiles)       ← JSON events, 30-second micro-batches
         │
         ▼
  Bronze: sensor_readings        ← Raw append-only + file provenance
         │
         ├──▶ Silver: sensor_aggregated    ← 5-min tumbling window agg
         │           (avg speed, volume,   ← Watermark: 10 min late tolerance
         │            congestion level)
         │
         └──▶ Silver: traffic_anomalies   ← Real-time anomaly flagging
                      (speed < 10 mph,      ← 30-second trigger
                       occupancy > 95%)
```

---

## Recommended Databricks Configuration

```yaml
Cluster:
  Type: Job Cluster (for notebooks 01-04) or All-Purpose Cluster
  DBR: 14.3 LTS ML (includes MLflow, Delta Lake, Spark 3.5)
  Node Type: i3.xlarge (driver) + i3.2xlarge (workers)
  Auto-scaling: 2–8 workers
  Spot Instances: enabled for workers

Streaming Cluster:
  Type: Job Cluster, always-on
  DBR: 14.3 LTS
  Node Type: r5.large

Unity Catalog:
  Metastore: enabled
  Catalog: dot_prod
  Schemas: bronze, silver, gold, streaming, models
```

---

## Suggested Workflow Schedule (Databricks Jobs)

```
Daily 02:00 AM   →  01_generate_sample_data.py  (or real ingestion)
Daily 02:30 AM   →  02_silver_transformations.py
Daily 03:00 AM   →  03_gold_analytics.py
Weekly Sunday    →  04_ml_models.py              (model retraining)
Continuous       →  05_streaming_dlt.py          (always running)
```

---

## Dashboard Recommendations (Databricks SQL)

| Dashboard             | Key Queries        | Refresh      |
|-----------------------|--------------------|--------------|
| Safety Scorecard      | Q1, Q2, Q7         | Daily        |
| Bridge Asset Health   | Q4                 | Weekly       |
| Pavement CIP          | Q5                 | Monthly      |
| Fleet & Emissions     | Q6                 | Daily        |
| YoY Trend Analysis    | Q3                 | Monthly      |
| Real-Time Traffic     | Streaming tables   | Near-realtime|
