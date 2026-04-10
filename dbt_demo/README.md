# dbt Dual-Target Demo

This folder is a standalone `dbt Core` demo layered on top of the existing DOT transportation project. It is additive only: nothing in `Demo_Code/`, `dot_app/`, or the current Databricks notebooks depends on this folder.

## What This Demo Shows

One shared dbt project can manage transformation logic for two deployment targets:

- `databricks`: build Silver and Gold models as Delta-backed tables through `dbt-databricks`
- `onprem`: show the same logic targeting an on-prem SQL Server warehouse through `dbt-sqlserver`

The model logic lives in one place. Platform differences are pushed into a small macro layer.

## Scope

This demo covers two source domains:

- traffic incidents
- bridge inspections

It builds:

- Silver:
  - `silver_traffic_incidents`
  - `silver_bridge_inspections`
- Gold:
  - `gold_incident_summary`
  - `gold_high_risk_corridors`
  - `gold_bridge_health_summary`

## Project Layout

```text
dbt_demo/
  macros/                  Cross-platform helper macros
  models/
    sources.yml            Shared source definitions
    silver/                Shared Silver transformations
    gold/                  Shared Gold marts
  bootstrap_databricks_sources.sql
  profiles.example.yml
  dbt_project.yml
```

## How It Fits This Repo

The original notebooks create and transform Delta data directly. This dbt demo starts at the transformation layer and assumes raw Bronze-style tables are already queryable as database objects.

For a live Databricks demo, register the Bronze Delta locations as external tables first. The helper SQL is in [bootstrap_databricks_sources.sql](/C:/Users/Frank.Matteson/Documents/GitHub/Demo_DOT/dbt_demo/bootstrap_databricks_sources.sql).

Those registrations are additive and do not change the existing notebook pipeline.

## Example Databricks Bootstrap

The notebooks write Bronze data to:

- `/Volumes/main/default/dot_lakehouse/bronze/traffic_incidents`
- `/Volumes/main/default/dot_lakehouse/bronze/bridge_inspections`

Run the bootstrap SQL in Databricks to register them under `main.bronze`.

## Setup

Install one or both adapters:

```bash
pip install dbt-core dbt-databricks
```

Optional on-prem adapter for architecture demos:

```bash
pip install dbt-sqlserver
```

This repo now includes a workspace-runnable profile at [profiles.yml](/C:/Users/Frank.Matteson/Documents/GitHub/Demo_DOT/dbt_demo/profiles.yml). For local development, you can either:

- point dbt at the project folder as the profiles directory, or
- copy [profiles.example.yml](/C:/Users/Frank.Matteson/Documents/GitHub/Demo_DOT/dbt_demo/profiles.example.yml) into your local dbt profiles directory as `profiles.yml`

Then set:

```bash
export DBT_DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/<warehouse-id>
export DATABRICKS_TOKEN=<your-token>
```

For local profile details, see [profiles/README.md](/C:/Users/Frank.Matteson/Documents/GitHub/Demo_DOT/dbt_demo/profiles/README.md).

If you run dbt from a Databricks Workflow `dbt_task`, Databricks injects `DBT_ACCESS_TOKEN` automatically for the configured Run As principal. The committed profile template supports both local `DATABRICKS_TOKEN` usage and workflow-injected `DBT_ACCESS_TOKEN`.

## Demo Commands

Databricks target:

```bash
dbt parse --project-dir dbt_demo
dbt build --project-dir dbt_demo --profiles-dir dbt_demo --target databricks
dbt docs generate --project-dir dbt_demo --profiles-dir dbt_demo --target databricks
```

On-prem target example only:

```bash
dbt build --project-dir dbt_demo --target onprem
```

If you do not have an on-prem SQL Server connected, leave this as a show-and-tell target during the demo. The shared models and target-aware macros are the important part.

## Production Databricks Execution Path

This repo now includes a separate Databricks Workflow job in the Databricks Asset Bundle:

- job key: `dot_dbt_demo`
- task type: native `dbt_task`
- project directory: `${workspace.file_path}/dbt_demo`
- profiles directory: `${workspace.file_path}/dbt_demo`

That Workflow runs:

```bash
dbt deps
dbt build --target databricks
```

After bundle deployment, you can run it with:

```bash
databricks bundle run dot_dbt_demo --target dev
```

Or trigger the `[dev] DOT dbt Demo` job in the Databricks Workflows UI.

## Demo Story

Use this folder to tell a simple architecture story:

1. Bronze/raw ingestion still happens outside dbt.
2. dbt owns reusable transformation logic for Silver and Gold.
3. The same models can be run against Databricks and an on-prem warehouse.
4. Adapter-specific differences stay in macros instead of duplicating models.

## Notes

- Databricks is the live/demo-capable target in this repo.
- SQL Server is scaffolded intentionally and remains disabled in the committed profile template.
- Replace the placeholder SQL warehouse HTTP path and token inputs before first production run.
- Existing project code remains unchanged by this folder.
