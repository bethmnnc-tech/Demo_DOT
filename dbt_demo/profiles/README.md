# Databricks Profile Setup

This folder documents the recommended profile setup for the `dot_dbt_demo` project.

## Local Development

1. Use the committed [../profiles.yml](/C:/Users/Frank.Matteson/Documents/GitHub/Demo_DOT/dbt_demo/profiles.yml) directly with `--profiles-dir dbt_demo`, or copy [../profiles.example.yml](/C:/Users/Frank.Matteson/Documents/GitHub/Demo_DOT/dbt_demo/profiles.example.yml) into your local dbt profiles directory as `profiles.yml`.
2. Set:
   - `DBT_DATABRICKS_HTTP_PATH`
   - `DATABRICKS_TOKEN`
3. Run:

```bash
dbt debug --project-dir dbt_demo --profiles-dir dbt_demo --target databricks
```

## Databricks Workflow Runs

The Databricks Workflow `dbt_task` injects `DBT_ACCESS_TOKEN` automatically for the Run As principal.

Because the profile template uses:

```yaml
token: "{{ env_var('DBT_ACCESS_TOKEN', env_var('DATABRICKS_TOKEN')) }}"
```

the same profile shape works for:

- local CLI runs using `DATABRICKS_TOKEN`
- Databricks Workflow runs using `DBT_ACCESS_TOKEN`
