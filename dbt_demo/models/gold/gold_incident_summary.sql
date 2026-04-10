{{ config(
    materialized='table',
    file_format='delta' if target.type == 'databricks' else none
) }}

select
    state_code,
    route_id,
    incident_year,
    incident_month,
    incident_type,
    severity,
    count(incident_id) as total_incidents,
    sum(fatalities) as total_fatalities,
    sum(injuries) as total_injuries,
    sum(total_casualties) as total_casualties,
    avg(vehicles_involved) as avg_vehicles_involved,
    count(distinct county_code) as counties_affected,
    sum(case when has_fatality then 1 else 0 end) as fatal_incidents,
    sum(case when is_weekend then 1 else 0 end) as weekend_incidents,
    sum(case when incident_hour between 6 and 9 then 1 else 0 end) as morning_rush_incidents,
    sum(case when incident_hour between 15 and 19 then 1 else 0 end) as evening_rush_incidents,
    sum(case when incident_hour between 22 and 23 or incident_hour between 0 and 5 then 1 else 0 end) as overnight_incidents,
    {{ dot_safe_divide('sum(fatalities)', 'count(incident_id)') }} as fatality_rate,
    {{ dot_safe_divide('sum(injuries)', 'count(incident_id)') }} as injury_rate,
    {{ dot_current_timestamp() }} as gold_loaded_at
from {{ ref('silver_traffic_incidents') }}
group by
    state_code,
    route_id,
    incident_year,
    incident_month,
    incident_type,
    severity

