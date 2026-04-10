{{ config(
    materialized='table',
    file_format='delta' if target.type == 'databricks' else none
) }}

with corridor_stats as (
    select
        state_code,
        route_id,
        county_code,
        count(incident_id) as incident_count,
        sum(fatalities) as total_fatalities,
        sum(injuries) as total_injuries,
        count(distinct incident_type) as incident_type_diversity,
        avg(severity_score) as avg_severity_score,
        max(incident_datetime) as most_recent_incident
    from {{ ref('silver_traffic_incidents') }}
    group by
        state_code,
        route_id,
        county_code
),

ranked as (
    select
        *,
        {{ dot_safe_divide('total_fatalities', 'incident_count') }} as fatality_concentration,
        rank() over (
            partition by state_code
            order by total_fatalities desc, incident_count desc, route_id
        ) as state_rank
    from corridor_stats
)

select
    state_code,
    route_id,
    county_code,
    incident_count,
    total_fatalities,
    total_injuries,
    incident_type_diversity,
    avg_severity_score,
    most_recent_incident,
    fatality_concentration,
    state_rank,
    case
        when state_rank <= 10 then 'Top 10 - Critical'
        when state_rank <= 25 then 'Top 25 - High Risk'
        when state_rank <= 50 then 'Top 50 - Elevated'
        else 'Standard'
    end as risk_tier,
    {{ dot_current_timestamp() }} as gold_loaded_at
from ranked

