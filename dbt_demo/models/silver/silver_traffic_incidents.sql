{{ config(
    materialized='table',
    file_format='delta' if target.type == 'databricks' else none
) }}

with source_data as (
    select *
    from {{ source('dot_raw', 'traffic_incidents') }}
),

cleaned as (
    select
        incident_id,
        {{ dot_trim_upper('incident_type') }} as incident_type,
        {{ dot_trim('severity') }} as severity,
        incident_datetime,
        {{ dot_trim_upper('state_code') }} as state_code,
        {{ dot_trim_upper('county_code') }} as county_code,
        {{ dot_trim_upper('route_id') }} as route_id,
        milepost,
        latitude,
        longitude,
        {{ dot_trim('road_condition') }} as road_condition,
        {{ dot_trim('weather_condition') }} as weather_condition,
        fatalities,
        injuries,
        vehicles_involved,
        {{ dot_trim('status') }} as status,
        ingestion_timestamp
    from source_data
    where incident_id is not null
      and incident_datetime is not null
),

enriched as (
    select
        incident_id,
        incident_type,
        severity,
        incident_datetime,
        state_code,
        county_code,
        route_id,
        milepost,
        latitude,
        longitude,
        road_condition,
        weather_condition,
        fatalities,
        injuries,
        vehicles_involved,
        status,
        ingestion_timestamp,
        {{ dot_extract('year', 'incident_datetime') }} as incident_year,
        {{ dot_extract('month', 'incident_datetime') }} as incident_month,
        {{ dot_extract('hour', 'incident_datetime') }} as incident_hour,
        {{ dot_day_of_week('incident_datetime') }} as day_of_week,
        case
            when {{ dot_day_of_week('incident_datetime') }} in (1, 7) then true
            else false
        end as is_weekend,
        case
            when fatalities > 0 then true
            else false
        end as has_fatality,
        coalesce(fatalities, 0) + coalesce(injuries, 0) as total_casualties,
        case
            when severity = 'Fatal' then 5
            when severity = 'Serious Injury' then 4
            when severity = 'Minor Injury' then 3
            when severity = 'Property Damage Only' then 2
            else 1
        end as severity_score,
        {{ dot_current_timestamp() }} as silver_loaded_at
    from cleaned
)

select *
from enriched

