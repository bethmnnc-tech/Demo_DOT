{{ config(
    materialized='table',
    file_format='delta' if target.type == 'databricks' else none
) }}

with source_data as (
    select *
    from {{ source('dot_raw', 'bridge_inspections') }}
),

cleaned as (
    select
        bridge_id,
        bridge_name,
        {{ dot_trim_upper('state_code') }} as state_code,
        {{ dot_trim_upper('county_code') }} as county_code,
        {{ dot_trim_upper('route_id') }} as route_id,
        milepost,
        latitude,
        longitude,
        {{ dot_trim('bridge_type') }} as bridge_type,
        {{ dot_trim('material_type') }} as material_type,
        {{ dot_trim('owner_type') }} as owner_type,
        year_built,
        span_length_ft,
        deck_width_ft,
        vertical_clearance_ft,
        avg_daily_traffic,
        last_inspection_date,
        deck_condition,
        superstructure_condition,
        substructure_condition,
        sufficiency_rating,
        {{ dot_bool_from_text('structurally_deficient') }} as structurally_deficient,
        {{ dot_bool_from_text('functionally_obsolete') }} as functionally_obsolete,
        estimated_repair_cost_k,
        ingestion_timestamp
    from source_data
    where bridge_id is not null
      and year_built between 1800 and 2025
),

enriched as (
    select
        bridge_id,
        bridge_name,
        state_code,
        county_code,
        route_id,
        milepost,
        latitude,
        longitude,
        bridge_type,
        material_type,
        owner_type,
        year_built,
        span_length_ft,
        deck_width_ft,
        vertical_clearance_ft,
        avg_daily_traffic,
        last_inspection_date,
        deck_condition,
        superstructure_condition,
        substructure_condition,
        sufficiency_rating,
        structurally_deficient,
        functionally_obsolete,
        estimated_repair_cost_k,
        ingestion_timestamp,
        2024 - year_built as bridge_age_years,
        case
            when 2024 - year_built < 20 then 'New (<20 yrs)'
            when 2024 - year_built < 40 then 'Mature (20-40 yrs)'
            when 2024 - year_built < 60 then 'Aging (40-60 yrs)'
            else 'Old (60+ yrs)'
        end as age_category,
        case
            when {{ dot_startswith('deck_condition', 'P') }}
              or {{ dot_startswith('superstructure_condition', 'P') }}
              or {{ dot_startswith('substructure_condition', 'P') }} then 'Poor'
            when {{ dot_startswith('deck_condition', 'F') }}
              or {{ dot_startswith('superstructure_condition', 'F') }}
              or {{ dot_startswith('substructure_condition', 'F') }} then 'Fair'
            else 'Good'
        end as worst_condition,
        round(
            ((100 - sufficiency_rating) * 0.5)
            + ((2024 - year_built) * 0.3)
            + (case when structurally_deficient then 20 else 0 end),
            1
        ) as risk_score,
        case
            when round(
                ((100 - sufficiency_rating) * 0.5)
                + ((2024 - year_built) * 0.3)
                + (case when structurally_deficient then 20 else 0 end),
                1
            ) >= 70 then 'Priority 1 - Immediate'
            when round(
                ((100 - sufficiency_rating) * 0.5)
                + ((2024 - year_built) * 0.3)
                + (case when structurally_deficient then 20 else 0 end),
                1
            ) >= 45 then 'Priority 2 - Near Term'
            when round(
                ((100 - sufficiency_rating) * 0.5)
                + ((2024 - year_built) * 0.3)
                + (case when structurally_deficient then 20 else 0 end),
                1
            ) >= 20 then 'Priority 3 - Planned'
            else 'Priority 4 - Monitor'
        end as priority_tier,
        {{ dot_days_since('last_inspection_date') }} as days_since_inspection,
        case
            when {{ dot_days_since('last_inspection_date') }} > 730 then true
            else false
        end as inspection_overdue,
        {{ dot_current_timestamp() }} as silver_loaded_at
    from cleaned
)

select *
from enriched

