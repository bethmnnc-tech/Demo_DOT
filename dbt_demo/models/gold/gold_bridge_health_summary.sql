{{ config(
    materialized='table',
    file_format='delta' if target.type == 'databricks' else none
) }}

select
    state_code,
    county_code,
    bridge_type,
    age_category,
    priority_tier,
    count(bridge_id) as bridge_count,
    avg(sufficiency_rating) as avg_sufficiency_rating,
    min(sufficiency_rating) as min_sufficiency_rating,
    avg(bridge_age_years) as avg_bridge_age,
    avg(avg_daily_traffic) as avg_daily_traffic,
    sum(avg_daily_traffic) as total_daily_traffic_exposed,
    sum(case when structurally_deficient then 1 else 0 end) as structurally_deficient_count,
    sum(case when functionally_obsolete then 1 else 0 end) as functionally_obsolete_count,
    sum(case when inspection_overdue then 1 else 0 end) as overdue_inspections,
    sum(estimated_repair_cost_k) as total_estimated_repair_cost_k,
    avg(risk_score) as avg_risk_score,
    max(risk_score) as max_risk_score,
    {{ dot_safe_divide('sum(case when structurally_deficient then 1 else 0 end)', 'count(bridge_id)') }} as deficiency_rate,
    sum(estimated_repair_cost_k) / 1000.0 as total_estimated_repair_cost_m,
    {{ dot_current_timestamp() }} as gold_loaded_at
from {{ ref('silver_bridge_inspections') }}
group by
    state_code,
    county_code,
    bridge_type,
    age_category,
    priority_tier

