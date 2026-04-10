select *
from {{ ref('silver_traffic_incidents') }}
where incident_datetime > current_timestamp

