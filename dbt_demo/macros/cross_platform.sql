{% macro dot_string_type() -%}
  {{ return(adapter.dispatch('dot_string_type', 'dot_dbt_demo')()) }}
{%- endmacro %}

{% macro default__dot_string_type() -%}
varchar
{%- endmacro %}

{% macro databricks__dot_string_type() -%}
string
{%- endmacro %}

{% macro dot_trim_upper(expression) -%}
upper(trim(cast({{ expression }} as {{ dot_string_type() }})))
{%- endmacro %}

{% macro dot_trim(expression) -%}
trim(cast({{ expression }} as {{ dot_string_type() }}))
{%- endmacro %}

{% macro dot_bool_from_text(expression) -%}
  {{ return(adapter.dispatch('dot_bool_from_text', 'dot_dbt_demo')(expression)) }}
{%- endmacro %}

{% macro default__dot_bool_from_text(expression) -%}
case
  when lower(trim(cast({{ expression }} as {{ dot_string_type() }}))) in ('true', 't', '1', 'yes', 'y') then true
  else false
end
{%- endmacro %}

{% macro sqlserver__dot_bool_from_text(expression) -%}
case
  when lower(ltrim(rtrim(cast({{ expression }} as {{ dot_string_type() }})))) in ('true', 't', '1', 'yes', 'y') then cast(1 as bit)
  else cast(0 as bit)
end
{%- endmacro %}

{% macro dot_extract(part, expression) -%}
  {{ return(adapter.dispatch('dot_extract', 'dot_dbt_demo')(part, expression)) }}
{%- endmacro %}

{% macro default__dot_extract(part, expression) -%}
extract({{ part }} from {{ expression }})
{%- endmacro %}

{% macro sqlserver__dot_extract(part, expression) -%}
datepart({{ part }}, {{ expression }})
{%- endmacro %}

{% macro dot_day_of_week(expression) -%}
  {{ return(adapter.dispatch('dot_day_of_week', 'dot_dbt_demo')(expression)) }}
{%- endmacro %}

{% macro default__dot_day_of_week(expression) -%}
dayofweek({{ expression }})
{%- endmacro %}

{% macro sqlserver__dot_day_of_week(expression) -%}
datepart(weekday, {{ expression }})
{%- endmacro %}

{% macro dot_current_timestamp() -%}
current_timestamp
{%- endmacro %}

{% macro dot_date_from_timestamp(expression) -%}
cast({{ expression }} as date)
{%- endmacro %}

{% macro dot_days_since(expression) -%}
  {{ return(adapter.dispatch('dot_days_since', 'dot_dbt_demo')(expression)) }}
{%- endmacro %}

{% macro default__dot_days_since(expression) -%}
datediff(current_date, cast({{ expression }} as date))
{%- endmacro %}

{% macro sqlserver__dot_days_since(expression) -%}
datediff(day, cast({{ expression }} as date), cast(getdate() as date))
{%- endmacro %}

{% macro dot_startswith(expression, prefix) -%}
  {{ return(adapter.dispatch('dot_startswith', 'dot_dbt_demo')(expression, prefix)) }}
{%- endmacro %}

{% macro default__dot_startswith(expression, prefix) -%}
cast({{ expression }} as {{ dot_string_type() }}) like '{{ prefix }}%'
{%- endmacro %}

{% macro sqlserver__dot_startswith(expression, prefix) -%}
cast({{ expression }} as {{ dot_string_type() }}) like '{{ prefix }}%'
{%- endmacro %}

{% macro dot_safe_divide(numerator, denominator) -%}
case
  when {{ denominator }} = 0 or {{ denominator }} is null then null
  else {{ numerator }} * 1.0 / {{ denominator }}
end
{%- endmacro %}

