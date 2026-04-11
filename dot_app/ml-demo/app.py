import os
import mlflow
mlflow.set_tracking_uri("databricks")
import pandas as pd
import streamlit as st
from databricks.sdk import WorkspaceClient

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="DOT ML Model Demo", layout="wide", page_icon="🚦")
st.title("DOT Transportation ML Models")
st.caption("Interactive predictions & data assistant from the DOT Lakehouse")

# ── Constants ────────────────────────────────────────────────────────────────
EXPERIMENT_ID = "947763301386294"
WAREHOUSE_ID = os.environ.get("DATABRICKS_WAREHOUSE_ID", "269f3f8ac232d6f2")
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

UC_MODELS = {
    "incident_severity_gbt": "models:/main.default.dot_incident_severity/4",
    "bridge_risk_score_gbt": "models:/main.default.dot_bridge_risk/4",
    "pavement_deterioration_rf": "models:/main.default.dot_pavement_deterioration/4",
}

MODEL_DISPLAY_NAMES = {
    "incident_severity_gbt": "Incident Severity",
    "bridge_risk_score_gbt": "Bridge Risk Score",
    "pavement_deterioration_rf": "Pavement Deterioration",
}

# ── SQL schema context for the chatbot ───────────────────────────────────────
TABLE_SCHEMAS = """
Available tables (Databricks Unity Catalog):

-- GOLD LAYER --
main.dot_gold.high_risk_corridors (47,667 rows)
  state_code STRING, route_id STRING, county_code STRING, incident_count LONG,
  total_fatalities LONG, total_injuries LONG, avg_severity_score DOUBLE,
  fatality_concentration DOUBLE, state_rank INT, risk_tier STRING

main.dot_gold.executive_kpi_scorecard (10 rows)
  state_code STRING, total_incidents LONG, total_fatalities LONG, total_injuries LONG,
  total_bridges LONG, avg_bridge_sufficiency DOUBLE, deficient_bridges LONG,
  total_lane_miles DOUBLE, avg_pcr DOUBLE, poor_pavement_miles DOUBLE,
  total_registered_vehicles LONG, electric_vehicles LONG

main.dot_gold.pavement_needs_assessment (18,876 rows)
  state_code STRING, county_code STRING, functional_class STRING, pavement_type STRING,
  maintenance_priority STRING, iri_category STRING, segment_count LONG,
  total_lane_miles DOUBLE, avg_pcr DOUBLE, avg_iri DOUBLE, avg_distress_index DOUBLE,
  avg_pavement_age DOUBLE, estimated_cost_usd DOUBLE

-- GEOSPATIAL LAYER --
main.dot_geo.incidents_geo_enriched (50,000 rows)
  incident_id STRING, incident_type STRING, severity STRING, incident_datetime TIMESTAMP,
  state_code STRING, county_code STRING, route_id STRING, latitude DOUBLE, longitude DOUBLE,
  road_condition STRING, weather_condition STRING, vehicles_involved INT, fatalities INT,
  injuries INT, has_fatality BOOLEAN, severity_score INT, area_type STRING

main.dot_geo.incident_hotspots (49,655 rows)
  h3_index_r8 STRING, incident_count LONG, total_fatalities LONG, avg_severity_score DOUBLE,
  is_hotspot BOOLEAN, hotspot_class STRING, z_score DOUBLE, risk_tier STRING

main.dot_geo.corridor_safety_rates (20 rows)
  seg_route_id STRING, functional_class STRING, area_type STRING,
  incidents_on_corridor LONG, fatalities_on_corridor LONG, avg_severity DOUBLE,
  reference_aadt DOUBLE, incidents_per_100m_vmt DOUBLE

main.dot_geo.bridges_with_taz (25 rows)
  bridge_id STRING, bridge_type STRING, risk_score DOUBLE, sufficiency_rating DOUBLE,
  avg_daily_traffic INT, latitude DOUBLE, longitude DOUBLE

-- SILVER LAYER --
main.dot_silver.traffic_incidents (50,000 rows)
  incident_id STRING, incident_type STRING, severity STRING, incident_datetime TIMESTAMP,
  state_code STRING, county_code STRING, route_id STRING, milepost DOUBLE,
  latitude DOUBLE, longitude DOUBLE, road_condition STRING, weather_condition STRING,
  vehicles_involved INT, fatalities INT, injuries INT, severity_score INT

main.dot_silver.bridge_inspections (10,000 rows)
  bridge_id STRING, bridge_name STRING, state_code STRING, county_code STRING,
  route_id STRING, latitude DOUBLE, longitude DOUBLE, bridge_type STRING,
  material_type STRING, year_built INT, span_length_ft DOUBLE, deck_width_ft DOUBLE,
  avg_daily_traffic INT, sufficiency_rating DOUBLE, days_since_inspection INT

main.dot_silver.pavement_conditions (18,876 rows)
  segment_id STRING, state_code STRING, county_code STRING, route_id STRING,
  functional_class STRING, pavement_type STRING, pavement_age INT,
  iri INT, psi DOUBLE, pcr DOUBLE, cracking_percent DOUBLE, rutting_in DOUBLE,
  condition_rating STRING, aadt INT, truck_percent INT, maintenance_priority STRING

main.dot_silver.vehicle_registrations (50,000 rows)
  vin STRING, state_code STRING, county_code STRING, vehicle_class STRING,
  make STRING, model_year INT, fuel_type STRING, is_electric BOOLEAN,
  is_commercial BOOLEAN, registration_status STRING
"""

SYSTEM_PROMPT = f"""You are a DOT Transportation Data Assistant. You help users explore
North Carolina Department of Transportation data stored in a Databricks Lakehouse.

When the user asks a data question:
1. Write a single Databricks SQL query to answer it
2. Wrap the query in ```sql ... ``` code fences
3. Keep queries concise — use LIMIT 20 unless the user asks for more
4. After the query, briefly explain what the query does

When the user asks a general question (not needing data), answer directly.

IMPORTANT RULES:
- Always use fully qualified table names (main.dot_gold.xxx, main.dot_geo.xxx, etc.)
- Use Databricks SQL syntax
- Never use DELETE, UPDATE, INSERT, DROP, ALTER, CREATE, or any DDL/DML — only SELECT
- If unsure which table to use, suggest options and ask

{TABLE_SCHEMAS}"""


# ── Helpers ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    mlflow.set_registry_uri("databricks-uc")
    models = {}
    for key, uri in UC_MODELS.items():
        try:
            models[key] = mlflow.sklearn.load_model(uri)
        except Exception:
            pass
    return models


@st.cache_resource
def get_workspace_client():
    return WorkspaceClient()


def run_sql(query: str) -> pd.DataFrame:
    """Execute SQL via Statement Execution API (OAuth-native, no connector needed)."""
    w = get_workspace_client()
    from databricks.sdk.service.sql import Disposition, Format
    resp = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID,
        statement=query,
        wait_timeout="50s",
        disposition=Disposition.INLINE,
        format=Format.JSON_ARRAY,
    )
    if resp.status and resp.status.error:
        raise Exception(resp.status.error.message)
    if not resp.manifest or not resp.result:
        return pd.DataFrame()
    cols = [c.name for c in resp.manifest.schema.columns]
    rows = resp.result.data_array or []
    return pd.DataFrame(rows, columns=cols)


def chat_llm(messages: list) -> str:
    """Call the foundation model endpoint via REST (OAuth from SDK)."""
    import requests as _req
    w = get_workspace_client()
    auth_headers = w.config.authenticate()
    resp = _req.post(
        f"{w.config.host}/serving-endpoints/{LLM_ENDPOINT}/invocations",
        headers={**auth_headers, "Content-Type": "application/json"},
        json={"messages": messages, "max_tokens": 1024, "temperature": 0.1},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def extract_sql(text: str):
    """Extract SQL from ```sql ... ``` fences."""
    if "```sql" in text:
        start = text.index("```sql") + 6
        end = text.index("```", start)
        return text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        return text[start:end].strip()
    return None


@st.cache_data(ttl=300)
def load_experiment_runs():
    """Fetch all MLflow runs for the DOT experiment, cached for 5 minutes."""
    try:
        runs = mlflow.search_runs(
            experiment_ids=[EXPERIMENT_ID],
            order_by=["start_time DESC"],
        )
        return runs
    except Exception:
        return pd.DataFrame()


# ── Load models ──────────────────────────────────────────────────────────────
models = load_models()

# ── Metrics sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model Performance")
    try:
        runs = mlflow.search_runs(experiment_ids=[EXPERIMENT_ID])
        for _, row in runs.iterrows():
            name = row.get("tags.mlflow.runName", "unknown")
            st.subheader(name)
            for col in row.index:
                if col.startswith("metrics.") and pd.notna(row[col]):
                    metric_name = col.replace("metrics.", "")
                    st.metric(metric_name, f"{row[col]:.4f}")
            st.divider()
    except Exception:
        st.info("Experiment metrics unavailable. Models loaded from Unity Catalog.")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🤖 Data Assistant",
    "⚡ Incident Severity",
    "🌉 Bridge Risk Score",
    "🛣️ Pavement Deterioration",
    "📊 Experiment Tracking",
])

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1: Data Assistant (Chatbot)
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    pass  # placeholder — defined further below to keep original tabs intact

with tab1:
    col_h, col_b = st.columns([5, 1])
    with col_h:
        st.header("DOT Data Assistant")
        st.write("Ask questions about NC transportation data — I'll query the Lakehouse for you.")
    with col_b:
        st.write("")  # spacer
        if st.button("🗑️ Clear chat", key="clear_chat"):
            st.session_state.chat_messages = []
            st.rerun()

    # Example questions
    with st.expander("💡 Example questions", expanded=False):
        examples = [
            "What are the top 10 highest-risk corridors by fatalities?",
            "How many incidents happened in rainy weather vs clear weather?",
            "Show me bridge inspection stats by bridge type",
            "Which counties have the worst pavement conditions?",
            "What's the executive KPI scorecard for NC?",
            "How many electric vehicles are registered?",
            "Show incident hotspots with z-score above 3",
            "Compare DUI vs collision incident counts by county",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{hash(ex)}"):
                st.session_state["chat_input_pending"] = ex

    # Chat state
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Show only the latest exchange (clean single-turn view)
    if st.session_state.chat_messages:
        # Find last user + assistant pair
        msgs = st.session_state.chat_messages
        last_user = None
        last_asst = None
        for m in reversed(msgs):
            if m["role"] == "assistant" and last_asst is None:
                last_asst = m
            elif m["role"] == "user" and last_user is None:
                last_user = m
            if last_user and last_asst:
                break
        if last_user:
            with st.chat_message("user"):
                st.markdown(last_user["content"])
        if last_asst:
            with st.chat_message("assistant"):
                st.markdown(last_asst["content"])
                if "dataframe" in last_asst:
                    st.dataframe(last_asst["dataframe"], use_container_width=True)
                if "sql" in last_asst:
                    with st.expander("SQL Query", expanded=False):
                        st.code(last_asst["sql"], language="sql")

    # Pending input from example button
    pending = st.session_state.pop("chat_input_pending", None)
    user_input = st.chat_input("Ask about DOT transportation data...")
    prompt = pending or user_input

    if prompt:
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Build messages for LLM
        llm_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        # Include last 6 messages for context
        for m in st.session_state.chat_messages[-6:]:
            llm_messages.append({"role": m["role"], "content": m["content"]})

        # Get LLM response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    reply = chat_llm(llm_messages)
                except Exception as e:
                    reply = f"Error calling model: {e}"

            # Check for SQL in response
            sql_query = extract_sql(reply)
            msg_data = {"role": "assistant", "content": reply}

            if sql_query:
                # Validate — only SELECT allowed
                sql_upper = sql_query.strip().upper()
                if not sql_upper.startswith("SELECT") and not sql_upper.startswith("WITH"):
                    st.warning("Only SELECT queries are allowed.")
                    st.markdown(reply)
                else:
                    st.markdown(reply)
                    msg_data["sql"] = sql_query
                    with st.expander("SQL Query", expanded=False):
                        st.code(sql_query, language="sql")
                    with st.spinner("Running query..."):
                        try:
                            df_result = run_sql(sql_query)
                            if len(df_result) == 0:
                                st.info("Query returned no results.")
                            else:
                                st.dataframe(df_result, use_container_width=True)
                                msg_data["dataframe"] = df_result

                                # Auto-chart for numeric columns
                                num_cols = df_result.select_dtypes(include="number").columns.tolist()
                                str_cols = df_result.select_dtypes(include="object").columns.tolist()
                                if len(num_cols) >= 1 and len(str_cols) >= 1 and len(df_result) <= 30:
                                    st.bar_chart(df_result.set_index(str_cols[0])[num_cols[:3]])
                        except Exception as e:
                            st.error(f"Query error: {e}")
            else:
                st.markdown(reply)

            st.session_state.chat_messages.append(msg_data)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2: Incident Severity Classifier
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Incident Severity Classifier")
    st.write("Predict the severity class of a traffic incident based on conditions.")

    if "incident_severity_gbt" not in models:
        st.error("Incident severity model not found.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            inc_type = st.selectbox("Incident Type",
                ["COLLISION", "DUI", "PEDESTRIAN", "ROLLOVER", "FIXED OBJECT", "SIDESWIPE"])
            weather = st.selectbox("Weather Condition",
                ["Clear", "Rain", "Snow", "Fog", "Sleet", "Wind"])
        with col2:
            road = st.selectbox("Road Condition",
                ["Dry", "Wet", "Icy", "Snow-Covered", "Muddy"])
            state = st.selectbox("State", ["NC", "VA", "SC", "GA", "TN"], index=0)
        with col3:
            hour = st.slider("Hour of Day", 0, 23, 17)
            dow = st.slider("Day of Week (1=Sun)", 1, 7, 3)
            vehicles = st.slider("Vehicles Involved", 1, 10, 2)

        if st.button("Predict Severity", type="primary", key="sev"):
            sample = pd.DataFrame([{
                "incident_type": inc_type,
                "state_code": state,
                "road_condition": road,
                "weather_condition": weather,
                "incident_hour": hour,
                "day_of_week": dow,
                "vehicles_involved": vehicles,
                "severity_score": 3,
                "is_weekend_int": 1 if dow in [1, 7] else 0,
            }])
            pred = models["incident_severity_gbt"].predict(sample)
            st.success(f"Predicted severity class: **{pred[0]}**")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 3: Bridge Risk Score Regressor
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Bridge Risk Score Regressor")
    st.write("Estimate a composite risk score for a bridge based on structural and traffic features.")

    if "bridge_risk_score_gbt" not in models:
        st.error("Bridge risk model not found.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            bridge_type = st.selectbox("Bridge Type",
                ["Steel", "Concrete", "Prestressed Concrete", "Timber", "Masonry"])
            material = st.selectbox("Material Type",
                ["Steel", "Concrete", "Wood", "Iron", "Aluminum"])
            owner = st.selectbox("Owner Type",
                ["State", "County", "City", "Federal", "Railroad", "Private"])
        with col2:
            age = st.slider("Bridge Age (years)", 1, 150, 55)
            sufficiency = st.slider("Sufficiency Rating (0-100)", 0, 100, 42)
            adt = st.number_input("Avg Daily Traffic", 100, 300000, 25000, step=1000)
        with col3:
            span = st.slider("Span Length (ft)", 10, 1000, 120)
            deck = st.slider("Deck Width (ft)", 10, 120, 36)
            days_insp = st.slider("Days Since Last Inspection", 0, 2000, 800)

        if st.button("Predict Risk Score", type="primary", key="brg"):
            sample = pd.DataFrame([{
                "bridge_type": bridge_type,
                "material_type": material,
                "owner_type": owner,
                "state_code": "NC",
                "bridge_age_years": age,
                "span_length_ft": float(span),
                "deck_width_ft": float(deck),
                "avg_daily_traffic": adt,
                "sufficiency_rating": float(sufficiency),
                "days_since_inspection": days_insp,
            }])
            score = models["bridge_risk_score_gbt"].predict(sample)
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Predicted Risk Score", f"{score[0]:.1f}")
            with col_b:
                if score[0] >= 70:
                    st.error("Priority 1 - Immediate attention required")
                elif score[0] >= 45:
                    st.warning("Priority 2 - Near term action needed")
                elif score[0] >= 20:
                    st.info("Priority 3 - Planned maintenance")
                else:
                    st.success("Priority 4 - Monitor only")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 4: Pavement Deterioration Classifier
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Pavement Deterioration Classifier")
    st.write("Predict whether a pavement segment will degrade to Poor/Very Poor condition.")

    if "pavement_deterioration_rf" not in models:
        st.error("Pavement deterioration model not found.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            pav_type = st.selectbox("Pavement Type",
                ["Asphalt", "Concrete", "Composite", "Gravel"])
            func_class = st.selectbox("Functional Class",
                ["Interstate", "US Route", "State Route", "Local Road", "Collector"])
            pav_age = st.slider("Pavement Age (years)", 0, 60, 25)
        with col2:
            iri = st.slider("IRI (roughness)", 0, 400, 120)
            psi = st.slider("PSI (serviceability)", 0.0, 5.0, 2.5, 0.1)
            pcr = st.slider("PCR (condition rating)", 0, 100, 60)
        with col3:
            cracking = st.slider("Cracking %", 0.0, 100.0, 15.0, 0.5)
            rutting = st.slider("Rutting (inches)", 0.0, 2.0, 0.3, 0.05)
            aadt = st.number_input("AADT", 100, 200000, 15000, step=1000, key="pav_aadt")
            truck_pct = st.slider("Truck %", 0, 50, 10)

        distress_idx = cracking * 0.4 + rutting * 5 + (100 - pcr) * 0.6

        if st.button("Predict Deterioration Risk", type="primary", key="pav"):
            sample = pd.DataFrame([{
                "pavement_type": pav_type,
                "functional_class": func_class,
                "state_code": "NC",
                "pavement_age": pav_age,
                "iri": float(iri),
                "psi": float(psi),
                "pcr": float(pcr),
                "cracking_percent": cracking,
                "rutting_in": rutting,
                "aadt": aadt,
                "truck_percent": truck_pct,
                "distress_index": distress_idx,
            }])
            pred = models["pavement_deterioration_rf"].predict(sample)
            proba = models["pavement_deterioration_rf"].predict_proba(sample)[0]

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Deterioration Probability", f"{proba[1]:.1%}")
            with col_b:
                if pred[0] == 1.0:
                    st.error("HIGH RISK - Predicted to degrade to Poor/Very Poor")
                else:
                    st.success("Low risk - Condition expected to hold")

            st.progress(min(proba[1], 1.0), text=f"Risk: {proba[1]:.1%}")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 5: MLflow Experiment Tracking
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("MLflow Experiment Tracking")
    st.write("Compare model training runs, metrics, and parameters from the DOT Transportation ML experiment.")

    all_runs = load_experiment_runs()

    if all_runs.empty:
        st.warning("No experiment runs found. Run the ML training pipeline to generate data.")
    else:
        # ── Latest run per model ─────────────────────────────────────────────
        st.subheader("Latest Run per Model")

        # Get the most recent run for each model type
        all_runs["run_name"] = all_runs.get("tags.mlflow.runName", "unknown")
        latest_runs = all_runs.drop_duplicates(subset=["run_name"], keep="first")

        # Build comparison table
        comparison_rows = []
        for _, run in latest_runs.iterrows():
            name = run["run_name"]
            row = {"Model": name, "Status": run.get("status", ""), "Duration (s)": ""}

            # Calculate duration
            if pd.notna(run.get("start_time")) and pd.notna(run.get("end_time")):
                duration = (run["end_time"] - run["start_time"]).total_seconds()
                row["Duration (s)"] = f"{duration:.0f}"

            # Add all metrics
            for col in run.index:
                if col.startswith("metrics.") and pd.notna(run[col]):
                    metric_name = col.replace("metrics.", "")
                    row[metric_name] = round(run[col], 4)

            # Add key params
            for col in run.index:
                if col.startswith("params.") and pd.notna(run[col]):
                    param_name = col.replace("params.", "")
                    row[param_name] = run[col]

            comparison_rows.append(row)

        if comparison_rows:
            df_compare = pd.DataFrame(comparison_rows)
            st.dataframe(df_compare, use_container_width=True, hide_index=True)

        # ── Metric comparison chart ──────────────────────────────────────────
        st.subheader("Metric Comparison")

        # Collect all metric columns
        metric_cols = [c for c in all_runs.columns if c.startswith("metrics.")]
        metric_names = [c.replace("metrics.", "") for c in metric_cols]

        if metric_names:
            selected_metric = st.selectbox(
                "Select metric to compare",
                metric_names,
                key="exp_metric_select"
            )
            metric_col = f"metrics.{selected_metric}"

            # Filter runs that have this metric
            runs_with_metric = all_runs[all_runs[metric_col].notna()].copy()

            if not runs_with_metric.empty:
                chart_data = runs_with_metric[["run_name", metric_col]].copy()
                chart_data.columns = ["Model", selected_metric]
                chart_data = chart_data.set_index("Model")
                st.bar_chart(chart_data)

        # ── Metric trends over training runs ─────────────────────────────────
        st.subheader("Metric Trends Over Training Runs")
        st.write("Track how each model's metrics evolve across successive training runs.")

        # Group runs by model name and plot metrics over time
        model_names = sorted(all_runs["run_name"].unique())

        # Let user pick which model to view trends for
        trend_model = st.selectbox(
            "Select model",
            model_names,
            key="trend_model_select",
        )

        # Filter to that model's runs, sorted chronologically
        model_runs = all_runs[all_runs["run_name"] == trend_model].copy()
        model_runs = model_runs.sort_values("start_time", ascending=True).reset_index(drop=True)

        if len(model_runs) < 2:
            st.info(f"Only {len(model_runs)} run found for **{trend_model}**. Trend charts require 2+ runs.")
        else:
            # Find metrics available for this model
            model_metric_cols = [
                c for c in model_runs.columns
                if c.startswith("metrics.") and model_runs[c].notna().any()
            ]

            if model_metric_cols:
                # Build trend dataframe: run number as index, one column per metric
                trend_df = pd.DataFrame()
                trend_df["Run"] = [
                    f"#{i+1} ({row['start_time'].strftime('%m/%d %H:%M')})"
                    if pd.notna(row.get("start_time")) else f"#{i+1}"
                    for i, (_, row) in enumerate(model_runs.iterrows())
                ]

                for mc in model_metric_cols:
                    clean_name = mc.replace("metrics.", "")
                    trend_df[clean_name] = model_runs[mc].values

                trend_df = trend_df.set_index("Run")

                # Let user pick which metrics to plot
                available_metrics = list(trend_df.columns)
                selected_trends = st.multiselect(
                    "Select metrics to plot",
                    available_metrics,
                    default=available_metrics[:3],
                    key="trend_metric_multi",
                )

                if selected_trends:
                    st.line_chart(trend_df[selected_trends])

                    # Show delta between first and last run
                    st.write("**Change from first to latest run:**")
                    delta_cols = st.columns(min(len(selected_trends), 4))
                    for i, metric in enumerate(selected_trends):
                        col_idx = i % len(delta_cols)
                        first_val = trend_df[metric].iloc[0]
                        last_val = trend_df[metric].iloc[-1]
                        if pd.notna(first_val) and pd.notna(last_val):
                            delta = last_val - first_val
                            delta_pct = (delta / first_val * 100) if first_val != 0 else 0
                            with delta_cols[col_idx]:
                                st.metric(
                                    metric,
                                    f"{last_val:.4f}",
                                    delta=f"{delta:+.4f} ({delta_pct:+.1f}%)",
                                )
            else:
                st.info(f"No metrics logged for **{trend_model}**.")

        # ── Run history timeline ─────────────────────────────────────────────
        st.subheader("Run History")

        # Show all runs sorted by time
        history_cols = ["run_name", "status", "start_time"]
        history_cols += [c for c in metric_cols if all_runs[c].notna().any()]
        param_cols = [c for c in all_runs.columns if c.startswith("params.") and all_runs[c].notna().any()]
        history_cols += param_cols

        available_cols = [c for c in history_cols if c in all_runs.columns]
        df_history = all_runs[available_cols].copy()

        # Clean column names for display
        df_history.columns = [
            c.replace("metrics.", "").replace("params.", "").replace("tags.mlflow.", "")
            for c in df_history.columns
        ]

        st.dataframe(df_history, use_container_width=True, hide_index=True)

        # ── Registered model versions ────────────────────────────────────────
        st.subheader("Registered Model Versions")

        uc_model_names = [
            "main.default.dot_incident_severity",
            "main.default.dot_bridge_risk",
            "main.default.dot_pavement_deterioration",
        ]

        try:
            w = get_workspace_client()
            for model_name in uc_model_names:
                with st.expander(f"📦 {model_name}", expanded=False):
                    try:
                        from mlflow import MlflowClient
                        client = MlflowClient(registry_uri="databricks-uc")
                        versions = client.search_model_versions(f"name='{model_name}'")
                        if versions:
                            version_rows = []
                            for v in versions:
                                version_rows.append({
                                    "Version": v.version,
                                    "Status": v.status,
                                    "Created": str(v.creation_timestamp),
                                    "Run ID": v.run_id or "",
                                })
                            st.dataframe(
                                pd.DataFrame(version_rows),
                                use_container_width=True,
                                hide_index=True,
                            )
                        else:
                            st.info("No versions found.")
                    except Exception as e:
                        st.info(f"Could not fetch versions: {e}")
        except Exception:
            st.info("Model version details unavailable.")

        # ── Refresh button ───────────────────────────────────────────────────
        if st.button("🔄 Refresh experiment data", key="refresh_exp"):
            st.cache_data.clear()
            st.rerun()
