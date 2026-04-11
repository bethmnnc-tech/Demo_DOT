import os
import mlflow
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
tab1, tab2, tab3, tab4 = st.tabs([
    "🤖 Data Assistant",
    "⚡ Incident Severity",
    "🌉 Bridge Risk Score",
    "🛣️ Pavement Deterioration",
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
