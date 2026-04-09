import os
import mlflow
import pandas as pd
import streamlit as st

st.set_page_config(page_title="DOT ML Model Demo", layout="wide")
st.title("DOT Transportation ML Models")
st.caption("Interactive predictions from the DOT_Transportation_ML experiment")

# ── Load models from Unity Catalog ────────────────────────────────────────────
EXPERIMENT_ID = "947763301386294"

UC_MODELS = {
    "incident_severity_gbt": "models:/main.default.dot_incident_severity/4",
    "bridge_risk_score_gbt": "models:/main.default.dot_bridge_risk/4",
    "pavement_deterioration_rf": "models:/main.default.dot_pavement_deterioration/4",
}

@st.cache_resource
def load_models():
    mlflow.set_registry_uri("databricks-uc")
    models = {}
    for key, uri in UC_MODELS.items():
        models[key] = mlflow.sklearn.load_model(uri)
    return models

models = load_models()

# ── Metrics sidebar (optional — experiment access may be restricted) ──────────
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
tab1, tab2, tab3 = st.tabs([
    "Incident Severity", "Bridge Risk Score", "Pavement Deterioration"
])

# ── Tab 1: Incident Severity Classifier ──────────────────────────────────────
with tab1:
    st.header("Incident Severity Classifier")
    st.write("Predict the severity class of a traffic incident based on conditions.")

    if "incident_severity_gbt" not in models:
        st.error("Incident severity model not found in experiment.")
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

# ── Tab 2: Bridge Risk Score Regressor ───────────────────────────────────────
with tab2:
    st.header("Bridge Risk Score Regressor")
    st.write("Estimate a composite risk score for a bridge based on structural and traffic features.")

    if "bridge_risk_score_gbt" not in models:
        st.error("Bridge risk model not found in experiment.")
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

# ── Tab 3: Pavement Deterioration Classifier ─────────────────────────────────
with tab3:
    st.header("Pavement Deterioration Classifier")
    st.write("Predict whether a pavement segment will degrade to Poor/Very Poor condition.")

    if "pavement_deterioration_rf" not in models:
        st.error("Pavement deterioration model not found in experiment.")
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
