import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import BLUE, GREEN, GREY, ORANGE, clean_data, meta, predictor
from seer_survival import config as C
from seer_survival.data import derive_stage_6th
from seer_survival.features import INPUT_COLS
from seer_survival.inference import ALLOWED, from_raw_format

pred = predictor()
meta = meta()
get_data = clean_data

st.title("🧑‍⚕️ Patient prediction")
st.caption("Uses the saved models; the same feature pipeline runs inside each model file.")
with st.form("patient"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Demographics**")
        age = st.slider("Age at diagnosis (years)", 18, 90, 55,
                        help="Training data covers 30–69; outside that range the prediction is an extrapolation.")
        race = st.selectbox("Race", ALLOWED["race"])
        marital = st.selectbox("Marital status", ALLOWED["marital_status"])
    with c2:
        st.markdown("**Tumour**")
        t_stage = st.selectbox("T stage (AJCC 6th)", ALLOWED["t_stage"], index=1)
        tumor_size = st.number_input("Tumour size (mm)", 1, 200, 30)
        grade = st.selectbox("Grade", ALLOWED["grade"], index=1,
                             format_func=lambda g: {1: "1 – well diff.", 2: "2 – moderately diff.",
                                                    3: "3 – poorly diff.", 4: "4 – undifferentiated"}[g])
        a_stage = st.selectbox("SEER summary stage", ALLOWED["a_stage"])
    with c3:
        st.markdown("**Nodes & receptors**")
        n_stage = st.selectbox("N stage (AJCC 6th)", ALLOWED["n_stage"])
        nodes_examined = st.number_input("Regional nodes examined", 1, 100, 14)
        nodes_positive = st.number_input("Regional nodes positive", 1, 100, 2)
        er = st.radio("Oestrogen receptor", ALLOWED["estrogen_status"], horizontal=True)
        pr = st.radio("Progesterone receptor", ALLOWED["progesterone_status"], horizontal=True)
    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

patient = pd.DataFrame([{
    "age": age, "race": race, "marital_status": marital, "t_stage": t_stage, "n_stage": n_stage,
    "grade": grade, "a_stage": a_stage, "tumor_size": tumor_size, "estrogen_status": er,
    "progesterone_status": pr, "nodes_examined": nodes_examined, "nodes_positive": nodes_positive,
}])

try:
    res = pred.predict(patient)
    input_error = None
except ValueError as e:
    res, input_error = None, str(e)

if input_error:
    st.error(f"Input problem: {input_error}")
else:
    for w in res.attrs.get("warnings", []):
        st.warning(w, icon="⚠️")
    stage = derive_stage_6th(t_stage, n_stage)
    st.caption(f"Derived AJCC 6th-edition stage group: **{stage}**  ·  node ratio = "
               f"{nodes_positive / nodes_examined:.2f}")

    r = res.iloc[0]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("5-year mortality risk", f"{r.risk_5y:.1%}")
    k2.metric("Risk group (cohort tertile)", r.risk_group,
              help=f"Tertiles of training-set risk: < {meta['risk_group_cutoffs'][0]:.1%} low, "
                   f"< {meta['risk_group_cutoffs'][1]:.1%} intermediate, otherwise high. Relative, not clinical, thresholds.")
    k3.metric("Predicted survival at 5 years", f"{r.survival_60m:.1%}")
    k4.metric("Predicted survival at 7 years", f"{r.survival_84m:.1%}")

    g1, g2 = st.columns([1, 1.4])
    with g1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=r.risk_5y * 100, number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "5-year all-cause mortality risk"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": ORANGE},
                   "steps": [{"range": [0, meta["risk_group_cutoffs"][0] * 100], "color": "#d1fae5"},
                             {"range": [meta["risk_group_cutoffs"][0] * 100, meta["risk_group_cutoffs"][1] * 100], "color": "#fef3c7"},
                             {"range": [meta["risk_group_cutoffs"][1] * 100, 100], "color": "#fee2e2"}],
                   "threshold": {"line": {"color": "black", "width": 3}, "value": meta["threshold"] * 100}},
        ))
        gauge.update_layout(height=300, margin=dict(t=60, b=10, l=30, r=30))
        st.plotly_chart(gauge, use_container_width=True)
        st.caption(f"Black line = decision threshold {meta['threshold']:.1%} (Youden index on training data).")
    with g2:
        curve = pred.survival_curve(patient)
        df_all = get_data()
        from sksurv.nonparametric import kaplan_meier_estimator
        t_km, s_km = kaplan_meier_estimator(df_all.event.astype(bool), df_all.survival_months)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t_km, y=s_km, mode="lines", line=dict(shape="hv", color=GREY, dash="dot"),
                                 name="Whole cohort (Kaplan–Meier)"))
        fig.add_trace(go.Scatter(x=curve.month, y=curve.survival, mode="lines", line=dict(shape="hv", color=BLUE, width=3),
                                 name="This patient"))
        fig.add_vline(x=60, line_dash="dash", line_color=GREY)
        fig.update_layout(title="Predicted overall survival", xaxis_title="Months since diagnosis",
                          yaxis_title="Survival probability", yaxis_range=[max(0.0, min(curve.survival.min(), s_km.min()) - 0.05), 1.01], height=320,
                          margin=dict(t=50, b=10), legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Why this prediction? (feature occlusion)", expanded=True):
        ex = pred.explain(patient)
        ex = ex[ex.contribution.abs() > 1e-4].iloc[::-1]
        fig = go.Figure(go.Bar(
            x=ex.contribution * 100, y=[f"{f} = {v}" for f, v in zip(ex.feature, ex.value)], orientation="h",
            marker_color=[ORANGE if c > 0 else GREEN for c in ex.contribution],
            hovertemplate="%{y}<br>%{x:+.2f} percentage points<extra></extra>"))
        fig.update_layout(xaxis_title="change in 5-year risk vs a typical patient (percentage points)",
                          height=max(250, 32 * len(ex)), margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Each input is swapped for the training-set typical value (median / most common category) and the "
                   "change in predicted risk is shown. Orange raises risk, green lowers it. Interactions are ignored, "
                   "so treat this as indicative. It describes the model, not causes of death.")

