import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from common import BLUE, ORANGE, STATUS_COLORS, clean_data, km_figure, logrank, wilson_ci
from seer_survival import config as C

st.title("📈 Live EDA and survival analysis")
df = clean_data().copy()
df["node_ratio"] = df.nodes_positive / df.nodes_examined
df["age_group"] = pd.cut(df.age, [29, 39, 49, 59, 69], labels=["30-39", "40-49", "50-59", "60-69"]).astype(str)

cat_vars = ["stage_6th", "t_stage", "n_stage", "grade", "a_stage", "estrogen_status", "progesterone_status",
            "race", "marital_status", "age_group"]
num_vars = ["age", "tumor_size", "nodes_examined", "nodes_positive", "node_ratio"]

# ---------------------------------------------------------------- filters
with st.expander("🔎 Filter the cohort (all charts below update)", expanded=False):
    f1, f2, f3 = st.columns(3)
    age_rng = f1.slider("Age", 30, 69, (30, 69))
    stages = f2.multiselect("Stage group", C.ORDINAL_LEVELS["stage_6th"], default=C.ORDINAL_LEVELS["stage_6th"])
    er = f3.multiselect("ER status", ["Positive", "Negative"], default=["Positive", "Negative"])
df = df[df.age.between(*age_rng) & df.stage_6th.isin(stages) & df.estrogen_status.isin(er)]
st.caption(f"{len(df):,} patients selected · {df.event.sum()} deaths ({df.event.mean():.1%})")
if len(df) < 30:
    st.warning("Too few patients for reliable statistics.")
    st.stop()

tab_cat, tab_num, tab_corr = st.tabs(["Categorical variable", "Numeric variable", "Correlation"])

with tab_cat:
    v = st.selectbox("Variable", cat_vars)
    levels = C.ORDINAL_LEVELS.get(v) or sorted(df[v].unique())
    g = df.groupby(v)["event"].agg(["size", "sum"]).reindex([l for l in levels if l in df[v].unique()])
    lo, hi = wilson_ci(g["sum"], g["size"])
    g["rate"] = g["sum"] / g["size"]
    ct = pd.crosstab(df[v], df.status)
    if ct.shape[0] >= 2 and ct.shape[1] == 2:
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        cramer = np.sqrt(chi2 / (ct.to_numpy().sum() * (min(ct.shape) - 1)))
        lr_chi, lr_p = logrank(df, v)
        c1, c2, c3 = st.columns(3)
        c1.metric("χ² test p-value", f"{p:.1e}")
        c2.metric("Cramér's V (effect size)", f"{cramer:.3f}")
        c3.metric("Log-rank p-value", f"{lr_p:.1e}")
    else:
        st.info("Tests need at least two groups and both outcomes in the current selection.")
    a, b = st.columns(2)
    fig = go.Figure(go.Bar(x=g.index.astype(str), y=g.rate, marker_color=ORANGE,
                           error_y=dict(type="data", symmetric=False, array=hi - g.rate, arrayminus=g.rate - lo),
                           text=[f"n={n}" for n in g["size"]], textposition="outside"))
    fig.add_hline(y=df.event.mean(), line_dash="dot")
    fig.update_layout(title="Crude death rate (95% Wilson CI)", yaxis_tickformat=".0%", height=420)
    a.plotly_chart(fig, use_container_width=True)
    b.plotly_chart(km_figure(df, v, levels=[l for l in levels if l in df[v].unique()]).update_layout(
        title="Kaplan–Meier survival"), use_container_width=True)
    st.caption("Crude rates mix prognosis with length of follow-up; the Kaplan–Meier curve and log-rank test handle "
               "censoring correctly. Associations are not causal.")

with tab_num:
    v = st.selectbox("Variable ", num_vars)
    dead, alive = df.loc[df.event == 1, v], df.loc[df.event == 0, v]
    c1, c2, c3 = st.columns(3)
    c1.metric("Median · Dead", f"{dead.median():.2f}" if len(dead) else "–")
    c2.metric("Median · Alive", f"{alive.median():.2f}" if len(alive) else "–")
    if len(dead) and len(alive):
        u, p = stats.mannwhitneyu(dead, alive)
        r = 1 - 2 * u / (len(dead) * len(alive))
        c3.metric("Mann–Whitney p / rank-biserial r", f"{p:.1e} / {-r:+.2f}")
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(df, x=v, color="status", nbins=40, barmode="overlay", opacity=.6,
                                histnorm="probability density", color_discrete_map=STATUS_COLORS,
                                title="Distribution by status"), use_container_width=True)
    q = pd.qcut(df[v], 3, duplicates="drop")
    d2 = df.assign(tertile=q.astype(str))
    b.plotly_chart(km_figure(d2, "tertile", levels=sorted(d2.tertile.unique(), key=lambda s: float(s.split(",")[0][1:])))
                   .update_layout(title=f"Kaplan–Meier by {v} tertile"), use_container_width=True)

with tab_corr:
    enc = df.copy()
    for c in ["t_stage", "n_stage", "stage_6th"]:
        enc[c] = enc[c].map({l: i for i, l in enumerate(C.ORDINAL_LEVELS[c], 1)})
    for c, pos in [("estrogen_status", "Positive"), ("progesterone_status", "Positive"), ("a_stage", "Distant")]:
        enc[c] = (enc[c] == pos).astype(int)
    cols = ["age", "tumor_size", "nodes_examined", "nodes_positive", "node_ratio", "t_stage", "n_stage", "stage_6th",
            "grade", "a_stage", "estrogen_status", "progesterone_status", "event"]
    corr = enc[cols].corr("spearman").round(2)
    st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, height=650,
                              title="Spearman correlation (ordinal variables as integers)"), use_container_width=True)
    st.caption("tumor_size ↔ t_stage and nodes_positive ↔ n_stage are strongly correlated; "
               "stage_6th is a deterministic function of T and N and is therefore not a model input.")
