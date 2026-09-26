import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import BLUE, GREEN, ORANGE, check, predictor, split_data
from seer_survival.features import INPUT_COLS

st.title("🔧 Feature pipeline trace")
st.caption("Which features the deployed models use, and how one patient flows through the pipeline.")

pred = predictor()
clf, surv = pred.classifier, pred.survival
eng_names = list(clf.named_steps["prep"].named_steps["engineer"].get_feature_names_out())
enc_names = list(clf.named_steps["prep"].named_steps["encode"].get_feature_names_out())
lr_coef = dict(zip(enc_names, clf.named_steps["model"].coef_.ravel()))
cox_names = list(surv.named_steps["prep"].named_steps["encode"].get_feature_names_out())
cox_coef = dict(zip(cox_names, surv.named_steps["model"].coef_.ravel()))

st.graphviz_chart(
    """
digraph {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef6fb", fontname="sans-serif", fontsize=11];
  raw [label="12 raw clinical inputs\\n(age, race, marital, T, N, grade,\\nA-stage, size, ER, PR,\\nnodes examined, nodes positive)"];
  eng [label="FeatureEngineer\\n+ node_ratio\\n+ log_tumor_size, log_nodes_positive\\n+ hr_status (ER/PR combo)\\n+ 0/1 recodes"];
  enc [label="ColumnTransformer\\nscale numeric/ordinal\\none-hot race, marital, hr_status\\n→ 23 columns"];
  lr  [label="L1 logistic regression\\n→ 5-year risk", fillcolor="#fdebd8"];
  cox [label="Coxnet\\n→ survival curve S(t)", fillcolor="#e3f4ec"];
  raw -> eng -> enc; enc -> lr; enc -> cox;
}
""")
st.caption("All three steps are saved inside each `.joblib` file.")

# ---------------------------------------------------------------- audit table
st.header("Every feature from the analysis: used or not?")

def used(name):
    return name in eng_names

rows = [
    ("age", "raw", used("age"), "input"),
    ("tumor_size → log_tumor_size", "notebook 05", used("log_tumor_size"), "log1p for skew"),
    ("nodes_positive → log_nodes_positive", "notebook 05", used("log_nodes_positive"), "log1p for skew"),
    ("nodes_examined", "raw", used("nodes_examined"), "input"),
    ("node_ratio (positive / examined)", "notebooks 03–05", used("node_ratio"), "strongest univariate signal (r = 0.35)"),
    ("t_stage, n_stage, grade (ordinal)", "notebook 02", all(map(used, ["t_stage", "n_stage", "grade"])), "clinical order kept"),
    ("er_positive, pr_positive", "notebook 05", used("er_positive") and used("pr_positive"), "0/1 recodes"),
    ("hr_status (ER/PR combination)", "notebooks 03, 05", used("hr_status"), "4 levels, one-hot"),
    ("distant_a_stage", "notebook 05", used("distant_a_stage"), "0/1 recode"),
    ("race, marital_status", "raw", used("race") and used("marital_status"), "one-hot"),
    ("stage_6th", "raw", used("stage_6th"), "EXCLUDED on purpose: 100 % determined by T + N (notebook 02)"),
    ("age² (non-linear age)", "notebook 05", used("age_sq"), "TESTED and rejected: no consistent CV gain"),
    ("consistency flags (T-size, N-count)", "notebook 02", used("flag_n_count"), "QC only, not predictors; sensitivity analysis done"),
    ("age_group, node-ratio tertiles", "notebooks 03–04", used("age_group"), "EDA display only (models use continuous values)"),
    ("survival_months", "raw", used("survival_months"), "NEVER an input: it is the outcome (leakage)"),
]
audit = pd.DataFrame(rows, columns=["feature", "created / studied in", "used by deployed models", "decision"])
audit["used by deployed models"] = audit["used by deployed models"].map({True: "✅ yes", False: "— no"})
st.dataframe(audit, hide_index=True, use_container_width=True, height=36 * (len(audit) + 1) + 4)

leak_free = not any(c in eng_names for c in ["survival_months", "status", "event"])
st.success(f"{check(leak_free)} No outcome column is among the model inputs.  "
           f"{check(set(INPUT_COLS) == set(pred.metadata['input_columns']))} Inputs match the training metadata.")

coef = pd.DataFrame({"feature": enc_names, "logistic coef (log-odds)": [lr_coef[n] for n in enc_names],
                     "Cox coef (log-hazard)": [cox_coef.get(n, np.nan) for n in enc_names]})
with st.expander("Model coefficients per encoded feature (0 = removed by L1 / elastic-net)"):
    st.dataframe(coef.style.format({"logistic coef (log-odds)": "{:+.3f}", "Cox coef (log-hazard)": "{:+.3f}"}),
                 hide_index=True, use_container_width=True)
    st.caption("Numeric/ordinal features are standardised, so coefficients are per standard deviation. "
               "node_ratio has one of the largest positive weights in both models.")

# ---------------------------------------------------------------- patient trace
st.header("Trace one test patient through the pipeline")
s = split_data()
Xte = s["X_test"]
i = st.number_input("Test patient index", 0, len(Xte) - 1, 1)
row = Xte.iloc[[i]]
eng = clf.named_steps["prep"].named_steps["engineer"].transform(row)
enc = clf.named_steps["prep"].named_steps["encode"].transform(eng)
c1, c2 = st.columns(2)
c1.markdown("**① Raw input**")
c1.dataframe(row.T.rename(columns={row.index[0]: "value"}).astype(str), use_container_width=True, height=460)
c2.markdown("**② Engineered features**")
c2.dataframe(eng.T.rename(columns={eng.index[0]: "value"}).astype(str), use_container_width=True, height=460)

st.markdown("**③ Encoded values × logistic coefficients → risk**")
b0 = float(clf.named_steps["model"].intercept_[0])
contrib = pd.DataFrame({"encoded value": enc.iloc[0].values, "coef": [lr_coef[n] for n in enc_names]}, index=enc_names)
contrib["contribution"] = contrib["encoded value"] * contrib["coef"]
logit = b0 + contrib.contribution.sum()
manual = 1 / (1 + np.exp(-logit))
model_p = float(clf.predict_proba(row)[:, 1][0])
nz = contrib[contrib.coef != 0].sort_values("contribution")
fig = go.Figure(go.Bar(x=nz.contribution, y=nz.index, orientation="h",
                       marker_color=[ORANGE if v > 0 else GREEN for v in nz.contribution]))
fig.update_layout(height=520, xaxis_title="contribution to log-odds", margin=dict(t=10))
st.plotly_chart(fig, use_container_width=True)
c1, c2, c3 = st.columns(3)
c1.metric("intercept + Σ contributions", f"{logit:+.3f}")
c2.metric("manual sigmoid → risk", f"{manual:.4f}")
c3.metric("model.predict_proba", f"{model_p:.4f}", delta=f"{check(abs(manual - model_p) < 1e-9)} identical",
          delta_color="off")
