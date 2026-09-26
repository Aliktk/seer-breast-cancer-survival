import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from common import BLUE, GREEN, ORANGE, PURPLE, clean_data
from seer_survival import config as C
from seer_survival.features import INPUT_COLS, make_preprocessor
from seer_survival.targets import horizon_label

st.title("⚠️ Leakage lab")
st.caption("Same model, different pipeline choices. Metrics are for the Dead class on a 20 % test split.")

c1, c2, c3 = st.columns(3)
target = c1.radio("Target", ["5-year death (correct)", "raw Status (ignores censoring)"])
model_name = c2.radio("Model", ["Logistic regression", "Gradient boosting"])
smote = c3.radio("SMOTE oversampling", ["none", "on training part only (valid)", "on ALL rows before split (invalid)"])
leak = st.toggle("Add `Survival Months` as an input feature (invalid: it is part of the outcome)", value=False)
thr = st.slider("Decision threshold", 0.05, 0.9, 0.5, 0.05)


@st.cache_data(show_spinner="Training…")
def run(target, model_name, smote, leak, thr):
    df = clean_data()
    X, t, e = df[INPUT_COLS], df[C.TIME_COL].to_numpy(), df[C.EVENT_COL].to_numpy()
    if target.startswith("5-year"):
        y, known = horizon_label(t, e)
        X, t, y = X[known].reset_index(drop=True), t[known], y[known]
    else:
        y = e
    prep = make_preprocessor(scale=True)
    make = (lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=42)) \
        if model_name.startswith("Gradient") else (lambda: LogisticRegression(max_iter=5000))

    if smote.startswith("on ALL"):
        from imblearn.over_sampling import SMOTE
        M = prep.fit_transform(X)
        if leak:
            M = M.assign(survival_months=t)
        Ms, ys = SMOTE(random_state=42).fit_resample(M, y)
        A, B, ya, yb = train_test_split(Ms, ys, test_size=0.2, stratify=ys, random_state=42)
    else:
        Xa, Xb, ya, yb, ta, tb = train_test_split(X, y, t, test_size=0.2, stratify=y, random_state=42)
        prep.fit(Xa)
        A, B = prep.transform(Xa), prep.transform(Xb)
        if leak:
            A, B = A.assign(survival_months=ta), B.assign(survival_months=tb)
        if smote.startswith("on training"):
            from imblearn.over_sampling import SMOTE
            A, ya = SMOTE(random_state=42).fit_resample(A, ya)
    p = make().fit(A, ya).predict_proba(B)[:, 1]
    yh = (p >= thr).astype(int)
    return {"accuracy": accuracy_score(yb, yh), "precision": precision_score(yb, yh, zero_division=0),
            "recall": recall_score(yb, yh), "F1": f1_score(yb, yh), "ROC-AUC": roc_auc_score(yb, p),
            "test size": len(yb), "test death rate": float(np.mean(yb))}


r = run(target, model_name, smote, leak, thr)
cols = st.columns(5)
for col, k in zip(cols, ["accuracy", "precision", "recall", "F1", "ROC-AUC"]):
    col.metric(k, f"{r[k]:.3f}")
fig = go.Figure(go.Bar(x=["accuracy", "precision", "recall", "F1", "ROC-AUC"],
                       y=[r[k] for k in ["accuracy", "precision", "recall", "F1", "ROC-AUC"]],
                       marker_color=[BLUE, ORANGE, GREEN, PURPLE, "#E69F00"]))
fig.add_hline(y=0.95, line_dash="dash", annotation_text="95 % target")
fig.update_layout(yaxis_range=[0, 1.05], height=360, margin=dict(t=20))
st.plotly_chart(fig, use_container_width=True)
st.caption(f"Test set: {r['test size']} rows, death rate {r['test death rate']:.1%}.")

invalid = leak or smote.startswith("on ALL")
if invalid:
    reasons = []
    if leak:
        reasons.append("**Survival Months** is only known after the outcome (time to death or last contact). "
                       "At diagnosis it does not exist.")
    if smote.startswith("on ALL"):
        reasons.append("**SMOTE before the split** puts synthetic near-copies of training patients into the test set "
                       "and makes the test set 50/50, while the real death rate is about 15 %.")
    st.error("❌ **Invalid evaluation.** These scores would not hold on new patients.\n\n- " + "\n- ".join(reasons))
else:
    st.success("✅ **Valid evaluation.** This is the kind of performance to expect on new patients: "
               "ROC-AUC around 0.70–0.75 (the main analysis reports 0.734 on its own test set).")

st.markdown("""
**Try this sequence:**
1. Correct target, no SMOTE, no leak → realistic scores.
2. Turn on *Survival Months* → accuracy and F1 jump.
3. Set SMOTE to *ALL rows before split* and switch to *Gradient boosting* → all metrics reach about 0.94–0.97.
   Flexible trees memorise the interpolated points, so contamination helps them most.
4. Set SMOTE to *training part only* → the gain disappears, which proves the jump came from contamination.
""")
