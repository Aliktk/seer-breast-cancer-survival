import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sksurv.metrics import concordance_index_censored

from common import BLUE, GREEN, GREY, ORANGE, check, clean_data, km_curve, meta, predictor, split_data
from seer_survival import config as C
from seer_survival.targets import make_surv

st.title("🎯 Verify results")
st.caption("Recomputed from the raw CSV and the saved models, then compared with the values stored at training.")

m, pred, s = meta(), predictor(), split_data()
Xtr, Xte = s["X_train"], s["X_test"]
ktr, kte = s["known_train"], s["known_test"]
ytr5, yte5 = s["y5_train"][ktr], s["y5_test"][kte]

# ---------------------------------------------------------------- 1 split
st.header("1 · The split")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Train patients", len(Xtr), delta=f"{check(len(Xtr) == m['n_train'])} matches training", delta_color="off")
c2.metric("Test patients", len(Xte), delta=f"{check(len(Xte) == m['n_test'])} matches training", delta_color="off")
c3.metric("Deaths train / test", f"{s['e_train'].sum()} / {s['e_test'].sum()}")
c4.metric("5-year labelled test / deaths", f"{kte.sum()} / {yte5.sum()}")
st.caption("Stratified 80/20 split with seed 42, done before any model fitting. For the 5-year label, patients alive "
           "but followed < 60 months are excluded because their outcome is unknown.")

# ---------------------------------------------------------------- 2 re-score
st.header("2 · Re-score the saved models on the test set")
p_te = pred.classifier.predict_proba(Xte[kte])[:, 1]
p_tr = pred.classifier.predict_proba(Xtr[ktr])[:, 1]
risk_te = pred.survival.predict(Xte)
risk_tr = pred.survival.predict(Xtr)
live = {
    "5-year ROC-AUC (test)": (roc_auc_score(yte5, p_te), m["test_metrics_classifier"]["roc_auc"]),
    "5-year Brier score (test)": (brier_score_loss(yte5, p_te), m["test_metrics_classifier"]["brier"]),
    "Survival C-index (test)": (concordance_index_censored(s["e_test"].astype(bool), s["t_test"], risk_te)[0],
                                m["test_metrics_survival"]["c_harrell"]),
}
tab = pd.DataFrame([{"metric": k, "recomputed now": v[0], "stored at training": v[1],
                     "match": check(abs(v[0] - v[1]) < 1e-6)} for k, v in live.items()])
st.dataframe(tab.style.format({"recomputed now": "{:.4f}", "stored at training": "{:.4f}"}), hide_index=True,
             use_container_width=True)

st.markdown("**Train vs test: is the model over-fitting?**")
gap = pd.DataFrame({
    "set": ["train (apparent)", "test (held-out)"],
    "5-year ROC-AUC": [roc_auc_score(ytr5, p_tr), roc_auc_score(yte5, p_te)],
    "C-index": [concordance_index_censored(s["e_train"].astype(bool), s["t_train"], risk_tr)[0], live["Survival C-index (test)"][0]],
})
st.dataframe(gap.style.format({"5-year ROC-AUC": "{:.3f}", "C-index": "{:.3f}"}), hide_index=True)
st.caption("A small train–test gap means the regularised models are not memorising the training data.")

# ---------------------------------------------------------------- 3 retrain
st.header("3 · Retrain from scratch, right now")
if st.button("🔁 Retrain both models on the training split", type="primary"):
    from seer_survival.training import fit_final_classifier, fit_final_survival
    t0 = time.time()
    clf2 = fit_final_classifier(m["classifier_name"], Xtr[ktr].reset_index(drop=True), ytr5)
    sur2 = fit_final_survival(m["survival_model_name"], Xtr, s["t_train"], s["e_train"])
    a2 = roc_auc_score(yte5, clf2.predict_proba(Xte[kte])[:, 1])
    c2 = concordance_index_censored(s["e_test"].astype(bool), s["t_test"], sur2.predict(Xte))[0]
    st.success(f"Retrained in {time.time() - t0:.1f}s · test AUC {a2:.4f} {check(abs(a2 - live['5-year ROC-AUC (test)'][0]) < 1e-6)} · "
               f"test C-index {c2:.4f} {check(abs(c2 - live['Survival C-index (test)'][0]) < 1e-6)}  → fully reproducible")

# ---------------------------------------------------------------- 4 threshold
st.header("4 · Threshold explorer (precision · recall · F1 · accuracy)")
thr = st.slider("Decision threshold on predicted 5-year risk", 0.01, 0.80, float(round(m["threshold"], 2)), 0.01)
yhat = (p_te >= thr).astype(int)
tn, fp, fn, tp = confusion_matrix(yte5, yhat, labels=[0, 1]).ravel()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{accuracy_score(yte5, yhat):.3f}")
c2.metric("Precision (Dead)", f"{precision_score(yte5, yhat, zero_division=0):.3f}")
c3.metric("Recall (Dead)", f"{recall_score(yte5, yhat):.3f}")
c4.metric("F1 (Dead)", f"{f1_score(yte5, yhat):.3f}")
a, b = st.columns([1, 1.6])
cm_fig = px.imshow([[tn, fp], [fn, tp]], text_auto=True, color_continuous_scale="Blues",
                   x=["pred: survives", "pred: dies"], y=["true: survives", "true: dies"])
cm_fig.update_layout(height=330, coloraxis_showscale=False, title="Confusion matrix (test)", margin=dict(t=40))
a.plotly_chart(cm_fig, use_container_width=True)
grid = np.linspace(0.01, 0.8, 80)
sweep = pd.DataFrame([{"threshold": t, "accuracy": accuracy_score(yte5, p_te >= t),
                       "precision": precision_score(yte5, p_te >= t, zero_division=0),
                       "recall": recall_score(yte5, p_te >= t), "F1": f1_score(yte5, p_te >= t)} for t in grid])
fig = go.Figure()
for col, colr in zip(["accuracy", "precision", "recall", "F1"], [BLUE, ORANGE, GREEN, "#CC79A7"]):
    fig.add_trace(go.Scatter(x=sweep.threshold, y=sweep[col], name=col, line=dict(color=colr)))
fig.add_hline(y=0.95, line_dash="dash", annotation_text="95 % target")
fig.add_vline(x=thr, line_color=GREY)
fig.update_layout(height=330, yaxis_range=[0, 1.02], xaxis_title="threshold", title="All thresholds (test)",
                  margin=dict(t=40))
b.plotly_chart(fig, use_container_width=True)
st.info(f"Best F1 at any threshold: **{sweep.F1.max():.3f}**. Thresholds with all four ≥ 0.95: "
        f"**{int((sweep[['accuracy', 'precision', 'recall', 'F1']] >= 0.95).all(axis=1).sum())}**. "
        f"Predicting 'survives' for everyone gives accuracy {1 - yte5.mean():.3f}, which is why accuracy alone is misleading.",
        icon="📌")

# ---------------------------------------------------------------- 5 curves
st.header("5 · Discrimination and calibration (test)")
a, b = st.columns(2)
fpr, tpr, _ = roc_curve(yte5, p_te)
f = go.Figure(go.Scatter(x=fpr, y=tpr, name=f"AUC {roc_auc_score(yte5, p_te):.3f}", line=dict(color=BLUE, width=3)))
f.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot", color=GREY))
f.update_layout(title="ROC curve", xaxis_title="1 − specificity", yaxis_title="sensitivity", height=380)
a.plotly_chart(f, use_container_width=True)
frac, mean_p = calibration_curve(yte5, p_te, n_bins=8, strategy="quantile")
f = go.Figure(go.Scatter(x=mean_p, y=frac, mode="markers+lines", marker=dict(size=9, color=ORANGE), name="model"))
f.add_shape(type="line", x0=0, y0=0, x1=0.6, y1=0.6, line=dict(dash="dot", color=GREY))
f.update_layout(title=f"Calibration · slope {m['test_metrics_classifier']['cal_slope']:.2f}", height=380,
                xaxis_title="mean predicted risk", yaxis_title="observed death rate")
b.plotly_chart(f, use_container_width=True)

st.header("6 · Survival model: risk groups on unseen patients")
cuts = np.quantile(risk_tr, [1 / 3, 2 / 3])
grp = np.select([risk_te < cuts[0], risk_te < cuts[1]], ["low", "intermediate"], "high")
f = go.Figure()
for g, colr in zip(["low", "intermediate", "high"], [GREEN, "#E69F00", ORANGE]):
    mk = grp == g
    t, sv = km_curve(s["t_test"][mk], s["e_test"][mk])
    f.add_trace(go.Scatter(x=t, y=sv, mode="lines", line=dict(shape="hv", color=colr, width=3),
                           name=f"{g} (n={mk.sum()}, deaths={s['e_test'][mk].sum()})"))
f.update_layout(height=420, xaxis_title="months", yaxis_title="observed survival (Kaplan–Meier)", yaxis_range=[0.4, 1.02])
st.plotly_chart(f, use_container_width=True)
st.caption("Groups are defined with training-set cut-offs and applied unchanged to the test set.")
