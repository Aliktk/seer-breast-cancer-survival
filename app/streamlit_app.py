"""SEER breast cancer survival: data lab, verification and inference app.

Run locally:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
for p in (APP_DIR, APP_DIR.parent / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common import meta  # noqa: E402

st.set_page_config(page_title="SEER Breast Cancer Survival", page_icon="🎗️", layout="wide")

pages = {
    "Start": [
        st.Page("views/overview.py", title="Overview", icon="🏠", default=True),
        st.Page("views/validity.py", title="Method check", icon="🧭"),
    ],
    "Data": [
        st.Page("views/data_lab.py", title="Data lab (raw → clean)", icon="🧪"),
        st.Page("views/eda.py", title="Live EDA & survival", icon="📈"),
    ],
    "Model": [
        st.Page("views/feature_pipeline.py", title="Feature pipeline trace", icon="🔧"),
        st.Page("views/verify.py", title="Verify results live", icon="🎯"),
        st.Page("views/leakage_lab.py", title="Leakage lab (the 95 % question)", icon="⚠️"),
    ],
    "Inference": [
        st.Page("views/patient.py", title="Patient prediction", icon="🧑‍⚕️"),
        st.Page("views/batch.py", title="Batch scoring", icon="📄"),
    ],
}

m = meta()
with st.sidebar:
    st.markdown("### 🎗️ SEER Breast Cancer")
    st.caption(f"4,024 patients · models trained {m['trained_at_utc']} UTC")
    st.markdown(
        f"**Test AUC** {m['test_metrics_classifier']['roc_auc']:.3f} · "
        f"**C-index** {m['test_metrics_survival']['c_harrell']:.3f}"
    )
    st.caption("Research & teaching tool, not for clinical use.")

st.navigation(pages).run()
