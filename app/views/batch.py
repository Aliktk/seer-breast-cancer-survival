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

st.title("📄 Batch scoring")
st.markdown("Upload a CSV with the 12 input columns. Both the **original SEER.csv headers** and the cleaned "
            "snake_case names are accepted. Extra columns (e.g. Status) are kept in the output but ignored.")
template = get_data()[INPUT_COLS].head(5)
st.download_button("Download a 5-row template", template.to_csv(index=False), "template.csv", "text/csv")
up = st.file_uploader("CSV file", type="csv")
if up is not None:
    try:
        raw = pd.read_csv(up)
        X = from_raw_format(raw)
        out = pred.predict(X)
        result = pd.concat([raw.reset_index(drop=True), out.reset_index(drop=True)], axis=1)
        for w in out.attrs.get("warnings", []):
            st.warning(w)
        st.success(f"Scored {len(result):,} rows.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean 5-year risk", f"{out.risk_5y.mean():.1%}")
        c2.metric("High-risk group", f"{(out.risk_group == 'High').mean():.0%}")
        c3.metric("Above decision threshold", f"{out.flag_high_risk.mean():.0%}")
        st.dataframe(result, use_container_width=True, height=380)
        st.download_button("Download predictions", result.to_csv(index=False), "predictions.csv", "text/csv",
                           type="primary")
    except Exception as e:  # show any parsing/validation problem to the user
        st.error(f"Could not score this file: {e}")

