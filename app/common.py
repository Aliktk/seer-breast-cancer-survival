"""Cached loaders and small helpers shared by the app pages."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from seer_survival import config as C  # noqa: E402
from seer_survival.data import clean, load_raw  # noqa: E402
from seer_survival.inference import SeerPredictor  # noqa: E402
from seer_survival.targets import horizon_label  # noqa: E402
from seer_survival.training import make_split  # noqa: E402

BLUE, ORANGE, GREEN, PURPLE, YELLOW, SKY, GREY = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#6b7280")
PALETTE = [BLUE, ORANGE, GREEN, PURPLE, YELLOW, SKY]
STATUS_COLORS = {"Alive": BLUE, "Dead": ORANGE}


# ---------------------------------------------------------------------------
# Cached data / models
# ---------------------------------------------------------------------------
@st.cache_resource
def predictor() -> SeerPredictor:
    return SeerPredictor.load(C.MODELS_DIR)


@st.cache_data
def raw_data() -> pd.DataFrame:
    return load_raw(C.RAW_DATA_PATH)


@st.cache_data
def clean_data() -> pd.DataFrame:
    return clean(raw_data())


@st.cache_data
def split_data():
    """Same stratified 80/20 split (seed 42) used for training."""
    s = make_split(clean_data())
    y5_tr, k_tr = horizon_label(s["t_train"], s["e_train"])
    y5_te, k_te = horizon_label(s["t_test"], s["e_test"])
    s.update({"y5_train": y5_tr, "known_train": k_tr, "y5_test": y5_te, "known_test": k_te})
    return s


def meta() -> dict:
    return predictor().metadata


# ---------------------------------------------------------------------------
# Small statistics helpers
# ---------------------------------------------------------------------------
def wilson_ci(k, n, z: float = 1.96):
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return centre - half, centre + half


def km_curve(time, event):
    from sksurv.nonparametric import kaplan_meier_estimator
    t, s = kaplan_meier_estimator(np.asarray(event).astype(bool), np.asarray(time, float))
    return np.r_[0, t], np.r_[1, s]


def km_figure(df: pd.DataFrame, group: str | None, levels=None, height: int = 420) -> go.Figure:
    fig = go.Figure()
    if group is None:
        t, s = km_curve(df[C.TIME_COL], df[C.EVENT_COL])
        fig.add_trace(go.Scatter(x=t, y=s, mode="lines", line=dict(shape="hv", color=BLUE, width=3), name="All"))
    else:
        levels = levels or C.ORDINAL_LEVELS.get(group) or sorted(df[group].unique())
        for i, lv in enumerate(levels):
            m = df[group] == lv
            if m.sum() == 0:
                continue
            t, s = km_curve(df.loc[m, C.TIME_COL], df.loc[m, C.EVENT_COL])
            fig.add_trace(go.Scatter(x=t, y=s, mode="lines", line=dict(shape="hv", color=PALETTE[i % 6]),
                                     name=f"{lv} (n={m.sum()})"))
    fig.update_layout(xaxis_title="Months since diagnosis", yaxis_title="Survival probability",
                      height=height, margin=dict(t=30, b=10), legend=dict(orientation="h", y=-0.2))
    return fig


def logrank(df: pd.DataFrame, group: str):
    from sksurv.compare import compare_survival
    from seer_survival.targets import make_surv
    y = make_surv(df[C.TIME_COL], df[C.EVENT_COL])
    chisq, p = compare_survival(y, df[group].astype(str).to_numpy())
    return float(chisq), float(p)


def check(ok: bool) -> str:
    return "✅" if ok else "❌"
