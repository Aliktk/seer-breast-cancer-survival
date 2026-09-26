import pandas as pd
import pytest

from seer_survival import config as C
from seer_survival.inference import SeerPredictor

pytestmark = pytest.mark.skipif(not (C.MODELS_DIR / "metadata.json").exists(),
                                reason="run scripts/train.py first")

PATIENT = {
    "age": 55, "race": "White", "marital_status": "Married", "t_stage": "T2", "n_stage": "N1",
    "grade": 2, "a_stage": "Regional", "tumor_size": 30, "estrogen_status": "Positive",
    "progesterone_status": "Positive", "nodes_examined": 14, "nodes_positive": 2,
}


@pytest.fixture(scope="module")
def pred():
    return SeerPredictor.load()


def test_predict_single(pred):
    out = pred.predict(pd.DataFrame([PATIENT]))
    assert 0 < out.loc[0, "risk_5y"] < 1
    s = out.loc[0, ["survival_36m", "survival_60m", "survival_84m"]].to_numpy(dtype=float)
    assert (s[:-1] >= s[1:]).all(), "survival must be non-increasing"


def test_higher_stage_means_higher_risk(pred):
    worse = {**PATIENT, "t_stage": "T4", "n_stage": "N3", "grade": 3, "nodes_positive": 14,
             "tumor_size": 80, "estrogen_status": "Negative", "progesterone_status": "Negative"}
    out = pred.predict(pd.DataFrame([PATIENT, worse]))
    assert out.loc[1, "risk_5y"] > out.loc[0, "risk_5y"]


def test_invalid_input_raises(pred):
    with pytest.raises(ValueError):
        pred.predict(pd.DataFrame([{**PATIENT, "t_stage": "T9"}]))
    with pytest.raises(ValueError):
        pred.predict(pd.DataFrame([{**PATIENT, "nodes_positive": 30}]))


def test_extrapolation_warning(pred):
    out = pred.predict(pd.DataFrame([{**PATIENT, "age": 85}]))
    assert any("age" in w for w in out.attrs["warnings"])


def test_explain_returns_all_inputs(pred):
    ex = pred.explain(pd.DataFrame([PATIENT]))
    assert len(ex) == 12


def test_raw_format_upload_is_accepted(pred):
    from seer_survival.data import load_raw
    from seer_survival.inference import from_raw_format
    raw = load_raw().head(20)
    out = pred.predict(from_raw_format(raw))
    assert len(out) == 20 and out["risk_5y"].between(0, 1).all()
