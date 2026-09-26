import numpy as np
import pandas as pd

from seer_survival.data import load_clean
from seer_survival.features import FeatureEngineer, INPUT_COLS, make_preprocessor
from seer_survival.targets import horizon_label


def test_engineered_features_are_sane():
    df = load_clean()
    out = FeatureEngineer().fit(df[INPUT_COLS]).transform(df[INPUT_COLS])
    assert out["node_ratio"].between(0, 1).all()
    assert set(out["hr_status"]) <= {"ER+/PR+", "ER+/PR-", "ER-/PR+", "ER-/PR-"}
    assert "survival_months" not in out.columns and "status" not in out.columns


def test_preprocessor_output_is_numeric_and_complete():
    df = load_clean()
    Xt = make_preprocessor().fit_transform(df[INPUT_COLS])
    assert Xt.shape[0] == len(df)
    assert np.isfinite(Xt.to_numpy(dtype=float)).all()


def test_horizon_label_logic():
    time = np.array([10, 70, 30, 60, 61, 60])
    event = np.array([1, 0, 0, 1, 1, 0])
    y, known = horizon_label(time, event, horizon=60)
    # died@10 -> 1 ; alive@70 -> 0 ; alive@30 -> unknown ; died@60 -> 1 ; died@61 -> 0 ; alive@60 -> unknown
    assert known.tolist() == [True, True, False, True, True, False]
    assert y[known].tolist() == [1, 0, 1, 0]
