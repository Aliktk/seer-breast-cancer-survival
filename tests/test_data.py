import numpy as np
import pytest

from seer_survival import config as C
from seer_survival.data import clean, consistency_flags, derive_stage_6th, load_raw, validate


@pytest.fixture(scope="module")
def df():
    return clean(load_raw())


def test_shape_and_columns(df):
    assert df.shape == (4024, 16)
    assert set(df.columns) == set(C.RAW_TO_CLEAN.values()) | {C.EVENT_COL}


def test_no_missing_and_event_encoding(df):
    assert not df.isna().any().any()
    assert df[C.EVENT_COL].sum() == (df["status"] == "Dead").sum() == 616


def test_hard_checks_pass(df):
    assert validate(df).hard_failures == []


def test_stage_is_deterministic_from_t_and_n(df):
    derived = [derive_stage_6th(t, n) for t, n in zip(df.t_stage, df.n_stage)]
    assert (np.array(derived) == df.stage_6th.to_numpy()).all()


def test_known_soft_issues_are_flagged(df):
    flags = consistency_flags(df)
    assert flags["flag_t3_size"].sum() == 7
    assert flags["flag_n_count"].sum() == 101
    assert df.duplicated().sum() == 1


def test_clean_rejects_unknown_columns():
    raw = load_raw()
    raw["unexpected"] = 1
    with pytest.raises(ValueError):
        clean(raw)
