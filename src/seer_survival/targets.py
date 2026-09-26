"""Outcome definitions: survival target and 5-year label.

Alive patients followed for less than the horizon have an unknown label and are
excluded from the 5-year task.
"""
from __future__ import annotations

import numpy as np
from sksurv.util import Surv

from . import config as C


def horizon_label(time, event, horizon: int = C.HORIZON_MONTHS):
    """Return (y, known): y = 1 if died by `horizon`; known = False if censored before it."""
    time = np.asarray(time)
    event = np.asarray(event).astype(bool)
    died_before = event & (time <= horizon)
    followed_past = time > horizon
    known = died_before | followed_past
    y = died_before.astype(int)
    return y, known


def make_surv(time, event):
    """scikit-survival structured array (event first, then time)."""
    return Surv.from_arrays(event=np.asarray(event).astype(bool), time=np.asarray(time, dtype=float))
