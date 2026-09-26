"""Candidate models for both tasks. No class re-weighting, since it distorts probabilities."""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis

from .config import RANDOM_STATE
from .features import make_preprocessor

try:  # optional heavy dependencies; the app does not need them
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None


# --------------------------------------------------------------------------
# Task B: fixed-horizon mortality classifiers
# --------------------------------------------------------------------------
def classifier_zoo() -> dict:
    """name -> (estimator, needs_scaling)."""
    zoo = {
        "LogReg (L2)": (LogisticRegression(C=1.0, max_iter=5000, random_state=RANDOM_STATE), True),
        "LogReg (L1)": (LogisticRegression(C=0.5, penalty="l1", solver="liblinear", max_iter=5000,
                                          random_state=RANDOM_STATE), True),
        "SVM (RBF)": (SVC(C=1.0, probability=True, random_state=RANDOM_STATE), True),
        "MLP": (MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-2, max_iter=2000,
                              early_stopping=True, random_state=RANDOM_STATE), True),
        "Random Forest": (RandomForestClassifier(n_estimators=500, min_samples_leaf=10,
                                                 max_features="sqrt", n_jobs=-1,
                                                 random_state=RANDOM_STATE), False),
        "HistGradBoost": (HistGradientBoostingClassifier(learning_rate=0.05, max_depth=3,
                                                         max_iter=300, l2_regularization=1.0,
                                                         random_state=RANDOM_STATE), False),
    }
    if XGBClassifier is not None:
        zoo["XGBoost"] = (XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=3,
                                        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                                        reg_lambda=1.0, eval_metric="logloss", n_jobs=-1,
                                        random_state=RANDOM_STATE), False)
    if LGBMClassifier is not None:
        zoo["LightGBM"] = (LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15,
                                          min_child_samples=20, subsample=0.8, subsample_freq=1,
                                          colsample_bytree=0.8, reg_lambda=1.0, verbose=-1,
                                          random_state=RANDOM_STATE), False)
    return zoo


def make_classifier_pipeline(estimator, scale: bool) -> Pipeline:
    return Pipeline([("prep", make_preprocessor(scale=scale)), ("model", estimator)])


# --------------------------------------------------------------------------
# Task A: survival models
# --------------------------------------------------------------------------
def survival_zoo() -> dict:
    """name -> (estimator, needs_scaling)."""
    return {
        "Cox PH (ridge)": (CoxPHSurvivalAnalysis(alpha=0.1, ties="efron"), True),
        "Coxnet (elastic net)": (CoxnetSurvivalAnalysis(l1_ratio=0.5, alphas=[0.01],
                                                        fit_baseline_model=True, max_iter=100000), True),
        "Random Survival Forest": (RandomSurvivalForest(n_estimators=300, min_samples_leaf=15,
                                                        max_features="sqrt", n_jobs=-1,
                                                        random_state=RANDOM_STATE), False),
        "Gradient Boosted Survival": (GradientBoostingSurvivalAnalysis(n_estimators=300, learning_rate=0.05,
                                                                       max_depth=3, subsample=0.8,
                                                                       min_samples_leaf=10,
                                                                       random_state=RANDOM_STATE), False),
    }


def make_survival_pipeline(estimator, scale: bool) -> Pipeline:
    return Pipeline([("prep", make_preprocessor(scale=scale)), ("model", estimator)])
