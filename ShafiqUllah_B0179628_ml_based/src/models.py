# src/models.py


import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb


# ---------------------------------------------------------------------------
# Default hyperparameters (pre-optimised reasonable defaults)
# ---------------------------------------------------------------------------

DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "random_forest": {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    },
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 1.0,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    },
    "lightgbm": {
        "n_estimators": 300,
        "max_depth": -1,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    },
    "svm": {
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale",
        "class_weight": "balanced",
        "probability": True,
        "random_state": 42,
    },
}


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(model_name: str, params: Optional[Dict[str, Any]] = None):
    """
    Build an sklearn-compatible classifier.

    Parameters
    ----------
    model_name : One of 'random_forest', 'xgboost', 'lightgbm', 'svm'
    params     : Optional dict of hyperparameters. Uses defaults if None.

    Returns
    -------
    Fitted-ready sklearn estimator.
    """
    p = dict(DEFAULT_PARAMS.get(model_name, {}))
    if params:
        p.update(params)

    if model_name == "random_forest":
        return RandomForestClassifier(**p)

    elif model_name == "xgboost":
        p.pop("use_label_encoder", None)  # deprecated in newer xgb
        return xgb.XGBClassifier(**p)

    elif model_name == "lightgbm":
        return lgb.LGBMClassifier(**p)

    elif model_name == "svm":
        # SVC with probability=True is slow on large datasets;
        # use CalibratedClassifierCV for better calibration on smaller sets
        return SVC(**p)

    else:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Choose from: random_forest, xgboost, lightgbm, svm")


def save_model(model, path: str) -> None:
    """Save a trained model to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Model saved → {path}")


def load_model(path: str):
    """Load a trained model from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def get_probabilities(model, X: np.ndarray) -> np.ndarray:
    """Get class-1 probabilities from any model."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # Sigmoid transform
        return 1.0 / (1.0 + np.exp(-scores))
    else:
        return model.predict(X).astype(float)
