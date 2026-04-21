# src/trainer.py


import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from models import build_model, save_model, get_probabilities
from config import SETTINGS



def _rf_space(trial) -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_categorical("max_depth", [None, 10, 20, 30]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }


def _xgb_space(trial) -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    }


def _lgb_space(trial) -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }


def _svm_space(trial) -> Dict[str, Any]:
    return {
        "C": trial.suggest_float("C", 0.01, 100.0, log=True),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        "kernel": "rbf",
        "class_weight": "balanced",
        "probability": True,
        "random_state": 42,
    }


SEARCH_SPACES = {
    "random_forest": _rf_space,
    "xgboost": _xgb_space,
    "lightgbm": _lgb_space,
    "svm": _svm_space,
}



def cross_validate(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    n_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    metrics = {"accuracy": [], "f1": [], "roc_auc": []}

    # Ensure numpy arrays for indexing
    if hasattr(X, "values"):
        X = X.values
    if hasattr(y, "values"):
        y = y.values

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = build_model(model_name, params)
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_val)
        y_prob = get_probabilities(model, X_val)

        metrics["accuracy"].append(accuracy_score(y_val, y_pred))
        metrics["f1"].append(f1_score(y_val, y_pred, zero_division=0))
        if len(set(y_val)) > 1:
            metrics["roc_auc"].append(roc_auc_score(y_val, y_prob))

    summary = {}
    for metric, values in metrics.items():
        if values:
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values))

    return summary



def tune_hyperparameters(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 30,
    n_folds: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  Optuna not installed — skipping tuning, using defaults.")
        return {}

    space_fn = SEARCH_SPACES.get(model_name)
    if space_fn is None:
        print(f"  No search space defined for {model_name}.")
        return {}

    def objective(trial):
        params = space_fn(trial)
        cv_results = cross_validate(
            model_name, X_train, y_train,
            params=params, n_folds=n_folds, random_state=random_state
        )
        return cv_results.get("f1_mean", 0.0)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"  Best F1 (val): {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    return study.best_params



def train_and_evaluate(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    results_dir: str = "../results",
) -> Dict[str, Any]:
    
    print(f"\n  Training {model_name}...")
    t0 = time.time()

    model = build_model(model_name, params)
    model.fit(X_train, y_train)

    train_time = time.time() - t0

    y_pred = model.predict(X_val)
    y_prob = get_probabilities(model, X_val)

    results = {
        "model": model_name,
        "accuracy": round(float(accuracy_score(y_val, y_pred)), 4),
        "f1": round(float(f1_score(y_val, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_val, y_prob)), 4) if len(set(y_val)) > 1 else None,
        "train_time_s": round(train_time, 2),
        "params": params or {},
    }

    print(f"  Accuracy: {results['accuracy']}  F1: {results['f1']}  ROC-AUC: {results['roc_auc']}")
    return model, results
