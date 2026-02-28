"""
models.py
---------
Model definitions, training pipeline with TimeSeriesSplit,
evaluation metrics, and model persistence.
"""

import numpy as np
import pandas as pd
import joblib
import os
import logging
from pathlib import Path

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)

MODEL_DIR = Path("trained_models")
MODEL_DIR.mkdir(exist_ok=True)

# ─── Model Registries ─────────────────────────────────────────────────────────
REGRESSION_MODELS = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
    ),
}

CLASSIFICATION_MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, C=0.1, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
    ),
}

# Models requiring feature scaling
NEEDS_SCALING = {"Linear Regression", "Logistic Regression"}


def build_pipeline(model_name: str, model, task: str) -> Pipeline:
    """
    Wrap model in a sklearn Pipeline. Add StandardScaler for linear models.
    """
    steps = []
    if model_name in NEEDS_SCALING:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def train_with_timeseries_split(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    task: str,
    n_splits: int = 5,
) -> dict:
    """
    Train using TimeSeriesSplit to prevent data leakage.
    Returns fold metrics and the final model trained on all data.
    """
    models_dict = REGRESSION_MODELS if task == "regression" else CLASSIFICATION_MODELS
    # clone a fresh model each time
    from sklearn.base import clone
    base_model = clone(models_dict[model_name])
    pipeline = build_pipeline(model_name, base_model, task)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []
    oof_preds = np.full(len(y), np.nan)

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        pipeline.fit(X_train, y_train)

        if task == "regression":
            preds = pipeline.predict(X_val)
            oof_preds[val_idx] = preds
            fold_metrics.append(evaluate_regression(y_val, preds))
        else:
            preds = pipeline.predict(X_val)
            proba = pipeline.predict_proba(X_val)[:, 1] if hasattr(
                pipeline.named_steps["model"], "predict_proba"
            ) else preds.astype(float)
            oof_preds[val_idx] = preds
            fold_metrics.append(evaluate_classification(y_val, preds, proba))

    # ── Train final model on ALL data ─────────────────────────────────────
    final_pipeline = build_pipeline(
        model_name, clone(models_dict[model_name]), task
    )
    final_pipeline.fit(X, y)

    return {
        "pipeline": final_pipeline,
        "fold_metrics": fold_metrics,
        "oof_preds": oof_preds,
        "model_name": model_name,
        "task": task,
    }


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred))
    return {"RMSE": rmse, "MAE": mae, "R²": r2, "Directional Accuracy": dir_acc}


def evaluate_classification(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict:
    """Compute classification evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_proba)
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(y_true, y_pred)
    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC-AUC": auc,
        "Confusion Matrix": cm,
    }


def aggregate_fold_metrics(fold_metrics: list[dict]) -> dict:
    """Mean ± std across folds for numeric metrics."""
    numeric_keys = [k for k in fold_metrics[0] if k != "Confusion Matrix"]
    agg = {}
    for k in numeric_keys:
        vals = [m[k] for m in fold_metrics]
        agg[k] = {"mean": np.mean(vals), "std": np.std(vals)}
    return agg


def get_feature_importance(result: dict, feature_cols: list[str]) -> pd.DataFrame | None:
    """
    Extract feature importances from tree-based models.
    Returns sorted DataFrame or None if not available.
    """
    model = result["pipeline"].named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        fi_df = pd.DataFrame(
            {"Feature": feature_cols, "Importance": importances}
        ).sort_values("Importance", ascending=False)
        return fi_df
    elif hasattr(model, "coef_"):
        coefs = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
        fi_df = pd.DataFrame(
            {"Feature": feature_cols, "Importance": np.abs(coefs)}
        ).sort_values("Importance", ascending=False)
        return fi_df
    return None


def save_model(result: dict, etf_label: str) -> str:
    """Persist trained pipeline to disk using joblib."""
    filename = (
        f"{etf_label.split()[0]}_{result['task']}_{result['model_name']}"
        .replace(" ", "_")
        .lower()
        + ".joblib"
    )
    path = MODEL_DIR / filename
    joblib.dump(result["pipeline"], path)
    logger.info(f"Model saved to {path}")
    return str(path)


def load_model(path: str):
    """Load a persisted pipeline."""
    return joblib.load(path)


def walk_forward_train(
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    model_name: str,
    task: str,
    train_window: int = 252,   # 1 year of trading days
    step: int = 21,            # refit monthly
) -> dict:
    """
    Walk-forward validation with fixed training window.
    Returns OOF predictions with timestamps.
    """
    from sklearn.base import clone
    models_dict = REGRESSION_MODELS if task == "regression" else CLASSIFICATION_MODELS

    preds_list = []
    dates_list = []
    n = len(X)

    for start in range(train_window, n - step + 1, step):
        train_start = max(0, start - train_window)
        X_train = X[train_start:start]
        y_train = y[train_start:start]
        X_val = X[start : start + step]
        val_dates = dates[start : start + step]

        pipe = build_pipeline(model_name, clone(models_dict[model_name]), task)
        pipe.fit(X_train, y_train)
        p = pipe.predict(X_val)

        preds_list.extend(p.tolist())
        dates_list.extend(val_dates.tolist())

    pred_series = pd.Series(preds_list, index=pd.DatetimeIndex(dates_list))
    return pred_series
