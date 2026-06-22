"""
AIXScript Machine Learning Engine.

Provides model instantiation, training, and metric evaluation using
scikit-learn.  This module is intentionally stateless — all mutable data
is stored on the ``ExperimentContext`` passed in by the interpreter.
"""

from __future__ import annotations

from typing import Any, Dict

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder

from src.context import ExperimentContext


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: Dict[str, type] = {
    "RandomForest": RandomForestClassifier,
    "DecisionTree": DecisionTreeClassifier,
    "KNN": KNeighborsClassifier,
    "LogisticRegression": LogisticRegression,
    "SVM": SVC,
}


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

_METRIC_REGISTRY: Dict[str, Any] = {
    "accuracy": accuracy_score,
    "precision": lambda y_true, y_pred: precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    ),
    "recall": lambda y_true, y_pred: recall_score(
        y_true, y_pred, average="weighted", zero_division=0
    ),
    "f1": lambda y_true, y_pred: f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(filepath: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame.

    Args:
        filepath: Path to the CSV file (relative or absolute).

    Returns:
        The loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be parsed as CSV.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    except Exception as exc:
        raise ValueError(f"Failed to read dataset '{filepath}': {exc}") from exc
    print(f"[AIXScript] Loaded dataset '{filepath}' — {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def split_data(ctx: ExperimentContext) -> None:
    """Perform a train/test split and store the result on *ctx*.

    The target column is label-encoded if it contains non-numeric values.

    Args:
        ctx: The current experiment context (must have ``dataset`` and
             ``target_column`` set).

    Raises:
        RuntimeError: If the dataset or target column is missing.
    """
    if not ctx.has_dataset():
        raise RuntimeError("No dataset loaded. Use LOAD first.")
    if not ctx.has_target():
        raise RuntimeError("No target column set. Use TARGET first.")

    assert ctx.dataset is not None  # for type checker
    assert ctx.target_column is not None

    if ctx.target_column not in ctx.dataset.columns:
        raise RuntimeError(
            f"Target column '{ctx.target_column}' not found in dataset. "
            f"Available columns: {list(ctx.dataset.columns)}"
        )

    X = ctx.dataset.drop(columns=[ctx.target_column])
    y = ctx.dataset[ctx.target_column]

    # Encode non-numeric targets
    if y.dtype == object:
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=y.name)

    test_size = ctx.test_pct / 100.0
    ctx.X_train, ctx.X_test, ctx.y_train, ctx.y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    print(
        f"[AIXScript] Split data — train: {len(ctx.X_train)}, "
        f"test: {len(ctx.X_test)}  ({ctx.train_pct}/{ctx.test_pct})."
    )


def get_model(model_name: str) -> Any:
    """Instantiate a scikit-learn model by its AIXScript name.

    Args:
        model_name: One of ``RandomForest``, ``DecisionTree``, ``KNN``,
                     ``LogisticRegression``, ``SVM``.

    Returns:
        An unfitted scikit-learn estimator.

    Raises:
        ValueError: If *model_name* is not in the registry.
    """
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Supported: {list(_MODEL_REGISTRY.keys())}"
        )
    model_cls = _MODEL_REGISTRY[model_name]
    if model_name == "LogisticRegression":
        return model_cls(max_iter=1000, random_state=42)
    if model_name == "SVM":
        return model_cls(random_state=42)
    if model_name in ("RandomForest", "DecisionTree"):
        return model_cls(random_state=42)
    return model_cls()


def train_model(ctx: ExperimentContext, model_name: str | None = None) -> None:
    """Train the selected (or named) model and store it on *ctx*.

    Args:
        ctx: The experiment context (must have split data).
        model_name: Optional override for the model to train. If ``None``,
                     ``ctx.current_model_name`` is used.

    Raises:
        RuntimeError: If no model has been selected or data has not been split.
    """
    name = model_name or ctx.current_model_name
    if name is None:
        raise RuntimeError("No model selected. Use MODEL first.")
    if not ctx.has_split():
        raise RuntimeError("Data not split. Use SPLIT first.")

    model = get_model(name)
    model.fit(ctx.X_train, ctx.y_train)
    ctx.trained_models[name] = model
    ctx.current_model_name = name
    ctx.current_model = model
    print(f"[AIXScript] Trained model: {name}.")


def evaluate_model(
    ctx: ExperimentContext,
    metric: str,
    model_name: str | None = None,
) -> float:
    """Evaluate a trained model and append the result to ``ctx.results``.

    Args:
        ctx:        The experiment context.
        metric:     Metric name (``accuracy``, ``precision``, ``recall``, ``f1``).
        model_name: Optional model-name override.

    Returns:
        The computed metric score.

    Raises:
        RuntimeError: If the model has not been trained.
        ValueError:   If the metric is unknown.
    """
    name = model_name or ctx.current_model_name
    if name is None or name not in ctx.trained_models:
        raise RuntimeError(f"Model '{name}' not trained. Use TRAIN first.")
    if metric not in _METRIC_REGISTRY:
        raise ValueError(
            f"Unknown metric '{metric}'. Supported: {list(_METRIC_REGISTRY.keys())}"
        )

    model = ctx.trained_models[name]
    y_pred = model.predict(ctx.X_test)
    score = _METRIC_REGISTRY[metric](ctx.y_test, y_pred)

    ctx.results.append({"model": name, "metric": metric, "score": round(score, 4)})
    print(f"[AIXScript] {name} — {metric}: {score:.4f}")
    return float(score)


def show_best(ctx: ExperimentContext) -> str | None:
    """Determine and print the best model based on accumulated results.

    The "best" model is the one with the highest score in the results list.

    Args:
        ctx: The experiment context.

    Returns:
        The name of the best model, or ``None`` if no results exist.
    """
    if not ctx.results:
        print("[AIXScript] No results to compare.")
        return None

    best = max(ctx.results, key=lambda r: r["score"])
    ctx.best_model = best["model"]
    print(
        f"[AIXScript] Best model: {best['model']} "
        f"({best['metric']}: {best['score']:.4f})"
    )
    return ctx.best_model


def save_results(ctx: ExperimentContext, filepath: str) -> None:
    """Save accumulated results to a CSV file.

    Args:
        ctx:      The experiment context.
        filepath: Destination CSV path.
    """
    import os

    if not ctx.results:
        print("[AIXScript] No results to save.")
        return

    df = pd.DataFrame(ctx.results)

    # Ensure the output directory exists
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    df.to_csv(filepath, index=False)
    print(f"[AIXScript] Results saved to '{filepath}'.")
