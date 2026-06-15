"""
AIXScript Experiment Context.

This module defines the ``ExperimentContext`` class which acts as a shared
mutable state container that flows through every stage of script
interpretation.  It stores the loaded dataset, split parameters, trained
models, evaluation metrics, and comparison results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class ExperimentContext:
    """Mutable state bag for a single AIXScript experiment run.

    The interpreter populates this context as it walks the AST, and the
    ML engine reads from / writes to it when training and evaluating models.

    Attributes:
        dataset:          The loaded ``pandas.DataFrame``.
        target_column:    Name of the target column.
        train_pct:        Training-set percentage (0–100).
        test_pct:         Test-set percentage (0–100).
        current_model_name: Name of the currently selected model.
        current_model:    The instantiated scikit-learn estimator.
        trained_models:   Mapping of model name → fitted estimator.
        results:          List of per-model result dicts (model, metric, score).
        best_model:       Name of the best model after comparison.
        X_train:          Training feature matrix.
        X_test:           Test feature matrix.
        y_train:          Training label vector.
        y_test:           Test label vector.
    """

    dataset: Optional[pd.DataFrame] = None
    target_column: Optional[str] = None
    train_pct: int = 80
    test_pct: int = 20
    current_model_name: Optional[str] = None
    current_model: Optional[Any] = None
    trained_models: Dict[str, Any] = field(default_factory=dict)
    results: List[Dict[str, Any]] = field(default_factory=list)
    best_model: Optional[str] = None
    X_train: Optional[Any] = None
    X_test: Optional[Any] = None
    y_train: Optional[Any] = None
    y_test: Optional[Any] = None

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def has_dataset(self) -> bool:
        """Return ``True`` if a dataset has been loaded."""
        return self.dataset is not None

    def has_target(self) -> bool:
        """Return ``True`` if a target column has been set."""
        return self.target_column is not None

    def has_split(self) -> bool:
        """Return ``True`` if train/test arrays have been created."""
        return self.X_train is not None and self.X_test is not None

    def has_trained_model(self, name: Optional[str] = None) -> bool:
        """Return ``True`` if the given (or current) model has been trained."""
        key = name or self.current_model_name
        return key is not None and key in self.trained_models

    def reset_results(self) -> None:
        """Clear accumulated results and best-model selection."""
        self.results.clear()
        self.best_model = None
