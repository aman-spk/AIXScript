"""
AIXScript Test Suite — ML Model Tests.

Verifies that the ML engine correctly instantiates models, trains them
on sample data, and computes evaluation metrics.
"""

from __future__ import annotations

# pyrefly: ignore [missing-import]
import pytest
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
from sklearn.datasets import load_iris

from src.ml_engine import (
    get_model,
    train_model,
    evaluate_model,
    split_data,
    show_best,
    save_results,
    load_dataset,
)
from src.context import ExperimentContext


@pytest.fixture
def iris_context() -> ExperimentContext:
    """Create a pre-loaded ExperimentContext with the Iris dataset."""
    iris = load_iris(as_frame=True)
    df = iris.frame  # type: ignore[attr-defined]
    ctx = ExperimentContext()
    ctx.dataset = df
    ctx.target_column = "target"
    ctx.train_pct = 80
    ctx.test_pct = 20
    split_data(ctx)
    return ctx


class TestModelInstantiation:
    """Test that every supported model can be instantiated."""

    @pytest.mark.parametrize(
        "model_name",
        ["RandomForest", "DecisionTree", "KNN", "LogisticRegression", "SVM"],
    )
    def test_get_model(self, model_name: str) -> None:
        model = get_model(model_name)
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown model"):
            get_model("XGBoost")


class TestTraining:
    """Test model training."""

    @pytest.mark.parametrize(
        "model_name",
        ["RandomForest", "DecisionTree", "KNN", "LogisticRegression", "SVM"],
    )
    def test_train_model(self, iris_context: ExperimentContext, model_name: str) -> None:
        iris_context.current_model_name = model_name
        train_model(iris_context, model_name)
        assert model_name in iris_context.trained_models
        trained = iris_context.trained_models[model_name]
        assert hasattr(trained, "predict")

    def test_train_without_split_raises(self) -> None:
        ctx = ExperimentContext()
        ctx.current_model_name = "RandomForest"
        with pytest.raises(RuntimeError, match="Data not split"):
            train_model(ctx)

    def test_train_without_model_raises(self) -> None:
        ctx = ExperimentContext()
        with pytest.raises(RuntimeError, match="No model selected"):
            train_model(ctx)


class TestEvaluation:
    """Test metric computation."""

    @pytest.mark.parametrize("metric", ["accuracy", "precision", "recall", "f1"])
    def test_evaluate_metrics(
        self, iris_context: ExperimentContext, metric: str
    ) -> None:
        train_model(iris_context, "RandomForest")
        score = evaluate_model(iris_context, metric, "RandomForest")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_evaluate_unknown_metric_raises(
        self, iris_context: ExperimentContext
    ) -> None:
        train_model(iris_context, "RandomForest")
        with pytest.raises(ValueError, match="Unknown metric"):
            evaluate_model(iris_context, "rmse", "RandomForest")

    def test_evaluate_untrained_raises(self, iris_context: ExperimentContext) -> None:
        with pytest.raises(RuntimeError, match="not trained"):
            evaluate_model(iris_context, "accuracy", "DecisionTree")


class TestShowBest:
    """Test best-model selection."""

    def test_show_best(self, iris_context: ExperimentContext) -> None:
        for name in ["RandomForest", "DecisionTree", "KNN"]:
            train_model(iris_context, name)
            evaluate_model(iris_context, "accuracy", name)
        best = show_best(iris_context)
        assert best is not None
        assert best in ["RandomForest", "DecisionTree", "KNN"]

    def test_show_best_empty(self) -> None:
        ctx = ExperimentContext()
        result = show_best(ctx)
        assert result is None


class TestSaveResults:
    """Test result persistence."""

    def test_save_results(
        self, iris_context: ExperimentContext, tmp_path: pytest.TempPathFactory
    ) -> None:
        train_model(iris_context, "RandomForest")
        evaluate_model(iris_context, "accuracy", "RandomForest")
        outfile = str(tmp_path / "test_results.csv")  # type: ignore[operator]
        save_results(iris_context, outfile)
        df = pd.read_csv(outfile)
        assert len(df) == 1
        assert "model" in df.columns
        assert "metric" in df.columns
        assert "score" in df.columns


class TestSplitData:
    """Test train/test splitting."""

    def test_split_sizes(self, iris_context: ExperimentContext) -> None:
        total = len(iris_context.X_train) + len(iris_context.X_test)
        assert total == 150  # Iris has 150 rows

    def test_split_without_dataset_raises(self) -> None:
        ctx = ExperimentContext()
        with pytest.raises(RuntimeError, match="No dataset loaded"):
            split_data(ctx)

    def test_split_without_target_raises(self) -> None:
        ctx = ExperimentContext()
        ctx.dataset = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(RuntimeError, match="No target column"):
            split_data(ctx)
