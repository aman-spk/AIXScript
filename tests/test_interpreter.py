"""
AIXScript Test Suite — Interpreter Tests.

End-to-end tests that parse a DSL script string, transform it into an AST,
and execute it through the interpreter, then assert on the resulting
experiment context.
"""

from __future__ import annotations

import os
import pytest
import pandas as pd

from src.parser import parse_script
from src.transformer import AIXScriptTransformer
from src.interpreter import Interpreter
from src.ast_nodes import ProgramNode
from src.context import ExperimentContext


def _run_script(script: str) -> ExperimentContext:
    """Helper: parse → transform → interpret and return the context."""
    tree = parse_script(script)
    transformer = AIXScriptTransformer()
    program: ProgramNode = transformer.transform(tree)
    interpreter = Interpreter()
    return interpreter.run(program)


class TestInterpreterBasic:
    """Basic end-to-end interpreter tests."""

    def test_load_and_target(self) -> None:
        script = "LOAD data/iris.csv\nTARGET species"
        ctx = _run_script(script)
        assert ctx.has_dataset()
        assert ctx.target_column == "species"

    def test_full_pipeline(self) -> None:
        script = (
            "LOAD data/iris.csv\n"
            "TARGET species\n"
            "SPLIT 80 20\n"
            "MODEL RandomForest\n"
            "TRAIN\n"
            "EVALUATE accuracy\n"
        )
        ctx = _run_script(script)
        assert ctx.has_trained_model("RandomForest")
        assert len(ctx.results) == 1
        assert ctx.results[0]["metric"] == "accuracy"
        assert 0.0 <= ctx.results[0]["score"] <= 1.0

    def test_compare_pipeline(self) -> None:
        script = (
            "LOAD data/iris.csv\n"
            "TARGET species\n"
            "SPLIT 80 20\n"
            "COMPARE\n"
            "    RandomForest\n"
            "    DecisionTree\n"
            "    KNN\n"
            "EVALUATE accuracy\n"
            "SHOW_BEST\n"
        )
        ctx = _run_script(script)
        assert len(ctx.trained_models) == 3
        assert len(ctx.results) == 3
        assert ctx.best_model is not None

    def test_save_results(self, tmp_path: pytest.TempPathFactory) -> None:
        outfile = str(tmp_path / "out.csv")  # type: ignore[operator]
        script = (
            "LOAD data/iris.csv\n"
            "TARGET species\n"
            "SPLIT 80 20\n"
            "MODEL DecisionTree\n"
            "TRAIN\n"
            "EVALUATE f1\n"
            f"SAVE_RESULTS {outfile}\n"
        )
        ctx = _run_script(script)
        assert os.path.exists(outfile)
        df = pd.read_csv(outfile)
        assert len(df) == 1

    def test_show_head(self) -> None:
        script = "LOAD data/iris.csv\nSHOW_HEAD 3"
        ctx = _run_script(script)
        assert ctx.has_dataset()

    def test_show_dataset_info(self) -> None:
        script = "LOAD data/iris.csv\nSHOW_DATASET_INFO"
        ctx = _run_script(script)
        assert ctx.has_dataset()


class TestInterpreterErrors:
    """Test interpreter error handling."""

    def test_target_before_load(self) -> None:
        with pytest.raises(RuntimeError, match="Cannot set TARGET before LOAD"):
            _run_script("TARGET species")

    def test_split_before_load(self) -> None:
        with pytest.raises(RuntimeError, match="No dataset loaded"):
            _run_script("SPLIT 80 20")

    def test_train_before_split(self) -> None:
        with pytest.raises(RuntimeError, match="Data not split"):
            _run_script("LOAD data/iris.csv\nTARGET species\nMODEL RandomForest\nTRAIN")

    def test_evaluate_before_train(self) -> None:
        with pytest.raises(RuntimeError, match="not trained"):
            _run_script(
                "LOAD data/iris.csv\n"
                "TARGET species\n"
                "SPLIT 80 20\n"
                "MODEL RandomForest\n"
                "EVALUATE accuracy"
            )

    def test_compare_before_split(self) -> None:
        with pytest.raises(RuntimeError, match="Data not split"):
            _run_script(
                "LOAD data/iris.csv\n"
                "TARGET species\n"
                "COMPARE\n"
                "    RandomForest\n"
                "    DecisionTree\n"
            )

    def test_invalid_target_column(self) -> None:
        with pytest.raises(RuntimeError, match="not found"):
            _run_script("LOAD data/iris.csv\nTARGET nonexistent_column")


class TestInterpreterComments:
    """Verify that comments are parsed but do not affect execution."""

    def test_comments_ignored(self) -> None:
        script = (
            "# This is a comment\n"
            "LOAD data/iris.csv\n"
            "# Another comment\n"
            "TARGET species\n"
        )
        ctx = _run_script(script)
        assert ctx.has_dataset()
        assert ctx.target_column == "species"
