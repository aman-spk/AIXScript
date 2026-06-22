"""
AIXScript Test Suite — Parser Tests.

Verifies that the Lark parser correctly tokenizes and structures
AIXScript source code into parse trees.
"""

from __future__ import annotations

# pyrefly: ignore [missing-import]
import pytest
# pyrefly: ignore [missing-import]
from lark import Tree

from src.parser import parse_script
from src.transformer import AIXScriptTransformer
from src.ast_nodes import (
    LoadNode,
    TargetNode,
    SplitNode,
    ModelNode,
    TrainNode,
    EvaluateNode,
    CompareNode,
    ShowBestNode,
    SaveResultsNode,
    ShowDatasetInfoNode,
    ShowHeadNode,
    CommentNode,
    ProgramNode,
)


class TestParserBasic:
    """Basic parser smoke tests."""

    def test_parse_load(self) -> None:
        tree = parse_script("LOAD data/iris.csv")
        assert isinstance(tree, Tree)
        assert tree.data == "start"

    def test_parse_target(self) -> None:
        tree = parse_script("TARGET species")
        assert isinstance(tree, Tree)

    def test_parse_split(self) -> None:
        tree = parse_script("SPLIT 80 20")
        assert isinstance(tree, Tree)

    def test_parse_model(self) -> None:
        tree = parse_script("MODEL RandomForest")
        assert isinstance(tree, Tree)

    def test_parse_train(self) -> None:
        tree = parse_script("TRAIN")
        assert isinstance(tree, Tree)

    def test_parse_evaluate(self) -> None:
        tree = parse_script("EVALUATE accuracy")
        assert isinstance(tree, Tree)

    def test_parse_show_best(self) -> None:
        tree = parse_script("SHOW_BEST")
        assert isinstance(tree, Tree)

    def test_parse_save_results(self) -> None:
        tree = parse_script("SAVE_RESULTS results.csv")
        assert isinstance(tree, Tree)

    def test_parse_show_dataset_info(self) -> None:
        tree = parse_script("SHOW_DATASET_INFO")
        assert isinstance(tree, Tree)

    def test_parse_show_head(self) -> None:
        tree = parse_script("SHOW_HEAD 10")
        assert isinstance(tree, Tree)

    def test_parse_comment(self) -> None:
        tree = parse_script("# This is a comment")
        assert isinstance(tree, Tree)


class TestParserTransform:
    """Tests that parsed trees transform into correct AST nodes."""

    def _transform(self, script: str) -> ProgramNode:
        tree = parse_script(script)
        transformer = AIXScriptTransformer()
        return transformer.transform(tree)

    def test_transform_load(self) -> None:
        prog = self._transform("LOAD data/iris.csv")
        assert len(prog.statements) == 1
        node = prog.statements[0]
        assert isinstance(node, LoadNode)
        assert node.filepath == "data/iris.csv"

    def test_transform_target(self) -> None:
        prog = self._transform("TARGET species")
        node = prog.statements[0]
        assert isinstance(node, TargetNode)
        assert node.column == "species"

    def test_transform_split(self) -> None:
        prog = self._transform("SPLIT 70 30")
        node = prog.statements[0]
        assert isinstance(node, SplitNode)
        assert node.train_pct == 70
        assert node.test_pct == 30

    def test_transform_model(self) -> None:
        for model in ["RandomForest", "DecisionTree", "KNN", "LogisticRegression", "SVM"]:
            prog = self._transform(f"MODEL {model}")
            node = prog.statements[0]
            assert isinstance(node, ModelNode)
            assert node.model_name == model

    def test_transform_train(self) -> None:
        prog = self._transform("TRAIN")
        assert isinstance(prog.statements[0], TrainNode)

    def test_transform_evaluate(self) -> None:
        for metric in ["accuracy", "precision", "recall", "f1"]:
            prog = self._transform(f"EVALUATE {metric}")
            node = prog.statements[0]
            assert isinstance(node, EvaluateNode)
            assert node.metric == metric

    def test_transform_compare(self) -> None:
        script = "COMPARE\n    RandomForest\n    DecisionTree\n    KNN"
        prog = self._transform(script)
        node = prog.statements[0]
        assert isinstance(node, CompareNode)
        assert node.models == ["RandomForest", "DecisionTree", "KNN"]

    def test_transform_show_best(self) -> None:
        prog = self._transform("SHOW_BEST")
        assert isinstance(prog.statements[0], ShowBestNode)

    def test_transform_save_results(self) -> None:
        prog = self._transform("SAVE_RESULTS outputs/results.csv")
        node = prog.statements[0]
        assert isinstance(node, SaveResultsNode)
        assert node.filepath == "outputs/results.csv"

    def test_transform_show_dataset_info(self) -> None:
        prog = self._transform("SHOW_DATASET_INFO")
        assert isinstance(prog.statements[0], ShowDatasetInfoNode)

    def test_transform_show_head(self) -> None:
        prog = self._transform("SHOW_HEAD 5")
        node = prog.statements[0]
        assert isinstance(node, ShowHeadNode)
        assert node.rows == 5

    def test_transform_comment(self) -> None:
        prog = self._transform("# hello world")
        node = prog.statements[0]
        assert isinstance(node, CommentNode)


class TestParserMultiline:
    """Test parsing of multi-statement scripts."""

    def test_full_script(self) -> None:
        script = (
            "LOAD data/iris.csv\n"
            "TARGET species\n"
            "SPLIT 80 20\n"
            "MODEL RandomForest\n"
            "TRAIN\n"
            "EVALUATE accuracy\n"
            "SAVE_RESULTS results.csv\n"
        )
        tree = parse_script(script)
        transformer = AIXScriptTransformer()
        prog = transformer.transform(tree)
        assert isinstance(prog, ProgramNode)
        assert len(prog.statements) == 7

    def test_script_with_comments(self) -> None:
        script = (
            "# Load data\n"
            "LOAD data/iris.csv\n"
            "# Set target\n"
            "TARGET species\n"
        )
        tree = parse_script(script)
        transformer = AIXScriptTransformer()
        prog = transformer.transform(tree)
        # 2 comments + 2 commands = 4
        assert len(prog.statements) == 4


class TestParserErrors:
    """Test that invalid scripts raise parse errors."""

    def test_invalid_command(self) -> None:
        with pytest.raises(Exception):
            parse_script("INVALID_COMMAND foo")

    def test_invalid_model(self) -> None:
        with pytest.raises(Exception):
            parse_script("MODEL UnknownModel")

    def test_invalid_metric(self) -> None:
        with pytest.raises(Exception):
            parse_script("EVALUATE unknown_metric")
