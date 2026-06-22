"""
AIXScript Lark Transformer.

This module converts a raw Lark parse tree into a well-typed
``ProgramNode`` AST.  Each grammar rule is handled by a corresponding
method in ``AIXScriptTransformer``.
"""

from __future__ import annotations

from typing import List

# pyrefly: ignore [missing-import]
from lark import Transformer, Token, Tree

from src.ast_nodes import (
    ASTNode,
    CommentNode,
    CompareNode,
    EvaluateNode,
    LoadNode,
    ModelNode,
    ProgramNode,
    SaveResultsNode,
    ShowBestNode,
    ShowDatasetInfoNode,
    ShowHeadNode,
    SplitNode,
    TargetNode,
    TrainNode,
)


class AIXScriptTransformer(Transformer):  # type: ignore[type-arg]
    """Transform a Lark parse tree into an AIXScript AST.

    Each method name matches the corresponding grammar rule and receives
    the child tokens / subtrees produced by the parser.
    """

    # ------------------------------------------------------------------
    # Top-level
    # ------------------------------------------------------------------

    def start(self, items: list) -> ProgramNode:
        """Collect all statements into a ``ProgramNode``."""
        statements: List[ASTNode] = [
            item for item in items if isinstance(item, ASTNode)
        ]
        return ProgramNode(statements=statements)

    # ------------------------------------------------------------------
    # Statements
    # ------------------------------------------------------------------

    def statement(self, items: list) -> ASTNode:
        """Unwrap the single child of a ``statement`` rule."""
        return items[0]

    def load_stmt(self, items: list) -> LoadNode:
        """Handle ``LOAD <filepath>``."""
        filepath = str(items[0]).strip()
        return LoadNode(filepath=filepath)

    def target_stmt(self, items: list) -> TargetNode:
        """Handle ``TARGET <column>``."""
        column = str(items[0]).strip()
        return TargetNode(column=column)

    def split_stmt(self, items: list) -> SplitNode:
        """Handle ``SPLIT <train_pct> <test_pct>``."""
        train_pct = int(items[0])
        test_pct = int(items[1])
        return SplitNode(train_pct=train_pct, test_pct=test_pct)

    def model_stmt(self, items: list) -> ModelNode:
        """Handle ``MODEL <model_name>``."""
        model_name = str(items[0]).strip()
        return ModelNode(model_name=model_name)

    def train_stmt(self, _items: list) -> TrainNode:
        """Handle ``TRAIN``."""
        return TrainNode()

    def evaluate_stmt(self, items: list) -> EvaluateNode:
        """Handle ``EVALUATE <metric>``."""
        metric = str(items[0]).strip()
        return EvaluateNode(metric=metric)

    def compare_stmt(self, items: list) -> CompareNode:
        """Handle ``COMPARE`` followed by one or more model names."""
        models = [str(tok).strip() for tok in items if str(tok).strip()]
        return CompareNode(models=models)

    def show_best_stmt(self, _items: list) -> ShowBestNode:
        """Handle ``SHOW_BEST``."""
        return ShowBestNode()

    def save_results_stmt(self, items: list) -> SaveResultsNode:
        """Handle ``SAVE_RESULTS <filepath>``."""
        filepath = str(items[0]).strip()
        return SaveResultsNode(filepath=filepath)

    def show_dataset_info_stmt(self, _items: list) -> ShowDatasetInfoNode:
        """Handle ``SHOW_DATASET_INFO``."""
        return ShowDatasetInfoNode()

    def show_head_stmt(self, items: list) -> ShowHeadNode:
        """Handle ``SHOW_HEAD <rows>``."""
        rows = int(items[0])
        return ShowHeadNode(rows=rows)

    def comment(self, items: list) -> CommentNode:
        """Handle comment lines (``# …``)."""
        text = str(items[0]).strip()
        return CommentNode(text=text)
