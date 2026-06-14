"""
AIXScript AST Node Definitions.

This module defines all Abstract Syntax Tree (AST) node classes used to
represent parsed AIXScript programs. Each node corresponds to a single
DSL command and carries the parameters extracted during parsing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Base Node
# ---------------------------------------------------------------------------

@dataclass
class ASTNode:
    """Base class for all AIXScript AST nodes."""

    pass


# ---------------------------------------------------------------------------
# Command Nodes
# ---------------------------------------------------------------------------

@dataclass
class LoadNode(ASTNode):
    """Represents a ``LOAD <filepath>`` command.

    Attributes:
        filepath: Path to the CSV dataset file.
    """

    filepath: str


@dataclass
class TargetNode(ASTNode):
    """Represents a ``TARGET <column>`` command.

    Attributes:
        column: Name of the target (label) column.
    """

    column: str


@dataclass
class SplitNode(ASTNode):
    """Represents a ``SPLIT <train_pct> <test_pct>`` command.

    Attributes:
        train_pct: Training split percentage (0-100).
        test_pct:  Testing split percentage (0-100).
    """

    train_pct: int
    test_pct: int


@dataclass
class ModelNode(ASTNode):
    """Represents a ``MODEL <model_name>`` command.

    Attributes:
        model_name: Name of the ML model to use.
    """

    model_name: str


@dataclass
class TrainNode(ASTNode):
    """Represents a ``TRAIN`` command."""

    pass


@dataclass
class EvaluateNode(ASTNode):
    """Represents an ``EVALUATE <metric>`` command.

    Attributes:
        metric: Evaluation metric name (accuracy, precision, recall, f1).
    """

    metric: str


@dataclass
class CompareNode(ASTNode):
    """Represents a ``COMPARE`` block with one or more model names.

    Attributes:
        models: List of model names to compare.
    """

    models: List[str] = field(default_factory=list)


@dataclass
class ShowBestNode(ASTNode):
    """Represents a ``SHOW_BEST`` command."""

    pass


@dataclass
class SaveResultsNode(ASTNode):
    """Represents a ``SAVE_RESULTS <filepath>`` command.

    Attributes:
        filepath: Destination file path for results CSV.
    """

    filepath: str


@dataclass
class ShowDatasetInfoNode(ASTNode):
    """Represents a ``SHOW_DATASET_INFO`` command."""

    pass


@dataclass
class ShowHeadNode(ASTNode):
    """Represents a ``SHOW_HEAD <rows>`` command.

    Attributes:
        rows: Number of rows to display.
    """

    rows: int


@dataclass
class CommentNode(ASTNode):
    """Represents a comment line (ignored during interpretation).

    Attributes:
        text: Raw comment text.
    """

    text: str


@dataclass
class ProgramNode(ASTNode):
    """Root node that holds the ordered list of statements in a script.

    Attributes:
        statements: Ordered sequence of AST command nodes.
    """

    statements: List[ASTNode] = field(default_factory=list)
