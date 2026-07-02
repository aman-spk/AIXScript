"""
AIXScript Interpreter.

The interpreter walks the AST produced by the transformer and dispatches
each node to the appropriate handler.  It maintains an ``ExperimentContext``
and delegates heavy lifting to the ML engine and utility modules.
"""

from __future__ import annotations

from typing import List

from src.ast_nodes import (
    ASTNode,
    CommentNode,
    CompareNode,
    CrossValidateNode,
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
from src.context import ExperimentContext
from src import ml_engine
from src.utils import show_dataset_info, show_head, print_comparison_table


class Interpreter:
    """Walk an AIXScript AST and execute each command.

    Attributes:
        ctx: The mutable experiment context shared across all commands.
    """

    def __init__(self) -> None:
        self.ctx = ExperimentContext()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, program: ProgramNode) -> ExperimentContext:
        """Execute every statement in the given program.

        Args:
            program: The root ``ProgramNode`` of the AST.

        Returns:
            The final ``ExperimentContext`` after all commands have run.
        """
        for stmt in program.statements:
            self._execute(stmt)
        return self.ctx

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _execute(self, node: ASTNode) -> None:
        """Dispatch a single AST node to its handler method."""
        handler_name = f"_exec_{type(node).__name__}"
        handler = getattr(self, handler_name, None)
        if handler is None:
            raise RuntimeError(f"No handler for AST node: {type(node).__name__}")
        handler(node)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _exec_LoadNode(self, node: LoadNode) -> None:
        self.ctx.dataset = ml_engine.load_dataset(node.filepath)

    def _exec_TargetNode(self, node: TargetNode) -> None:
        if not self.ctx.has_dataset():
            raise RuntimeError("Cannot set TARGET before LOAD.")
        assert self.ctx.dataset is not None
        if node.column not in self.ctx.dataset.columns:
            raise RuntimeError(
                f"Column '{node.column}' not found. "
                f"Available: {list(self.ctx.dataset.columns)}"
            )
        self.ctx.target_column = node.column
        print(f"[AIXScript] Target column set to '{node.column}'.")

    def _exec_SplitNode(self, node: SplitNode) -> None:
        self.ctx.train_pct = node.train_pct
        self.ctx.test_pct = node.test_pct
        ml_engine.split_data(self.ctx)

    def _exec_ModelNode(self, node: ModelNode) -> None:
        self.ctx.current_model_name = node.model_name
        self.ctx.current_model = ml_engine.get_model(node.model_name)
        print(f"[AIXScript] Model selected: {node.model_name}.")

    def _exec_TrainNode(self, _node: TrainNode) -> None:
        ml_engine.train_model(self.ctx)

    def _exec_EvaluateNode(self, node: EvaluateNode) -> None:
        # If we are in "compare" mode, evaluate all trained models.
        if len(self.ctx.trained_models) > 1:
            for model_name in self.ctx.trained_models:
                ml_engine.evaluate_model(self.ctx, node.metric, model_name)
            print_comparison_table(self.ctx.results)
        else:
            ml_engine.evaluate_model(self.ctx, node.metric)

    def _exec_CompareNode(self, node: CompareNode) -> None:
        """Train every model listed in the COMPARE block."""
        if not self.ctx.has_split():
            raise RuntimeError("Data not split. Use SPLIT before COMPARE.")
        self.ctx.reset_results()
        for model_name in node.models:
            ml_engine.train_model(self.ctx, model_name)

    def _exec_ShowBestNode(self, _node: ShowBestNode) -> None:
        ml_engine.show_best(self.ctx)

    def _exec_SaveResultsNode(self, node: SaveResultsNode) -> None:
        ml_engine.save_results(self.ctx, node.filepath)

    def _exec_ShowDatasetInfoNode(self, _node: ShowDatasetInfoNode) -> None:
        show_dataset_info(self.ctx)

    def _exec_ShowHeadNode(self, node: ShowHeadNode) -> None:
        show_head(self.ctx, node.rows)

    def _exec_CrossValidateNode(self, node: CrossValidateNode) -> None:
        if not self.ctx.has_dataset():
            raise RuntimeError("No dataset loaded. Use LOAD first.")
        if not self.ctx.has_target():
            raise RuntimeError("No target column set. Use TARGET first.")
        if self.ctx.current_model_name is None:
            raise RuntimeError("No model selected. Use MODEL first.")
        ml_engine.cross_validate_model(self.ctx, node.k)

    def _exec_CommentNode(self, _node: CommentNode) -> None:
        # Comments are no-ops at runtime.
        pass
