"""
AIXScript Utilities.

Helper functions used across the project — dataset inspection, table
formatting, and general-purpose output routines.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from src.context import ExperimentContext


def show_dataset_info(ctx: ExperimentContext) -> None:
    """Print summary information about the loaded dataset.

    Displays shape, column names, data types, and null counts.

    Args:
        ctx: The experiment context (must have a loaded dataset).

    Raises:
        RuntimeError: If no dataset has been loaded.
    """
    if not ctx.has_dataset():
        raise RuntimeError("No dataset loaded. Use LOAD first.")

    assert ctx.dataset is not None

    df = ctx.dataset
    print("\n" + "=" * 60)
    print("  DATASET INFORMATION")
    print("=" * 60)
    print(f"  Shape     : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Columns   : {list(df.columns)}")
    print(f"  Dtypes    :")
    for col in df.columns:
        print(f"    {col:20s}  {df[col].dtype}")
    print(f"  Null counts:")
    for col in df.columns:
        nulls = df[col].isnull().sum()
        print(f"    {col:20s}  {nulls}")
    print("=" * 60 + "\n")


def show_head(ctx: ExperimentContext, rows: int = 5) -> None:
    """Print the first *rows* rows of the loaded dataset.

    Args:
        ctx:  The experiment context (must have a loaded dataset).
        rows: Number of rows to display (default 5).

    Raises:
        RuntimeError: If no dataset has been loaded.
    """
    if not ctx.has_dataset():
        raise RuntimeError("No dataset loaded. Use LOAD first.")

    assert ctx.dataset is not None

    print(f"\n[AIXScript] First {rows} rows:")
    print(ctx.dataset.head(rows).to_string(index=False))
    print()


def print_comparison_table(results: List[Dict[str, Any]]) -> None:
    """Pretty-print a comparison table of model evaluation results.

    Args:
        results: A list of result dicts with keys ``model``, ``metric``,
                 and ``score``.
    """
    if not results:
        print("[AIXScript] No results to display.")
        return

    df = pd.DataFrame(results)
    print("\n" + "=" * 50)
    print("  MODEL COMPARISON RESULTS")
    print("=" * 50)
    print(df.to_string(index=False))
    print("=" * 50 + "\n")
