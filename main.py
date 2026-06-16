#!/usr/bin/env python3
"""
AIXScript — AI Experiment DSL Runner.

Usage::

    python main.py <script.aix>

This is the CLI entry point.  It parses the given ``.aix`` file, transforms
the parse tree into an AST, and runs the interpreter.
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.parser import parse_file
from src.transformer import AIXScriptTransformer
from src.interpreter import Interpreter


def run_script(filepath: str) -> None:
    """Parse, transform, and interpret an AIXScript file.

    Args:
        filepath: Path to the ``.aix`` script.
    """
    path = Path(filepath)
    if not path.exists():
        print(f"Error: file '{filepath}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"[AIXScript] Running script: {path}")
    print("=" * 60)

    # 1. Parse
    tree = parse_file(filepath)

    # 2. Transform → AST
    transformer = AIXScriptTransformer()
    program = transformer.transform(tree)

    # 3. Interpret
    interpreter = Interpreter()
    ctx = interpreter.run(program)

    print("=" * 60)
    print("[AIXScript] Script execution completed.")


def main() -> None:
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python main.py <script.aix>", file=sys.stderr)
        sys.exit(1)

    script_path = sys.argv[1]
    run_script(script_path)


if __name__ == "__main__":
    main()
