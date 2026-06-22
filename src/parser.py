"""
AIXScript Parser.

This module provides the Lark-based parser that reads raw ``.aix`` script
text and produces a Lark parse tree.  The grammar file is loaded from
``grammar/aixscript.lark`` relative to the project root.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

# pyrefly: ignore [missing-import]
from lark import Lark, Tree


# Resolve the grammar file path relative to *this* file's location so
# that the parser works regardless of the caller's working directory.
_GRAMMAR_DIR = Path(__file__).resolve().parent.parent / "grammar"
_GRAMMAR_FILE = _GRAMMAR_DIR / "aixscript.lark"


def _load_grammar() -> str:
    """Read the Lark grammar file and return its contents as a string.

    Returns:
        The grammar text.

    Raises:
        FileNotFoundError: If the grammar file does not exist at the
            expected location.
    """
    if not _GRAMMAR_FILE.exists():
        raise FileNotFoundError(
            f"Grammar file not found: {_GRAMMAR_FILE}"
        )
    return _GRAMMAR_FILE.read_text(encoding="utf-8")


def get_parser() -> Lark:
    """Create and return a configured Lark parser for AIXScript.

    The parser uses the Earley algorithm so that the grammar can remain
    unambiguous without requiring left-factoring.

    Returns:
        A ``lark.Lark`` parser instance.
    """
    grammar_text = _load_grammar()
    return Lark(
        grammar_text,
        parser="earley",
        start="start",
    )


def parse_script(script: str) -> Tree:
    """Parse an AIXScript source string and return its parse tree.

    Args:
        script: The raw DSL source text.

    Returns:
        A ``lark.Tree`` representing the parsed program.

    Raises:
        lark.exceptions.UnexpectedInput: If the script contains
            syntax errors.
    """
    parser = get_parser()
    return parser.parse(script)


def parse_file(filepath: Union[str, os.PathLike]) -> Tree:
    """Parse an AIXScript file from disk.

    Args:
        filepath: Path to the ``.aix`` script file.

    Returns:
        A ``lark.Tree`` representing the parsed program.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Script file not found: {path}")
    script = path.read_text(encoding="utf-8")
    return parse_script(script)
