# AIXScript Architecture

This document describes the internal architecture of AIXScript — the pipeline
that transforms a human-readable `.aix` script into machine-learning results.

---

## Pipeline Overview

```
┌─────────────────┐
│  .aix Script    │   Plain-text DSL source
└───────┬─────────┘
        │
        ▼
┌─────────────────┐
│  Lark Parser    │   Tokenises & parses using grammar/aixscript.lark
└───────┬─────────┘
        │
        ▼
┌─────────────────┐
│  Parse Tree     │   Lark's concrete syntax tree (CST)
└───────┬─────────┘
        │
        ▼
┌─────────────────┐
│  Transformer    │   Converts CST → typed AST nodes
└───────┬─────────┘
        │
        ▼
┌─────────────────┐
│  AST            │   List of dataclass command nodes
└───────┬─────────┘
        │
        ▼
┌─────────────────┐
│  Interpreter    │   Walks the AST and dispatches commands
└───────┬─────────┘
        │
        ▼
┌─────────────────┐
│  ML Engine      │   scikit-learn model training & evaluation
└───────┬─────────┘
        │
        ▼
┌─────────────────┐
│  Results / CSV  │   Evaluation scores & comparison tables
└─────────────────┘
```

---

## Stage 1 — Lexing & Parsing

**Module:** `src/parser.py`
**Grammar:** `grammar/aixscript.lark`

The Lark library reads the grammar file (written in EBNF-like notation) and
constructs a parser using the **Earley** algorithm.  Earley was chosen because
it handles the grammar without requiring left-factoring or explicit precedence
rules.

Key grammar features:

| Token          | Pattern                    | Purpose                    |
|----------------|----------------------------|----------------------------|
| `FILEPATH`     | `[a-zA-Z0-9_/.\-]+`       | Dataset / output paths     |
| `IDENTIFIER`   | `[a-zA-Z_][a-zA-Z0-9_]*`  | Column names               |
| `MODEL_NAME`   | Enum of 5 classifier names | Restricts to known models  |
| `METRIC_NAME`  | Enum of 4 metric names     | Restricts to known metrics |
| `NUMBER`       | `[0-9]+`                   | Split percentages, rows    |
| `COMMENT`      | `#[^\n]*`                  | Single-line comments       |

Whitespace (spaces and tabs) is ignored; newlines are significant because they
delimit statements.

---

## Stage 2 — AST Generation

**Module:** `src/transformer.py`
**Node definitions:** `src/ast_nodes.py`

The `AIXScriptTransformer` (a Lark `Transformer` subclass) visits each rule
in the parse tree and replaces it with a strongly-typed Python dataclass.
For example, the rule `load_stmt` produces a `LoadNode(filepath=...)`.

All nodes inherit from a common `ASTNode` base class.  The root of the tree
is always a `ProgramNode` containing an ordered list of statements.

---

## Stage 3 — Interpretation

**Module:** `src/interpreter.py`

The `Interpreter` class receives a `ProgramNode` and iterates over its
statements.  For each node it calls the matching handler method
(`_exec_<NodeType>`).  Handlers either:

* mutate the **ExperimentContext** directly (e.g., setting the target column), or
* delegate to the **ML Engine** (e.g., training, evaluation).

The interpreter enforces ordering constraints (e.g., `TARGET` must follow
`LOAD`, `TRAIN` must follow `SPLIT`) and raises `RuntimeError` with a
descriptive message on violations.

---

## Stage 4 — ML Execution

**Module:** `src/ml_engine.py`

This module is intentionally **stateless** — all mutable state lives on the
`ExperimentContext`.  It provides:

| Function          | Responsibility                              |
|-------------------|---------------------------------------------|
| `load_dataset`    | Read a CSV into a DataFrame                 |
| `split_data`      | Train/test split with label encoding        |
| `get_model`       | Instantiate a scikit-learn classifier       |
| `train_model`     | Fit a model on training data                |
| `evaluate_model`  | Score a model with a given metric           |
| `show_best`       | Pick the model with the highest score       |
| `save_results`    | Write results to a CSV file                 |

---

## Shared State — ExperimentContext

**Module:** `src/context.py`

A single `ExperimentContext` dataclass flows through the entire pipeline.
It stores:

* The loaded DataFrame and target column name
* Train/test split arrays
* All trained models (name → estimator)
* Accumulated evaluation results
* The current "best" model name

Convenience predicates (`has_dataset()`, `has_split()`, etc.) make guard
checks readable in the interpreter.

---

## Utilities

**Module:** `src/utils.py`

Provides helper functions for user-facing output:

* `show_dataset_info()` — prints shape, dtypes, null counts
* `show_head()` — prints the first N rows
* `print_comparison_table()` — renders a pandas table of results

---

## Error Handling Strategy

Errors are handled at three levels:

1. **Syntax errors** — raised by Lark during parsing (e.g., unknown command).
2. **Semantic errors** — raised by the interpreter when commands are used out
   of order (e.g., `TRAIN` before `SPLIT`).
3. **Runtime errors** — raised by the ML engine (e.g., file not found,
   unknown model).

All errors propagate as Python exceptions with human-readable messages.
