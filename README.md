# AIXScript: AI Experiment DSL

<p align="center">
  <strong>A Domain-Specific Language for Machine Learning Experimentation</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/parser-Lark-orange?style=flat-square" alt="Lark">
  <img src="https://img.shields.io/badge/ML-scikit--learn-green?style=flat-square" alt="scikit-learn">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square" alt="License">
</p>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Motivation](#motivation)
- [What is a DSL?](#what-is-a-dsl)
- [Why AIXScript?](#why-aixscript)
- [Architecture](#architecture)
- [Grammar Overview](#grammar-overview)
- [Supported Commands](#supported-commands)
- [Supported Models](#supported-models)
- [Supported Metrics](#supported-metrics)
- [Example Scripts](#example-scripts)
- [Installation](#installation)
- [Running Instructions](#running-instructions)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)
- [Screenshots](#screenshots)
- [Future Work](#future-work)

---

## Project Overview

**AIXScript** is a Domain-Specific Language (DSL) that allows users to define
machine learning experiments using simple, English-like commands instead of
writing Python code.  It abstracts away boilerplate such as data loading,
splitting, model instantiation, training, evaluation, and result persistence
into a clean, readable script format.

```
LOAD data/iris.csv
TARGET species
SPLIT 80 20
MODEL RandomForest
TRAIN
EVALUATE accuracy
SAVE_RESULTS results.csv
```

The seven lines above replace roughly 40 lines of equivalent Python/scikit-learn
code.

---

## Motivation

Machine learning experimentation involves repetitive boilerplate:

1. Load data with pandas.
2. Encode categorical columns.
3. Split into train/test sets.
4. Instantiate a model.
5. Fit and predict.
6. Compute metrics.
7. Record results.

For beginners and rapid-prototyping scenarios this boilerplate is a barrier.
AIXScript eliminates it by providing a **declarative** experiment description
language that is parsed and executed automatically.

---

## What is a DSL?

A **Domain-Specific Language** is a programming language tailored to a
particular application domain.  Unlike general-purpose languages (Python, Java),
DSLs trade generality for expressiveness within their niche.

| Aspect           | General-Purpose Language | DSL                  |
|------------------|--------------------------|----------------------|
| Scope            | Unlimited                | Narrow domain        |
| Learning curve   | Steep                    | Gentle               |
| Verbosity        | Higher                   | Minimal              |
| Flexibility      | Maximum                  | Constrained          |

Examples of well-known DSLs include SQL (databases), HTML (web markup), and
Makefile (build systems).

---

## Why AIXScript?

| Pain Point                                 | AIXScript Solution                        |
|-------------------------------------------|-------------------------------------------|
| Verbose boilerplate code                  | English-like commands                     |
| Difficult for non-programmers             | No Python knowledge required              |
| Hard to compare models                    | Built-in `COMPARE` + `SHOW_BEST`          |
| Results scattered in notebooks            | Automatic CSV export with `SAVE_RESULTS`  |
| Inconsistent experiment tracking          | Deterministic, reproducible scripts       |

---

## Architecture

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
│  Parse Tree     │   Lark's concrete syntax tree
└───────┬─────────┘
        │
        ▼
┌─────────────────┐
│  Transformer    │   Converts parse tree → typed AST nodes
└───────┬─────────┘
        │
        ▼
┌─────────────────┐
│  AST            │   Ordered list of dataclass command nodes
└───────┬─────────┘
        │
        ▼
┌─────────────────┐
│  Interpreter    │   Walks AST, dispatches commands
└───────┬─────────┘
        │
        ▼
┌─────────────────┐
│  ML Engine      │   scikit-learn training & evaluation
└───────┬─────────┘
        │
        ▼
┌─────────────────┐
│  Results / CSV  │   Scores, comparison tables, saved output
└─────────────────┘
```

Detailed architecture documentation: [`docs/architecture.md`](docs/architecture.md)

---

## Grammar Overview

The grammar is defined in [`grammar/aixscript.lark`](grammar/aixscript.lark)
using the Lark EBNF format.  Key design decisions:

- **Line-oriented:** each line is one statement.
- **Fixed keyword set:** commands are all UPPER_CASE.
- **Typed terminals:** model names and metric names are enumerated in the
  grammar, so invalid values are caught at parse time.
- **Comments:** lines starting with `#` are ignored.

---

## Supported Commands

| Command                          | Description                              |
|----------------------------------|------------------------------------------|
| `LOAD <file>`                    | Load a CSV dataset                       |
| `TARGET <column>`                | Set the target variable                  |
| `SPLIT <train%> <test%>`         | Split into train/test sets               |
| `MODEL <name>`                   | Select a single model                    |
| `TRAIN`                          | Train the selected model                 |
| `EVALUATE <metric>`              | Evaluate trained model(s)                |
| `COMPARE` + indented model list  | Train multiple models for comparison     |
| `SHOW_BEST`                      | Display the best-performing model        |
| `SAVE_RESULTS <file>`            | Export results to CSV                    |
| `SHOW_DATASET_INFO`              | Print dataset summary statistics         |
| `SHOW_HEAD <n>`                  | Print the first *n* rows                 |

Full language specification: [`docs/language_spec.md`](docs/language_spec.md)

---

## Supported Models

| AIXScript Name      | scikit-learn Class             |
|---------------------|--------------------------------|
| `RandomForest`      | `RandomForestClassifier`       |
| `DecisionTree`      | `DecisionTreeClassifier`       |
| `KNN`               | `KNeighborsClassifier`         |
| `LogisticRegression` | `LogisticRegression`          |
| `SVM`               | `SVC`                          |

---

## Supported Metrics

| Metric      | scikit-learn Function            |
|-------------|----------------------------------|
| `accuracy`  | `accuracy_score`                 |
| `precision` | `precision_score` (weighted)     |
| `recall`    | `recall_score` (weighted)        |
| `f1`        | `f1_score` (weighted)            |

---

## Example Scripts

### Basic — Single Model

```
LOAD data/iris.csv
TARGET species
SPLIT 80 20
MODEL RandomForest
TRAIN
EVALUATE accuracy
SAVE_RESULTS outputs/basic_results.csv
```

### Advanced — Model Comparison

```
LOAD data/iris.csv
TARGET species
SHOW_DATASET_INFO
SPLIT 80 20

COMPARE
    RandomForest
    DecisionTree
    KNN

EVALUATE accuracy
SHOW_BEST
SAVE_RESULTS outputs/comparison_results.csv
```

More examples in the [`examples/`](examples/) directory.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/AIXScript.git
cd AIXScript

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate      # macOS / Linux
# venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Running Instructions

```bash
# Activate the virtual environment
source venv/bin/activate

# Run the basic example
python main.py examples/basic.aix

# Run the comparison example
python main.py examples/compare_models.aix
```

### Sample Output

```
[AIXScript] Running script: examples/basic.aix
============================================================
[AIXScript] Loaded dataset 'data/iris.csv' — 150 rows, 5 columns.
[AIXScript] Target column set to 'species'.
[AIXScript] Split data — train: 120, test: 30  (80/20).
[AIXScript] Model selected: RandomForest.
[AIXScript] Trained model: RandomForest.
[AIXScript] RandomForest — accuracy: 1.0000
[AIXScript] Results saved to 'outputs/basic_results.csv'.
============================================================
[AIXScript] Script execution completed.
```

---

## Running Tests

```bash
# Activate the virtual environment
source venv/bin/activate

# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_parser.py -v
pytest tests/test_models.py -v
pytest tests/test_interpreter.py -v
```

---

## Project Structure

```
AIXScript/
├── grammar/
│   └── aixscript.lark          # Lark EBNF grammar
├── src/
│   ├── __init__.py
│   ├── parser.py               # Lark parser wrapper
│   ├── transformer.py          # Parse tree → AST transformer
│   ├── ast_nodes.py            # AST node dataclasses
│   ├── interpreter.py          # AST walker / command dispatcher
│   ├── ml_engine.py            # scikit-learn model & metric engine
│   ├── context.py              # Experiment state container
│   └── utils.py                # Display helpers
├── examples/
│   ├── basic.aix               # Single-model example
│   └── compare_models.aix      # Multi-model comparison example
├── data/
│   └── iris.csv                # Classic Iris dataset
├── tests/
│   ├── __init__.py
│   ├── test_parser.py          # Parser & transformer tests
│   ├── test_models.py          # ML engine tests
│   └── test_interpreter.py     # End-to-end interpreter tests
├── docs/
│   ├── architecture.md         # Internal architecture guide
│   └── language_spec.md        # Language reference
├── outputs/                    # Generated result files
├── main.py                     # CLI entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── .gitignore
```

---

## Screenshots

### Running the basic example

```
$ python main.py examples/basic.aix

[AIXScript] Running script: examples/basic.aix
============================================================
[AIXScript] Loaded dataset 'data/iris.csv' — 150 rows, 5 columns.
[AIXScript] Target column set to 'species'.
[AIXScript] Split data — train: 120, test: 30  (80/20).
[AIXScript] Model selected: RandomForest.
[AIXScript] Trained model: RandomForest.
[AIXScript] RandomForest — accuracy: 1.0000
[AIXScript] Results saved to 'outputs/basic_results.csv'.
============================================================
[AIXScript] Script execution completed.
```

### Running the comparison example

```
$ python main.py examples/compare_models.aix

[AIXScript] Running script: examples/compare_models.aix
============================================================
[AIXScript] Loaded dataset 'data/iris.csv' — 150 rows, 5 columns.
[AIXScript] Split data — train: 120, test: 30  (80/20).
[AIXScript] Trained model: RandomForest.
[AIXScript] Trained model: DecisionTree.
[AIXScript] Trained model: KNN.
[AIXScript] RandomForest — accuracy: 1.0000
[AIXScript] DecisionTree — accuracy: 1.0000
[AIXScript] KNN — accuracy: 1.0000

==================================================
  MODEL COMPARISON RESULTS
==================================================
         model   metric  score
  RandomForest accuracy 1.0000
  DecisionTree accuracy 1.0000
           KNN accuracy 1.0000
==================================================

[AIXScript] Best model: RandomForest (accuracy: 1.0000)
[AIXScript] Results saved to 'outputs/comparison_results.csv'.
============================================================
[AIXScript] Script execution completed.
```

---

## Future Work

- **Regression support** — extend to regression tasks with metrics like RMSE and MAE.
- **Hyperparameter tuning** — add `TUNE` command with grid/random search.
- **Cross-validation** — add `CROSS_VALIDATE k` command.
- **Data preprocessing** — `NORMALIZE`, `FILL_MISSING`, `ENCODE` commands.
- **Visualization** — `PLOT confusion_matrix`, `PLOT feature_importance`.
- **Custom models** — allow users to register Python classes as named models.
- **Export to Python** — transpile `.aix` scripts to standalone Python files.
- **Web interface** — browser-based editor with syntax highlighting and live output.

---

## License

This project is developed as a university project for educational purposes.

---

<p align="center">
  <em>Built with ❤️ using Python, Lark, and scikit-learn</em>
</p>
