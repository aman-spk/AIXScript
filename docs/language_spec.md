# AIXScript Language Specification

**Version:** 1.0
**File extension:** `.aix`

---

## Overview

AIXScript is a line-oriented, case-sensitive DSL.  Each non-blank, non-comment
line contains exactly one command.  Commands are executed sequentially from top
to bottom.

---

## Lexical Rules

| Element      | Rule                                          |
|-------------|-----------------------------------------------|
| **Commands** | UPPER_CASE keywords (e.g., `LOAD`, `TRAIN`)  |
| **Strings**  | Bare tokens — no quotes needed               |
| **Numbers**  | Unsigned integers                            |
| **Comments** | Lines starting with `#`                      |
| **Whitespace** | Spaces/tabs are ignored; newlines are significant |

---

## Commands

### `LOAD <filepath>`

Load a CSV dataset from `<filepath>`.

```
LOAD data/iris.csv
```

* Must be the first non-comment command in a script.
* The file must be a valid CSV with a header row.

---

### `TARGET <column>`

Set the target (label) column for classification.

```
TARGET species
```

* Must appear after `LOAD`.
* `<column>` must match an existing column name.

---

### `SPLIT <train_percent> <test_percent>`

Split the dataset into training and testing sets.

```
SPLIT 80 20
```

* Both values are integers (0–100).
* Must appear after `LOAD` and `TARGET`.
* Uses a fixed random seed (42) for reproducibility.
* Non-numeric target columns are automatically label-encoded.

---

### `MODEL <model_name>`

Select a single model for training.

```
MODEL RandomForest
```

Supported model names:

| Name                 | scikit-learn Class                |
|----------------------|-----------------------------------|
| `RandomForest`       | `RandomForestClassifier`          |
| `DecisionTree`       | `DecisionTreeClassifier`          |
| `KNN`                | `KNeighborsClassifier`            |
| `LogisticRegression` | `LogisticRegression`              |
| `SVM`                | `SVC`                             |

---

### `TRAIN`

Train the currently selected model on the training set.

```
TRAIN
```

* Requires a prior `MODEL` and `SPLIT`.

---

### `EVALUATE <metric>`

Evaluate trained model(s) on the test set.

```
EVALUATE accuracy
```

Supported metrics:

| Metric      | scikit-learn Function                   |
|-------------|------------------------------------------|
| `accuracy`  | `accuracy_score`                         |
| `precision` | `precision_score` (weighted average)     |
| `recall`    | `recall_score` (weighted average)        |
| `f1`        | `f1_score` (weighted average)            |

* If multiple models have been trained (via `COMPARE`), all are evaluated.
* Results are appended to the internal results list.

---

### `COMPARE` (multi-line block)

Train multiple models for comparison.

```
COMPARE
    RandomForest
    DecisionTree
    KNN
```

* Model names are listed on subsequent indented lines.
* Requires a prior `SPLIT`.
* Each listed model is trained automatically.

---

### `SHOW_BEST`

Print the model with the highest evaluation score.

```
SHOW_BEST
```

* Must appear after at least one `EVALUATE`.

---

### `SAVE_RESULTS <filepath>`

Save accumulated evaluation results to a CSV file.

```
SAVE_RESULTS outputs/results.csv
```

* Creates parent directories if they do not exist.
* Output columns: `model`, `metric`, `score`.

---

### `SHOW_DATASET_INFO`

Print summary statistics about the loaded dataset (shape, dtypes, null counts).

```
SHOW_DATASET_INFO
```

* Must appear after `LOAD`.

---

### `SHOW_HEAD <rows>`

Print the first `<rows>` rows of the dataset.

```
SHOW_HEAD 5
```

* Must appear after `LOAD`.

---

### `CROSS_VALIDATE <k>`

Perform k-fold cross-validation on the selected model.

```
CROSS_VALIDATE 5
```

* Must appear after `LOAD`, `TARGET`, and `MODEL`.
* If a data split (`SPLIT`) has been run, cross-validation is performed on the training split to keep the test split isolated.
* If no split has been run, cross-validation is performed on the full dataset (non-numeric target columns are automatically label-encoded).
* Records the mean score in the results list under the metric name `cv_<k>_fold_accuracy`.

---

## Comments

Lines starting with `#` are comments and are ignored during execution.

```
# This is a comment
LOAD data/iris.csv
```

---

## Complete Example

```
# Full experiment pipeline
LOAD data/iris.csv
TARGET species
SHOW_DATASET_INFO
SHOW_HEAD 5
SPLIT 80 20

COMPARE
    RandomForest
    DecisionTree
    KNN
    LogisticRegression
    SVM

EVALUATE accuracy
SHOW_BEST
SAVE_RESULTS outputs/full_comparison.csv
```

---

## Error Handling

| Error Type       | When                                          | Example Message                             |
|-----------------|-----------------------------------------------|---------------------------------------------|
| **Syntax Error** | Unknown command or invalid token              | `Unexpected token 'INVALID'`                |
| **Semantic Error** | Command used out of order                   | `Cannot set TARGET before LOAD`             |
| **Runtime Error** | File missing, unknown model/metric           | `Dataset not found: missing.csv`            |

All errors are reported with descriptive messages to help the user fix their
script.
