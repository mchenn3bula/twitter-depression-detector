# BERT text classification and keyword robustness

Does a text classifier learn broader language patterns, or rely on a conspicuous keyword? This project compares **BERT fine-tuning**, a **TF-IDF / logistic regression baseline**, and **frozen Transformer embeddings / Random Forest**, then measures their sensitivity to removing `depression` from the same held-out examples.

The original academic notebook has been refactored into a command-line pipeline with fixed data splits, validation-based model selection, paired robustness evaluation and offline regression tests. The models predict dataset labels; this is not a clinical assessment tool.

**Status:** the new pipeline is tested on synthetic data with a small randomly initialized local BERT. Full-dataset training and benchmark results for this implementation are pending because the original CSV is not included.

## What changed

| Original notebook issue | Current pipeline |
| --- | --- |
| A second BERT model reused the first model's optimizer | Fresh AdamW and scheduler per run; a regression test verifies the intended model's weights change |
| Supervised feature selection was fitted before splitting | Feature selection is inside a pipeline fitted only on training data |
| Random Forest tuning repeatedly evaluated test data | Validation macro F1 selects models; test evaluation is a separate command |
| Row-level splits could share users or duplicate posts | Connected user IDs and normalized/ablated duplicates stay in one partition |
| Keyword removal depended on the true label | The same transformation applies to every example, independent of its label |
| DistilBERT features flattened variable-length token sequences | Evaluation mode and attention-mask-aware mean pooling produce fixed-size features |

## Quick start

Python 3.12 is used for local verification. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux instead: source .venv/bin/activate

python -m pip install -r requirements-test.txt
python -m pip install -e . --no-deps
python -m pytest -q
keyword-study --help
```

The pinned test environment installs CPU PyTorch. For GPU training, create a separate environment, install the appropriate [official PyTorch build](https://pytorch.org/get-started/locally/), then run `python -m pip install -e ".[dev]"`. Every experiment records its runtime package versions. Tests download no datasets or pretrained weights.

## Prepare the data once

Place the original `Mental-Health-Twitter.csv` under `data/`. Required columns are `post_text` (nonempty text), `label` (0 or 1), and `user_id` (author ID).

```bash
keyword-study prepare --csv data/Mental-Health-Twitter.csv --output data/prepared --seed 42
```

The nominal split is **60% train / 20% validation / 20% test**, preserving connected groups. Actual sizes and class counts are recorded in `manifest.json` with source/split hashes. Large groups can change proportions. Missing values, invalid labels and unusable splits fail explicitly.

Column names are configurable. Without author IDs, explicitly pass `--group-column none`; duplicate grouping still applies, but author separation cannot be guaranteed. Conservative cleaning preserves negations and punctuation, so this protocol does **not** reproduce the old notebook's preprocessing or split.

## Train and select on validation

Start with the inexpensive baseline:

```bash
keyword-study train --data data/prepared --output runs/tfidf-original --method tfidf
```

Fine-tune BERT under two matched training conditions:

```bash
keyword-study train --data data/prepared --output runs/bert-original --method bert --variant original --epochs 3
keyword-study train --data data/prepared --output runs/bert-removed --method bert --variant removed --epochs 3
```

Both runs use the same prepared examples. Each initializes a new model, optimizer and scheduler. Keyword removal applies to training and validation examples of **both labels**. Run folders cannot overwrite existing experiments.

Defaults: `bert-base-uncased`, batch size 8, maximum length 128, learning rate 2e-5. Choose `--device cpu`, `--device cuda`, or automatic selection. `--model` accepts a model ID or local directory; `--revision` can pin a Hugging Face commit. First use of a remote model downloads its weights.

The frozen-feature comparison uses `distilbert-base-uncased` by default:

```bash
keyword-study train --data data/prepared --output runs/distilbert-rf --method frozen-rf
```

TF-IDF compares C values 0.1, 1 and 10. Random Forest compares depths 10 and unrestricted with 200 trees. Feature selection, vocabulary fitting and classifier fitting use training data only. Validation macro F1 selects candidates and BERT epochs, retaining the first best on ties. Training does not read the test file.

## Evaluate the held-out examples

Freeze experiment choices before running:

```bash
keyword-study evaluate --data data/prepared --run runs/bert-original
keyword-study evaluate --data data/prepared --run runs/bert-removed
```

Each saved model is evaluated on the original and keyword-removed versions of the **same test examples**. Reports include accuracy, macro F1, per-class metrics, confusion matrices, prediction flip rate and metrics for the keyword-affected subset. The command refuses to overwrite an existing test report.

| Artifact | Purpose |
| --- | --- |
| `config.json`, `data_manifest.json` | Parameters, fingerprints, source commit and runtime versions |
| `validation_history.json` | Candidate / epoch selection evidence |
| `model/` or `pipeline.joblib` | Selected local artifacts |
| `test_metrics.json` | Paired holdout evaluation |
| `test_predictions.csv` | Row indices, labels and predictions; no post text or author IDs |

Do not use test reports to choose later settings. Plan repeated seeds on the **same prepared split** using `train --seed`, select settings on validation and report every prespecified run. Load Joblib artifacts only from your own trusted runs.

## Results and evidence

| Evidence | Interpretation |
| --- | --- |
| Original notebook: **84.95% accuracy on 6,000 test examples**, approximately 0.85 macro F1 | Historical initial BERT output under the old 70/30 row split; not a result from this pipeline |
| Offline synthetic-data tests with a small local BERT | Splitting, optimization, selection, checkpoint reload and evaluation checks; not real-world performance evidence |
| New full-dataset evaluation | **Pending**; no replacement accuracy is claimed |

The [original notebook](NLP_PROJECT%20%282%29.ipynb) and [academic report](NLP_Final_Project.pdf) remain historical artifacts, including their original defects and outputs. Use the package for new experiments. See the [evaluation protocol](docs/evaluation.md) and [validation record](docs/validation.md).

## Repository map

```text
src/keyword_study/data.py    Validation, transformations, grouped splitting and hashes
src/keyword_study/models.py Fitting, pooling, optimization and metrics
src/keyword_study/cli.py    Prepare / train / evaluate entry points
tests/                     Offline correctness and end-to-end tests
docs/                      Protocol and verification evidence
```

Data, environments, weights and run outputs are ignored by Git. The original dataset's provenance, permissions and labeling process still need documentation before a new public benchmark claim. No dataset or trained checkpoint is redistributed here.
