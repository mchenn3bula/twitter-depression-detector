# Local verification record

Verified on 21 September 2026 with Python 3.12.14 on Windows, CPU PyTorch 2.14.0, Transformers 5.17.0 and scikit-learn 1.9.1. Exact dependency pins are in `requirements-test.txt`.

## Checks performed

- **18 tests passed** in the final local run (8.80 seconds of pytest execution).
- Ruff lint passed; source and tests were formatted with Ruff.
- `pip check` reported no broken requirements.
- The installed `keyword-study --help` entry point ran successfully.
- Tests used `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

The Windows account's default pytest temporary directory was inaccessible, so the local run used a fresh directory under ignored `runs/`:

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
python -m pytest -q --basetemp=runs/pytest-local-03
```

For normal environments, `python -m pytest -q` is sufficient. When specifying a base directory, use a dedicated temporary directory: pytest manages its contents.

## What the tests establish

| Area | Evidence |
| --- | --- |
| Data splitting | Deterministic manifests; disjoint row/group membership; both classes present; changed CSV rejected |
| Duplicate prevention | Transitive user links and original/keyword-removed duplicate text share a component |
| Input checks | Invalid labels, blank texts and missing users rejected; no-user mode must be explicit |
| Optimizer ownership | An optimizer from another model is rejected; fresh optimizer updates the intended second BERT model without changing the first |
| Fitting boundaries | Validation-only tokens stay out of TF-IDF vocabulary; supervised selector receives only training examples/labels |
| Padding | Mean pooling ignores masked token positions |
| End-to-end | TF-IDF, tiny BERT fine-tuning and tiny DistilBERT frozen features each run under original/removed text conditions |
| Holdout isolation | Training succeeds while the test CSV is absent; evaluation rejects a changed manifest |
| Saved artifacts | Checkpoints/pipelines reload for paired test evaluation; existing runs and test reports cannot be overwritten |

End-to-end tests generate 120 synthetic texts across 60 synthetic users. Transformer tests create small randomly initialized models and local vocabularies. They do not download pretrained checkpoints.

## Not established

No training was performed on `Mental-Health-Twitter.csv`, which is absent from the repository. No new benchmark, convergence result, confidence interval, GPU performance claim or clinical validity claim is made. The old notebook's 84.95% initial BERT accuracy remains historical evidence under a different protocol.

GitHub Actions is configured to repeat the offline lint/tests on Ubuntu. Its current status is available on the repository's Actions page; the Windows result above is the local verification record.
