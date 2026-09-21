# Evaluation protocol

## Research question

Measure how dataset-label predictions change when a predefined keyword is removed. Keyword sensitivity is a robustness observation, not a causal estimate of mental health.

## Before training

1. Record dataset provenance, labeling process, permitted use and acquisition date; the legacy repository does not establish these facts.
2. Validate binary labels, nonempty texts and author IDs. Replace URLs/mentions, normalize whitespace and retain negations and punctuation.
3. Connect rows sharing an author or case-insensitive normalized text. Also connect original/keyword-removed duplicate texts so an ablation cannot introduce exact cross-partition duplicates. Near duplicates are not detected.
4. Split connected groups with seeded StratifiedGroupKFold: one of five folds is test, one of four development folds is validation. Nominal proportions are 60/20/20. Each partition must contain both classes; inspect group and class counts before training.
5. Freeze the keyword and split. The default keyword comes from the historical study; do not search test data for new keywords.

Grouping uses text and author information before fitting; stratification uses labels for partition assignment. No learned feature transform is fitted on validation or test data. Prepared group IDs are local component numbers, not author IDs.

## Matched experiments

Use the same prepared split for original and keyword-removed runs. Apply removal to both labels. Each run starts from a fresh model and explicit seed; exact numerical equality across hardware/library versions is not promised.

Select BERT epochs or classifier hyperparameters by validation macro F1. Feature selection stays inside the train-fitted pipeline. Frozen encoders run in evaluation mode; attention-mask-aware pooling excludes padding. Ties retain the first candidate/epoch. The selected model is not refitted on validation.

Training never opens the test file. A test verifies training succeeds with that file temporarily absent.

## Final evaluation

Evaluate both versions of test text using the same labels and order. Report both conditions, affected-subset metrics and prediction flip rate. Confusion matrices use label order `[0, 1]`. Test reports cannot be overwritten by the CLI.

Do not tune based on final reports. Preventing overwrite protects artifacts; it cannot prevent a researcher from reusing holdout data. Declare repeated seeds in advance and report all runs. The package records individual runs but does not yet compute multi-seed confidence intervals.

## Remaining limitations

- Original dataset absent; provenance and full-scale performance unverified.
- Grouping does not detect semantic near duplicates or unknown authors.
- Keyword removal can alter meaning; changed predictions do not prove one causal mechanism.
- Label semantics and annotation quality limit interpretation.
- Historical scores use different preprocessing/splits and are not directly comparable.
- Offline tests validate behavior, not convergence, model quality or GPU throughput.

## Implementation references

- [scikit-learn StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)
- [Hugging Face model loading and saving](https://huggingface.co/docs/transformers/main_classes/model)
- [PyTorch AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
