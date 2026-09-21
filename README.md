# BERT text classification and keyword-shortcut analysis

An academic NLP project comparing fine-tuned BERT with a feature-based classification approach on a labeled Twitter dataset. A central question is whether predictions depend on an explicit keyword rather than broader language patterns.

The repository name reflects the original assignment. The models predict **dataset labels**; the experiment does not establish a person's mental-health condition or clinical validity.

## Start here

- [Experiment notebook](NLP_PROJECT%20%282%29.ipynb): preprocessing, BERT fine-tuning, saved outputs and feature-based experiments.
- [Project report](NLP_Final_Project.pdf): original academic write-up.

## Methods explored

- Text cleaning and tokenization with Hugging Face Transformers.
- Fine-tuning `bert-base-uncased` for binary classification.
- Removing the word `depression` from selected positive-label examples to investigate keyword dependence.
- DistilBERT features, feature selection, Random Forest classification and Ray Tune experiments.

## Recorded BERT result

The original notebook records **84.95% test accuracy on 6,000 examples** for the initial BERT experiment. The code uses a 70/30 random split with `random_state=42`; the saved classification report gives approximately 0.85 macro F1.

This is a historical notebook result, not a newly reproduced benchmark or an independent external evaluation.

## Reproduction and evaluation limits

The notebook expects `Mental-Health-Twitter.csv`, which is not included here. Obtain the original dataset with its provenance and permitted use, then review paths and library versions before running. The environment is not pinned and some APIs reflect the original experiment period.

Two follow-up checks are especially important:

- The keyword-removal branch constructs a new model but reuses existing optimizer and scheduler variables. That training setup needs correction and rerunning before comparing its result with the original model.
- The feature-selection / tuning branch needs a clean train/validation/test separation. Feature selection and hyperparameter choices should be fitted using training/validation data only.

## Next steps

Separate the experiments into reproducible scripts, rebuild each optimizer for its own model, evaluate keyword robustness on a fixed untouched holdout and report uncertainty across repeated runs.
