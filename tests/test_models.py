import numpy as np
import pytest
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from keyword_study.models import (
    build_optimizer,
    forest_pipeline,
    make_loader,
    masked_mean,
    select_pipeline,
    tfidf_pipeline,
    train_epoch,
)


def test_fresh_optimizer_updates_second_model_and_rejects_old_optimizer(tiny_bert):
    first = AutoModelForSequenceClassification.from_pretrained(tiny_bert)
    second = AutoModelForSequenceClassification.from_pretrained(tiny_bert)
    tokenizer = AutoTokenizer.from_pretrained(tiny_bert)
    loader = make_loader(["happy", "tired depression"], [0, 1], tokenizer, 2, 16, 42)
    wrong, wrong_schedule = build_optimizer(first, 0.01, 1)
    with pytest.raises(ValueError, match="does not own"):
        train_epoch(second, loader, wrong, wrong_schedule, torch.device("cpu"))
    before = {k: v.clone() for k, v in second.state_dict().items()}
    optimizer, scheduler = build_optimizer(second, 0.01, 1)
    train_epoch(second, loader, optimizer, scheduler, torch.device("cpu"))
    assert any(not torch.equal(before[k], v) for k, v in second.state_dict().items())
    assert all(torch.equal(before[k], v) for k, v in first.state_dict().items())


def test_masked_mean_ignores_padding():
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [900.0, -900.0]]])
    assert torch.equal(masked_mean(hidden, torch.tensor([[1, 1, 0]])), torch.tensor([[2.0, 3.0]]))


def test_vectorizer_never_fits_validation_vocabulary():
    model, _ = select_pipeline(
        [({}, tfidf_pipeline(1, 42))],
        ["calm happy", "sad tired"],
        [0, 1],
        ["validationonly happy", "validationonly sad"],
        [0, 1],
    )
    assert "validationonly" not in model.named_steps["tfidf"].vocabulary_


def test_supervised_selector_sees_only_training_labels(monkeypatch):
    from sklearn.ensemble import ExtraTreesClassifier

    observed = []
    original = ExtraTreesClassifier.fit

    def tracked(self, x, y, *args, **kwargs):
        observed.append((len(x), list(y)))
        return original(self, x, y, *args, **kwargs)

    monkeypatch.setattr(ExtraTreesClassifier, "fit", tracked)
    rng = np.random.default_rng(42)
    train_x, val_x = rng.normal(size=(20, 8)), rng.normal(size=(6, 8))
    labels = [0, 1] * 10
    select_pipeline([({}, forest_pipeline(10, 42))], train_x, labels, val_x, [0, 1] * 3)
    assert observed == [(20, labels)]
