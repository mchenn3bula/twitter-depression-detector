"""Small explicit training utilities shared by the CLI and offline tests."""

import random

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def metrics(labels, predictions):
    return {
        "n": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, labels=[0, 1], average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=[0, 1],
            target_names=["label_0", "label_1"],
            output_dict=True,
            zero_division=0,
        ),
    }


def tfidf_pipeline(c, seed):
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=30000)),
            ("classifier", LogisticRegression(C=c, max_iter=1000, random_state=seed)),
        ]
    )


def forest_pipeline(depth, seed):
    # Both supervised selection and classification see only training examples at fit time.
    return Pipeline(
        [
            (
                "selection",
                SelectFromModel(
                    ExtraTreesClassifier(n_estimators=100, random_state=seed, n_jobs=1),
                    threshold="median",
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200, max_depth=depth, random_state=seed, n_jobs=1
                ),
            ),
        ]
    )


def select_pipeline(candidates, train_x, train_y, val_x, val_y):
    best, best_score, history = None, -1.0, []
    for parameters, pipeline in candidates:
        pipeline.fit(train_x, train_y)
        scores = metrics(val_y, pipeline.predict(val_x))
        history.append({"parameters": parameters, "validation": scores})
        if scores["macro_f1"] > best_score:
            best, best_score = pipeline, scores["macro_f1"]
    return best, history


class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts, self.labels = list(texts), list(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index], int(self.labels[index])


def make_loader(texts, labels, tokenizer, batch_size, max_length, seed, shuffle=False):
    def collate(rows):
        encoded = tokenizer(
            [row[0] for row in rows],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([row[1] for row in rows], dtype=torch.long)
        return encoded

    return DataLoader(
        TextDataset(texts, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(seed),
    )


def build_optimizer(model, lr, total_steps):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    return optimizer, scheduler


def train_epoch(model, loader, optimizer, scheduler, device):
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    owned = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if not expected <= owned:
        raise ValueError("Optimizer does not own this model's trainable parameters")
    model.train()
    total_loss, count = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        if not torch.isfinite(output.loss):
            raise ValueError("Non-finite training loss")
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        size = batch["labels"].shape[0]
        total_loss += output.loss.item() * size
        count += size
    return total_loss / count


@torch.inference_mode()
def predict_bert(model, loader, device):
    model.eval()
    predictions = []
    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        predictions.extend(model(**inputs).logits.argmax(dim=-1).cpu().tolist())
    return np.array(predictions)


def masked_mean(hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


@torch.inference_mode()
def embed_texts(model, tokenizer, texts, device, batch_size, max_length):
    model.eval()
    batches = []
    for start in range(0, len(texts), batch_size):
        inputs = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        hidden = model(**inputs).last_hidden_state
        batches.append(masked_mean(hidden, inputs["attention_mask"]).cpu().numpy())
    return np.concatenate(batches)
