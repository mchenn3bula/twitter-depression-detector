"""Validate data and split connected users/duplicate texts before any model fitting."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_text(text):
    """Conservative normalization: keep negations, punctuation and word order for BERT."""
    text = re.sub(r"https?://\S+|www\.\S+", " [URL] ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\w)@\w+", " [USER] ", text)
    return " ".join(text.split())


def remove_keyword(text, keyword="depression"):
    return " ".join(
        re.sub(r"(?<!\w)" + re.escape(keyword) + r"(?!\w)", "", text, flags=re.IGNORECASE).split()
    )


def transform_texts(texts, variant, keyword):
    if variant not in {"original", "removed"}:
        raise ValueError("Unknown text variant")
    return [remove_keyword(t, keyword) if variant == "removed" else t for t in texts]


def connected_groups(texts, users, keyword):
    """Union user IDs and duplicate normalized/ablated texts, including transitive links."""
    parent = list(range(len(texts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, text in enumerate(texts):
        keys = [("text", text.casefold()), ("text", remove_keyword(text, keyword).casefold())]
        if users is not None:
            keys.append(("user", str(users[i])))
        for key in keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    return np.array([find(i) for i in range(len(texts))])


def prepare(
    csv_path,
    output,
    text_column="post_text",
    label_column="label",
    group_column="user_id",
    seed=42,
    keyword="depression",
):
    if not keyword.strip():
        raise ValueError("Keyword must not be empty")
    frame = pd.read_csv(csv_path, dtype={group_column: "string"} if group_column else None)
    required = [text_column, label_column] + ([group_column] if group_column else [])
    if any(c not in frame for c in required):
        raise ValueError(
            f"Required columns: {required}. Use --group-column none only if user IDs are unavailable."
        )
    if frame[required].isna().any().any():
        raise ValueError("Missing text, label or user ID; correct the source data first")
    if not frame[label_column].isin([0, 1]).all() or set(frame[label_column]) != {0, 1}:
        raise ValueError("Labels must contain both integer classes 0 and 1")
    if not frame[text_column].map(lambda x: isinstance(x, str)).all():
        raise ValueError("All text values must be strings")
    texts = frame[text_column].map(clean_text).tolist()
    if any(not text for text in texts):
        raise ValueError("Empty normalized text is not supported")
    users = frame[group_column].tolist() if group_column else None
    groups = connected_groups(texts, users, keyword)
    labels = frame[label_column].astype(int).to_numpy()
    if len(np.unique(groups)) < 5 or min(np.bincount(labels)) < 5:
        raise ValueError("Need at least five independent groups and five examples per class")
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    development, test = next(outer.split(texts, labels, groups))
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    train_rel, val_rel = next(inner.split(development, labels[development], groups[development]))
    indices = {"train": development[train_rel], "validation": development[val_rel], "test": test}
    for name, ids in indices.items():
        if set(labels[ids]) != {0, 1}:
            raise ValueError(
                f"{name} lacks a class; group structure cannot support this split/seed"
            )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "source_sha256": file_hash(csv_path),
        "seed": seed,
        "keyword": keyword,
        "text_column": text_column,
        "label_column": label_column,
        "group_column": group_column,
        "grouping": "user + normalized/keyword-removed text connected components",
        "target_fractions": {"train": 0.6, "validation": 0.2, "test": 0.2},
        "splits": {},
    }
    for name, ids in indices.items():
        part = pd.DataFrame(
            {
                "row_id": ids,
                "group_id": groups[ids],
                "text": [texts[i] for i in ids],
                "label": labels[ids],
            }
        )
        path = output / f"{name}.csv"
        part.to_csv(path, index=False)
        manifest["splits"][name] = {
            "rows": len(part),
            "groups": int(part.group_id.nunique()),
            "class_counts": {str(k): int(v) for k, v in part.label.value_counts().items()},
            "sha256": file_hash(path),
        }
    write_json(output / "manifest.json", manifest)
    return manifest


def load_split(directory, name):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    path = directory / f"{name}.csv"
    if file_hash(path) != manifest["splits"][name]["sha256"]:
        raise ValueError(f"{name} data changed after preparation")
    return pd.read_csv(path, keep_default_na=False), manifest
