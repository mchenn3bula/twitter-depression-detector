import pandas as pd
import pytest
import torch
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertModel,
    DistilBertTokenizer,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    # Tiny models are faster with fewer threads and need no model downloads.
    torch.set_num_threads(1)


@pytest.fixture
def source_csv(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "post_text": f"{'calm happy' if label == 0 else 'tired depression'} sample {i} entry {j}",
                "label": label,
                "user_id": f"user-{i}",
            }
            for i in range(60)
            for j in range(2)
            for label in [i % 2]
        ]
    )
    path = tmp_path / "source.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def tiny_bert(tmp_path):
    path = tmp_path / "tiny-bert"
    path.mkdir()
    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "calm",
        "happy",
        "tired",
        "depression",
        "sample",
        "entry",
    ] + list("0123456789")
    (path / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    tokenizer = BertTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.save_pretrained(path)
    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        num_labels=2,
    )
    BertForSequenceClassification(config).save_pretrained(path)
    return path


@pytest.fixture
def tiny_distilbert(tmp_path):
    path = tmp_path / "tiny-distilbert"
    path.mkdir()
    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "calm",
        "happy",
        "tired",
        "depression",
        "sample",
        "entry",
    ] + list("0123456789")
    (path / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    tokenizer = DistilBertTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.save_pretrained(path)
    config = DistilBertConfig(
        vocab_size=len(tokenizer),
        dim=16,
        n_layers=1,
        n_heads=2,
        hidden_dim=32,
        max_position_embeddings=64,
        dropout=0.0,
        attention_dropout=0.0,
    )
    DistilBertModel(config).save_pretrained(path)
    return path
