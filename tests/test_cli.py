import json

import pytest

from keyword_study.cli import main


@pytest.mark.parametrize("method", ["tfidf", "bert", "frozen-rf"])
@pytest.mark.parametrize("variant", ["original", "removed"])
def test_offline_end_to_end(source_csv, tiny_bert, tiny_distilbert, tmp_path, method, variant):
    data, run = tmp_path / "prepared", tmp_path / "run"
    main(["prepare", "--csv", str(source_csv), "--output", str(data)])
    # Training must succeed even when the test file is unavailable.
    test_bytes = (data / "test.csv").read_bytes()
    (data / "test.csv").unlink()
    main(
        [
            "train",
            "--data",
            str(data),
            "--output",
            str(run),
            "--method",
            method,
            "--variant",
            variant,
            "--model",
            str(tiny_distilbert if method == "frozen-rf" else tiny_bert),
            "--epochs",
            "1",
            "--batch-size",
            "16",
            "--max-length",
            "24",
            "--device",
            "cpu",
        ]
    )
    assert not (run / "test_metrics.json").exists()
    (data / "test.csv").write_bytes(test_bytes)
    main(["evaluate", "--data", str(data), "--run", str(run), "--device", "cpu"])
    scores = json.loads((run / "test_metrics.json").read_text())
    assert scores["original"]["n"] == scores["removed"]["n"] > 0
    assert scores["keyword_affected_examples"] > 0
    assert 0 <= scores["prediction_flip_rate"] <= 1
    assert scores["training_variant"] == variant
    with pytest.raises(SystemExit):
        main(["evaluate", "--data", str(data), "--run", str(run)])
    with pytest.raises(SystemExit):
        main(["train", "--data", str(data), "--output", str(run), "--method", "tfidf"])


def test_evaluation_rejects_changed_manifest(source_csv, tmp_path):
    data, run = tmp_path / "prepared", tmp_path / "run"
    main(["prepare", "--csv", str(source_csv), "--output", str(data)])
    main(["train", "--data", str(data), "--output", str(run), "--method", "tfidf"])
    manifest = data / "manifest.json"
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(SystemExit):
        main(["evaluate", "--data", str(data), "--run", str(run)])
