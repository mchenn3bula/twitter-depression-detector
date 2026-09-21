"""Prepare once, train using validation only, then explicitly evaluate the holdout."""

import argparse
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from .data import file_hash, load_split, prepare, transform_texts, write_json
from .models import (
    build_optimizer,
    embed_texts,
    forest_pipeline,
    make_loader,
    metrics,
    predict_bert,
    seed_all,
    select_pipeline,
    tfidf_pipeline,
    train_epoch,
)


def runtime_info():
    try:
        checkout = Path(__file__).resolve().parents[2]
        if not (checkout / ".git").exists():
            raise OSError("Package is not running from its source checkout")
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=checkout, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=checkout,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": revision,
        "git_dirty": dirty,
        "packages": {
            p: importlib.metadata.version(p)
            for p in ["torch", "transformers", "scikit-learn", "pandas", "numpy"]
        },
    }


def device_for(request):
    if request == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if request == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable in this PyTorch installation")
    return torch.device(request)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 4 or args.lr <= 0:
        raise ValueError("Positive epochs, batch size and learning rate; max length >= 4 required")
    seed_all(args.seed)
    train_frame, manifest = load_split(args.data, "train")
    val_frame, _ = load_split(args.data, "validation")
    # Never load the test split during model selection.
    train_x = transform_texts(train_frame.text.tolist(), args.variant, manifest["keyword"])
    val_x = transform_texts(val_frame.text.tolist(), args.variant, manifest["keyword"])
    train_y, val_y = train_frame.label.to_numpy(), val_frame.label.to_numpy()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = {k: v for k, v in vars(args).items() if k != "command"}
    config.update(
        {
            "manifest_sha256": file_hash(Path(args.data) / "manifest.json"),
            "keyword": manifest["keyword"],
            "runtime": runtime_info(),
        }
    )
    write_json(output / "config.json", config)
    write_json(output / "data_manifest.json", manifest)
    device = device_for(args.device)
    if args.method == "tfidf":
        candidates = [({"C": c}, tfidf_pipeline(c, args.seed)) for c in [0.1, 1.0, 10.0]]
        best, history = select_pipeline(candidates, train_x, train_y, val_x, val_y)
        joblib.dump(best, output / "pipeline.joblib")
    else:
        model_name = args.model or (
            "bert-base-uncased" if args.method == "bert" else "distilbert-base-uncased"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=args.revision)
        model_class = AutoModelForSequenceClassification if args.method == "bert" else AutoModel
        model_kwargs = {"num_labels": 2} if args.method == "bert" else {}
        model = model_class.from_pretrained(model_name, revision=args.revision, **model_kwargs).to(
            device
        )
        config.update(
            {
                "model": model_name,
                "resolved_model_revision": getattr(model.config, "_commit_hash", None),
                "resolved_device": str(device),
            }
        )
        write_json(output / "config.json", config)
        tokenizer.save_pretrained(output / "model")
        if args.method == "bert":
            train_loader = make_loader(
                train_x,
                train_y,
                tokenizer,
                args.batch_size,
                args.max_length,
                args.seed,
                shuffle=True,
            )
            val_loader = make_loader(
                val_x, val_y, tokenizer, args.batch_size, args.max_length, args.seed
            )
            optimizer, scheduler = build_optimizer(model, args.lr, len(train_loader) * args.epochs)
            history, best_score = [], -1.0
            for epoch in range(args.epochs):
                loss = train_epoch(model, train_loader, optimizer, scheduler, device)
                scores = metrics(val_y, predict_bert(model, val_loader, device))
                history.append({"epoch": epoch + 1, "train_loss": loss, "validation": scores})
                if scores["macro_f1"] > best_score:
                    best_score = scores["macro_f1"]
                    model.save_pretrained(output / "model", safe_serialization=True)
                    write_json(
                        output / "selection.json",
                        {"epoch": epoch + 1, "validation_macro_f1": best_score},
                    )
                print(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "train_loss": loss,
                            "validation_macro_f1": scores["macro_f1"],
                        }
                    ),
                    flush=True,
                )
        else:
            train_features = embed_texts(
                model, tokenizer, train_x, device, args.batch_size, args.max_length
            )
            val_features = embed_texts(
                model, tokenizer, val_x, device, args.batch_size, args.max_length
            )
            candidates = [({"max_depth": d}, forest_pipeline(d, args.seed)) for d in [10, None]]
            best, history = select_pipeline(
                candidates, train_features, train_y, val_features, val_y
            )
            joblib.dump(best, output / "pipeline.joblib")
            # Store frozen encoder locally so evaluation uses the identical weights.
            model.save_pretrained(output / "model", safe_serialization=True)
    write_json(output / "validation_history.json", history)
    write_json(output / "complete.json", {"status": "training_complete"})
    print(f"Training complete: {output}. Test data was not used.")


def evaluate(args):
    run = Path(args.run)
    if not (run / "complete.json").exists():
        raise ValueError("Training did not complete")
    if (run / "test_metrics.json").exists():
        raise ValueError("Test metrics already exist; preserve this evaluation artifact")
    config = read_json(run / "config.json")
    if file_hash(Path(args.data) / "manifest.json") != config["manifest_sha256"]:
        raise ValueError("Evaluation data manifest differs from the training manifest")
    test, _ = load_split(args.data, "test")
    seed_all(config["seed"])
    device = device_for(args.device)
    original = test.text.tolist()
    removed = transform_texts(original, "removed", config["keyword"])
    predictions = {}
    if config["method"] == "tfidf":
        pipeline = joblib.load(run / "pipeline.joblib")
        predict = pipeline.predict
    else:
        tokenizer = AutoTokenizer.from_pretrained(run / "model", local_files_only=True)
        model_class = (
            AutoModelForSequenceClassification if config["method"] == "bert" else AutoModel
        )
        model = model_class.from_pretrained(run / "model", local_files_only=True).to(device)
        if config["method"] == "bert":

            def predict(texts):
                loader = make_loader(
                    texts,
                    test.label,
                    tokenizer,
                    config["batch_size"],
                    config["max_length"],
                    config["seed"],
                )
                return predict_bert(model, loader, device)
        else:
            pipeline = joblib.load(run / "pipeline.joblib")

            def predict(texts):
                features = embed_texts(
                    model, tokenizer, texts, device, config["batch_size"], config["max_length"]
                )
                return pipeline.predict(features)

    for name, texts in [("original", original), ("removed", removed)]:
        predictions[name] = predict(texts)
    affected = np.array([a != b for a, b in zip(original, removed)])
    scores = {name: metrics(test.label, pred) for name, pred in predictions.items()}
    scores["keyword_affected_examples"] = int(affected.sum())
    scores["prediction_flip_rate"] = float(
        np.mean(predictions["original"] != predictions["removed"])
    )
    scores["affected_subset"] = (
        {
            name: metrics(test.label.to_numpy()[affected], pred[affected])
            for name, pred in predictions.items()
        }
        if affected.any()
        else None
    )
    scores["training_variant"] = config["variant"]
    scores["runtime"] = runtime_info()
    # No raw text or user IDs in prediction artifacts.
    rows = test[["row_id", "label"]].copy()
    rows["prediction_original"] = predictions["original"]
    rows["prediction_removed"] = predictions["removed"]
    rows.to_csv(run / "test_predictions.csv", index=False)
    write_json(run / "test_metrics.json", scores)
    print(
        json.dumps({name: scores[name]["macro_f1"] for name in ["original", "removed"]}, indent=2)
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser(
        "prepare", help="Create fixed, disjoint train/validation/test groups"
    )
    prep.add_argument("--csv", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--text-column", default="post_text")
    prep.add_argument("--label-column", default="label")
    prep.add_argument(
        "--group-column", default="user_id", help="Use 'none' only when user IDs are unavailable"
    )
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--keyword", default="depression")
    fit = commands.add_parser("train", help="Select models on validation, never test")
    fit.add_argument("--data", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--method", choices=["tfidf", "bert", "frozen-rf"], default="bert")
    fit.add_argument("--variant", choices=["original", "removed"], default="original")
    fit.add_argument("--model", help="Hugging Face model ID or local model directory")
    fit.add_argument(
        "--revision", default="main", help="Prefer an immutable model commit for published runs"
    )
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--epochs", type=int, default=3)
    fit.add_argument("--batch-size", type=int, default=8)
    fit.add_argument("--max-length", type=int, default=128)
    fit.add_argument("--lr", type=float, default=2e-5)
    fit.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    test = commands.add_parser(
        "evaluate", help="Evaluate a selected model on both paired test conditions"
    )
    test.add_argument("--data", required=True)
    test.add_argument("--run", required=True)
    test.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            manifest = prepare(
                args.csv,
                args.output,
                args.text_column,
                args.label_column,
                None if args.group_column == "none" else args.group_column,
                args.seed,
                args.keyword,
            )
            print(json.dumps(manifest["splits"], indent=2))
        elif args.command == "train":
            train(args)
        else:
            evaluate(args)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
