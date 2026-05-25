"""Evaluate a trained RF IoT classifier on a labeled signal directory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import BATCH_SIZE, SEED, WINDOW_SIZE
from dataset import build_dataset
from evaluation import (
    save_confusion_matrix_plot,
    write_classification_report,
)
from infer import load_classes
from preprocessing import class_distribution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/best_iot_classifier.h5"))
    parser.add_argument("--metadata", type=Path, default=Path("models/metadata.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/evaluation"))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--file-template",
        help="Optional file pattern such as '{class_name}_test.npy'. Defaults to auto-detecting raw and test names.",
    )
    parser.add_argument("--balance", action="store_true", help="Balance class windows before evaluation.")
    parser.add_argument("--allow-missing", action="store_true", help="Skip classes with no matching .npy file.")
    return parser.parse_args()


def write_predictions_csv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray,
    classes: list[str],
    output_path: Path,
) -> None:
    """Write one row per evaluated window."""
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "index",
                "true_label",
                "predicted_label",
                "confidence",
                "correct",
            ],
        )
        writer.writeheader()
        for idx, (true_idx, pred_idx, confidence) in enumerate(zip(y_true, y_pred, confidences)):
            writer.writerow(
                {
                    "index": idx,
                    "true_label": classes[int(true_idx)],
                    "predicted_label": classes[int(pred_idx)],
                    "confidence": f"{float(confidence):.8f}",
                    "correct": int(true_idx == pred_idx),
                }
            )


def run_evaluation(
    model_path: Path,
    metadata_path: Path,
    data_dir: Path,
    output_dir: Path,
    batch_size: int = BATCH_SIZE,
    window_size: int = WINDOW_SIZE,
    seed: int = SEED,
    file_template: str | None = None,
    balance: bool = False,
    allow_missing: bool = False,
) -> dict[str, object]:
    """Evaluate model performance on labeled files and save reports."""
    classes = load_classes(metadata_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    x, y = build_dataset(
        data_dir,
        classes=classes,
        window_size=window_size,
        balance=balance,
        seed=seed,
        file_template=file_template,
        allow_missing=allow_missing,
    )

    model = tf.keras.models.load_model(str(model_path))
    y_cat = tf.keras.utils.to_categorical(y, len(classes))

    loss, accuracy = model.evaluate(x, y_cat, batch_size=batch_size, verbose=0)
    probabilities = model.predict(x, batch_size=batch_size, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)

    report = write_classification_report(
        y,
        y_pred,
        classes,
        output_dir / "classification_report.txt",
    )
    save_confusion_matrix_plot(
        y,
        y_pred,
        classes,
        output_dir / "confusion_matrix.png",
    )
    write_predictions_csv(
        y,
        y_pred,
        confidences,
        classes,
        output_dir / "predictions.csv",
    )

    metrics = {
        "data_dir": str(data_dir),
        "model": str(model_path),
        "samples": int(len(y)),
        "loss": float(loss),
        "accuracy": float(accuracy),
        "class_distribution": {
            classes[label]: count for label, count in class_distribution(y).items()
        },
        "balanced": bool(balance),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print(f"Evaluated {len(y)} windows from {data_dir}")
    print(f"Accuracy: {accuracy:.4f}")
    print(report)
    print(f"Saved reports to: {output_dir}")

    return metrics


def main() -> None:
    args = parse_args()
    run_evaluation(
        model_path=args.model,
        metadata_path=args.metadata,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        window_size=args.window_size,
        seed=args.seed,
        file_template=args.file_template,
        balance=args.balance,
        allow_missing=args.allow_missing,
    )


if __name__ == "__main__":
    main()
