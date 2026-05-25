"""Test a trained RF IoT classifier on external labeled test files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import WINDOW_SIZE
from dataset import iter_class_signal_paths
from evaluation import (
    save_confusion_matrix_plot,
    write_classification_report,
)
from infer import load_classes, predict_signal


def default_test_data_dir() -> Path:
    """Prefer data/tests, but support the existing data/test folder."""
    for candidate in (Path("data/tests"), Path("data/test")):
        if candidate.exists():
            return candidate
    return Path("data/tests")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/best_iot_classifier.h5"))
    parser.add_argument("--metadata", type=Path, default=Path("models/metadata.json"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/test"))
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--step", type=int, default=1024)
    parser.add_argument(
        "--file-template",
        help="Optional file pattern such as '{class_name}_test.npy'. Defaults to auto-detecting raw and test names.",
    )
    parser.add_argument(
        "--require-all-classes",
        action="store_true",
        help="Fail if any class is missing from the test directory.",
    )
    return parser.parse_args()


def write_file_predictions(
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    """Write one row per test signal file."""
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file",
                "true_label",
                "predicted_label",
                "confidence",
                "windows",
                "correct",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_test(
    model_path: Path,
    metadata_path: Path,
    data_dir: Path,
    output_dir: Path,
    window_size: int = WINDOW_SIZE,
    step: int = 1024,
    file_template: str | None = None,
    allow_missing: bool = True,
) -> dict[str, object]:
    """Run file-level sliding-window ensemble tests and save reports."""
    classes = load_classes(metadata_path)
    signal_paths = iter_class_signal_paths(
        data_dir,
        classes=classes,
        file_template=file_template,
        allow_missing=allow_missing,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(str(model_path))
    rows: list[dict[str, object]] = []
    probabilities_by_file: dict[str, dict[str, float]] = {}
    y_true: list[int] = []
    y_pred: list[int] = []

    for class_name, signal_path in signal_paths:
        signal = np.load(signal_path)
        result = predict_signal(
            model,
            signal,
            classes,
            window_size=window_size,
            step=step,
        )

        true_idx = classes.index(class_name)
        pred_idx = classes.index(str(result["prediction"]))
        correct = int(true_idx == pred_idx)

        y_true.append(true_idx)
        y_pred.append(pred_idx)
        probabilities_by_file[signal_path.name] = result["probabilities"]
        rows.append(
            {
                "file": str(signal_path),
                "true_label": class_name,
                "predicted_label": result["prediction"],
                "confidence": f"{float(result['confidence']):.8f}",
                "windows": int(result["windows"]),
                "correct": correct,
            }
        )

    y_true_array = np.asarray(y_true, dtype=np.int64)
    y_pred_array = np.asarray(y_pred, dtype=np.int64)
    accuracy = float(np.mean(y_true_array == y_pred_array)) if len(y_true_array) else 0.0

    report = write_classification_report(
        y_true_array,
        y_pred_array,
        classes,
        output_dir / "classification_report.txt",
    )
    save_confusion_matrix_plot(
        y_true_array,
        y_pred_array,
        classes,
        output_dir / "confusion_matrix.png",
    )
    write_file_predictions(rows, output_dir / "predictions.csv")
    (output_dir / "probabilities.json").write_text(
        json.dumps(probabilities_by_file, indent=2),
        encoding="utf-8",
    )

    metrics = {
        "data_dir": str(data_dir),
        "model": str(model_path),
        "files": len(rows),
        "accuracy": accuracy,
        "window_size": int(window_size),
        "step": int(step),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print(f"Tested {len(rows)} files from {data_dir}")
    print(f"File-level accuracy: {accuracy:.4f}")
    print(report)
    print(f"Saved reports to: {output_dir}")

    return metrics


def main() -> None:
    args = parse_args()
    run_test(
        model_path=args.model,
        metadata_path=args.metadata,
        data_dir=args.data_dir or default_test_data_dir(),
        output_dir=args.output_dir,
        window_size=args.window_size,
        step=args.step,
        file_template=args.file_template,
        allow_missing=not args.require_all_classes,
    )


if __name__ == "__main__":
    main()
