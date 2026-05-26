"""Test a trained RF IoT classifier on external labeled test files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import WINDOW_SIZE
from dataset import legacy_recordings, load_manifest
from evaluation import (
    save_confusion_matrix_plot,
    write_classification_report,
)
from infer import load_classes, predict_signal


def wilson_confidence_interval(
    successes: int,
    total: int,
    z: float = 1.96,
) -> tuple[float, float]:
    """Return a Wilson score confidence interval for a binomial accuracy."""
    if total <= 0:
        return 0.0, 0.0

    proportion = successes / total
    denominator = 1 + (z**2 / total)
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * np.sqrt((proportion * (1 - proportion) / total) + (z**2 / (4 * total**2)))
        / denominator
    )
    low = 0.0 if successes == 0 else max(0.0, float(center - margin))
    high = 1.0 if successes == total else min(1.0, float(center + margin))
    return low, high


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
    parser.add_argument(
        "--manifest",
        type=Path,
        help="CSV recording manifest. When provided, test the selected manifest split instead of --data-dir.",
    )
    parser.add_argument(
        "--manifest-split",
        default="test",
        help="Manifest split to test when --manifest is used (default: test).",
    )
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
                "scenario",
                "capture_id",
                "split",
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
    data_dir: Path | None,
    output_dir: Path,
    window_size: int = WINDOW_SIZE,
    step: int = 1024,
    file_template: str | None = None,
    allow_missing: bool = True,
    manifest_path: Path | None = None,
    manifest_split: str = "test",
) -> dict[str, object]:
    """Run file-level sliding-window ensemble tests and save reports."""
    classes = load_classes(metadata_path)
    if manifest_path:
        recordings = load_manifest(manifest_path, classes=classes, split=manifest_split)
        if not allow_missing:
            available = {recording.class_name for recording in recordings}
            missing = [class_name for class_name in classes if class_name not in available]
            if missing:
                raise ValueError(
                    f"Manifest split={manifest_split} has no recordings for classes: {', '.join(missing)}"
                )
    elif data_dir:
        recordings = legacy_recordings(
            data_dir,
            classes=classes,
            file_template=file_template,
            allow_missing=allow_missing,
        )
    else:
        raise ValueError("Provide data_dir or manifest_path")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(str(model_path))
    rows: list[dict[str, object]] = []
    probabilities_by_file: dict[str, dict[str, float]] = {}
    y_true: list[int] = []
    y_pred: list[int] = []

    for recording in recordings:
        signal = np.load(recording.path)
        result = predict_signal(
            model,
            signal,
            classes,
            window_size=window_size,
            step=step,
        )

        true_idx = classes.index(recording.class_name)
        pred_idx = classes.index(str(result["prediction"]))
        correct = int(true_idx == pred_idx)

        y_true.append(true_idx)
        y_pred.append(pred_idx)
        probabilities_by_file[str(recording.path)] = result["probabilities"]
        rows.append(
            {
                "file": str(recording.path),
                "true_label": recording.class_name,
                "scenario": recording.scenario,
                "capture_id": recording.capture_id,
                "split": recording.split or "",
                "predicted_label": result["prediction"],
                "confidence": f"{float(result['confidence']):.8f}",
                "windows": int(result["windows"]),
                "correct": correct,
            }
        )

    y_true_array = np.asarray(y_true, dtype=np.int64)
    y_pred_array = np.asarray(y_pred, dtype=np.int64)
    correct_files = int(np.sum(y_true_array == y_pred_array))
    accuracy = float(correct_files / len(y_true_array)) if len(y_true_array) else 0.0
    accuracy_ci_low, accuracy_ci_high = wilson_confidence_interval(
        correct_files,
        len(y_true_array),
    )
    scenario_metrics = summarize_scenarios(rows)

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
        "evaluation_role": "external-file-level-ensemble",
        "evaluation_unit": "source_file",
        "data_dir": str(data_dir) if data_dir else None,
        "manifest": str(manifest_path) if manifest_path else None,
        "manifest_split": manifest_split if manifest_path else None,
        "model": str(model_path),
        "files": len(rows),
        "correct_files": correct_files,
        "accuracy": accuracy,
        "accuracy_ci_95_wilson": {
            "low": accuracy_ci_low,
            "high": accuracy_ci_high,
        },
        "interpretation": "Each file contributes one prediction; window count is not the number of independent test examples.",
        "scenario_metrics": scenario_metrics,
        "window_size": int(window_size),
        "step": int(step),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    source = f"{manifest_path} split={manifest_split}" if manifest_path else str(data_dir)
    print(f"Tested {len(rows)} files from {source}")
    print(f"File-level accuracy: {accuracy:.4f}")
    print(report)
    print(f"Saved reports to: {output_dir}")

    return metrics


def summarize_scenarios(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    """Summarize independent file-level outcomes for each scenario."""
    grouped: dict[str, list[int]] = {}
    for row in rows:
        scenario = str(row.get("scenario") or "unspecified")
        grouped.setdefault(scenario, []).append(int(row["correct"]))

    summary: dict[str, dict[str, object]] = {}
    for scenario, correctness in sorted(grouped.items()):
        files = len(correctness)
        correct_files = int(sum(correctness))
        low, high = wilson_confidence_interval(correct_files, files)
        summary[scenario] = {
            "files": files,
            "correct_files": correct_files,
            "accuracy": float(correct_files / files),
            "accuracy_ci_95_wilson": {"low": low, "high": high},
        }
    return summary


def main() -> None:
    args = parse_args()
    run_test(
        model_path=args.model,
        metadata_path=args.metadata,
        data_dir=None if args.manifest else (args.data_dir or default_test_data_dir()),
        output_dir=args.output_dir,
        window_size=args.window_size,
        step=args.step,
        file_template=args.file_template,
        allow_missing=not args.require_all_classes,
        manifest_path=args.manifest,
        manifest_split=args.manifest_split,
    )


if __name__ == "__main__":
    main()
