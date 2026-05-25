"""Run inference on a raw signal .npy file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import CLASSES, WINDOW_SIZE
from preprocessing import sliding_windows, windows_to_model_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/best_iot_classifier.h5"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=Path("models/metadata.json"))
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--step", type=int, default=1024)
    return parser.parse_args()


def load_classes(metadata_path: Path) -> list[str]:
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return metadata.get("classes", CLASSES)
    return CLASSES


def predict_signal(
    model: tf.keras.Model,
    signal: np.ndarray,
    classes: list[str],
    window_size: int = WINDOW_SIZE,
    step: int = 1024,
) -> dict[str, object]:
    windows = sliding_windows(signal, window_size=window_size, step=step)
    if len(windows) == 0:
        raise ValueError(
            f"Input signal has {len(signal)} samples, fewer than window_size={window_size}"
        )

    specs = windows_to_model_input(windows)
    raw_probs = model.predict(specs)
    avg_probs = np.mean(raw_probs, axis=0)
    final_idx = int(np.argmax(avg_probs))

    return {
        "prediction": classes[final_idx],
        "confidence": float(avg_probs[final_idx]),
        "windows": int(len(windows)),
        "probabilities": {
            class_name: float(probability)
            for class_name, probability in zip(classes, avg_probs)
        },
    }


def main() -> None:
    args = parse_args()
    classes = load_classes(args.metadata)
    model = tf.keras.models.load_model(str(args.model))
    signal = np.load(args.input)

    result = predict_signal(
        model,
        signal,
        classes,
        window_size=args.window_size,
        step=args.step,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

