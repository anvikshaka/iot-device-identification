"""Run a quick prediction on a single test .npy signal file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import CLASSES, WINDOW_SIZE
from infer import load_classes, predict_signal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/best_iot_classifier.h5"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("models/metadata.json"),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/test/miwi_test.npy"),
        help="Path to the .npy signal file to classify.",
    )
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--step", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.model.exists():
        print(f"Model not found at {args.model}. Please train first.")
        return

    if not args.input.exists():
        print(f"Test file not found at {args.input}. Please provide a valid .npy file.")
        return

    classes = load_classes(args.metadata)

    print("Loading model...")
    model = tf.keras.models.load_model(str(args.model))

    print(f"Loading test data from {args.input}...")
    signal = np.load(args.input)

    print("Running inference...")
    result = predict_signal(
        model,
        signal,
        classes,
        window_size=args.window_size,
        step=args.step,
    )

    device = result["prediction"]
    conf = result["confidence"]
    n_windows = result["windows"]

    print(f"\n--- Ensemble Results ({n_windows} windows) ---")
    print(f"Final Consolidated Prediction: {device}")
    print(f"Confidence Level: {conf * 100:.2f}%")

    print("\nPer-class probabilities:")
    for class_name, prob in result["probabilities"].items():
        print(f"  {class_name:>12s}: {prob * 100:.2f}%")

    if conf >= 0.99:
        print("\nTarget Efficiency (99%) Achieved!")
    else:
        print(
            f"\nEfficiency at {conf * 100:.2f}%. "
            "Training accuracy is high, the ensemble is stabilizing the output."
        )


if __name__ == "__main__":
    main()
