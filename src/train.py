"""Train the RF IoT spectrogram classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from config import BATCH_SIZE, CLASSES, EPOCHS, NUM_CLASSES, SEED
from dataset import build_dataset, split_dataset
from evaluation import (
    save_confusion_matrix_plot,
    save_training_curves,
    write_classification_report,
)
from model import build_cnn
from preprocessing import class_distribution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-balance", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    x, y = build_dataset(
        args.data_dir,
        classes=CLASSES,
        balance=not args.no_balance,
        seed=args.seed,
    )
    print(f"Dataset shape: {x.shape}")
    print(f"Label distribution: {class_distribution(y)}")

    x_train, x_val, x_test, y_train, y_val, y_test = split_dataset(
        x,
        y,
        test_size=0.10,
        validation_size=0.10,
        seed=args.seed,
    )

    y_train_cat = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val_cat = tf.keras.utils.to_categorical(y_val, NUM_CLASSES)
    y_test_cat = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

    model = build_cnn(
        input_shape=x_train.shape[1:],
        num_classes=NUM_CLASSES,
        learning_rate=args.learning_rate,
    )
    model.summary()

    best_model_path = args.output_dir / "best_iot_classifier.h5"
    callbacks = [
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            str(best_model_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        x_train,
        y_train_cat,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(x_val, y_val_cat),
        callbacks=callbacks,
        verbose=1,
    )

    final_model_path = args.output_dir / "final_iot_classifier.h5"
    model.save(str(final_model_path))

    test_loss, test_accuracy = model.evaluate(x_test, y_test_cat, verbose=0)
    pred_probs = model.predict(x_test)
    pred_classes = np.argmax(pred_probs, axis=1)

    report_path = args.output_dir / "classification_report.txt"
    report = write_classification_report(y_test, pred_classes, CLASSES, report_path)
    save_confusion_matrix_plot(
        y_test,
        pred_classes,
        CLASSES,
        args.output_dir / "confusion_matrix.png",
    )
    save_training_curves(history, args.output_dir / "training_curves.png")

    metadata = {
        "classes": CLASSES,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "input_shape": list(x_train.shape[1:]),
        "balanced": not args.no_balance,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"\nTest accuracy: {test_accuracy:.4f}")
    print(report)
    print(f"Saved best model: {best_model_path}")
    print(f"Saved final model: {final_model_path}")


if __name__ == "__main__":
    main()
