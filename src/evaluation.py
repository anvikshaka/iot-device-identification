"""Evaluation reports and plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def write_classification_report(
    y_true,
    y_pred,
    classes: list[str],
    output_path: Path,
) -> str:
    """Write the sklearn classification report and return it."""
    labels = list(range(len(classes)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=classes,
        zero_division=0,
    )
    output_path.write_text(report, encoding="utf-8")
    return report


def save_confusion_matrix_plot(
    y_true,
    y_pred,
    classes: list[str],
    output_path: Path,
) -> None:
    """Save a confusion-matrix heatmap."""
    labels = list(range(len(classes)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_training_curves(history, output_path: Path) -> None:
    """Save accuracy and loss curves from a Keras training history."""
    h = history.history
    
    # Check if we have multi-output keys
    has_multi = "device_accuracy" in h
    
    if has_multi:
        plt.figure(figsize=(12, 10))
        
        # Device Accuracy
        plt.subplot(2, 2, 1)
        plt.plot(h["device_accuracy"], label="Train Device Accuracy")
        plt.plot(h["val_device_accuracy"], label="Val Device Accuracy")
        plt.title("Device Accuracy")
        plt.legend()
        
        # Device Loss
        plt.subplot(2, 2, 2)
        plt.plot(h["device_loss"], label="Train Device Loss")
        plt.plot(h["val_device_loss"], label="Val Device Loss")
        plt.title("Device Loss")
        plt.legend()
        
        # Location Accuracy
        plt.subplot(2, 2, 3)
        if "location_accuracy" in h:
            plt.plot(h["location_accuracy"], label="Train Location Accuracy")
        if "val_location_accuracy" in h:
            plt.plot(h["val_location_accuracy"], label="Val Location Accuracy")
        plt.title("Location Accuracy")
        plt.legend()
        
        # Location Loss
        plt.subplot(2, 2, 4)
        if "location_loss" in h:
            plt.plot(h["location_loss"], label="Train Location Loss")
        if "val_location_loss" in h:
            plt.plot(h["val_location_loss"], label="Val Location Loss")
        plt.title("Location Loss")
        plt.legend()
    else:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        acc_key = "accuracy" if "accuracy" in h else ("device_accuracy" if "device_accuracy" in h else None)
        val_acc_key = "val_accuracy" if "val_accuracy" in h else ("val_device_accuracy" if "val_device_accuracy" in h else None)
        if acc_key:
            plt.plot(h[acc_key], label="Train Accuracy")
        if val_acc_key:
            plt.plot(h[val_acc_key], label="Validation Accuracy")
        plt.title("Accuracy")
        plt.legend()
        
        plt.subplot(1, 2, 2)
        loss_key = "loss" if "loss" in h else ("device_loss" if "device_loss" in h else None)
        val_loss_key = "val_loss" if "val_loss" in h else ("val_device_loss" if "val_device_loss" in h else None)
        if loss_key:
            plt.plot(h[loss_key], label="Train Loss")
        if val_loss_key:
            plt.plot(h[val_loss_key], label="Validation Loss")
        plt.title("Loss")
        plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
