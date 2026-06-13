"""Generate documentation diagrams for the RF IoT classification pipeline."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

CACHE_DIR = Path(tempfile.gettempdir()) / "iot-device-classification-matplotlib"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
NAVY = "#15324b"
BLUE = "#d9edf7"
LIGHT_BLUE = "#eef6fb"
GREEN = "#dff0d8"
ORANGE = "#fce5cd"
PURPLE = "#e6ddf2"
GRAY = "#667085"


def box(ax, x, y, width, height, title, body, color):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=color,
        edgecolor=NAVY,
        linewidth=1.2,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height * 0.66, title, ha="center", va="center", color=NAVY, fontweight="bold", fontsize=9)
    ax.text(x + width / 2, y + height * 0.34, body, ha="center", va="center", color=NAVY, fontsize=8, linespacing=1.35)


def arrow(ax, start, end, label=None, label_xy=None):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.3,
            color=NAVY,
        )
    )
    if label:
        ax.text(*label_xy, label, ha="center", va="center", fontsize=8, color=GRAY)


def generate_flowchart():
    fig, ax = plt.subplots(figsize=(15, 8.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.955, "RF IoT Classification Pipeline (Multi-Output)", ha="center", va="center", fontsize=20, fontweight="bold", color=NAVY)

    stages = [
        ("Recordings", "manifest CSV\nclass + scenario", BLUE),
        ("Split assignment", "whole captures\ntrain / val / test", GREEN),
        ("Preprocess", "4096 windows +\nspectrogram", LIGHT_BLUE),
        ("Dataset", "sample + balance\n(X, y_dev, y_loc)", GREEN),
        ("CNN training", "train device & loc\nlosses + callbacks", ORANGE),
    ]
    positions = [(0.035, 0.15), (0.225, 0.16), (0.425, 0.16), (0.625, 0.15), (0.815, 0.15)]
    for (title, body, color), (x, width) in zip(stages, positions):
        box(ax, x, 0.63, width, 0.15, title, body, color)
    for x1, x2 in [(0.185, 0.225), (0.385, 0.425), (0.585, 0.625), (0.775, 0.815)]:
        arrow(ax, (x1, 0.705), (x2, 0.705))

    ax.plot([0.04, 0.96], [0.565, 0.565], color="#d0d5dd", linewidth=1)
    ax.text(0.5, 0.535, "INFERENCE AND SCENARIO TESTING", ha="center", color=GRAY, fontsize=10, fontweight="bold")
    lower = [
        ("Test recordings", "manifest test split\nscenario metadata", PURPLE),
        ("Sliding windows", "4096 samples\nstep = 1024", LIGHT_BLUE),
        ("Shared transform", "normalize +\n257 x 61 x 1", LIGHT_BLUE),
        ("CNN predictions", "average probabilities\ndevice & loc heads", ORANGE),
    ]
    lower_positions = [(0.06, 0.17), (0.285, 0.17), (0.51, 0.17), (0.735, 0.20)]
    for (title, body, color), (x, width) in zip(lower, lower_positions):
        box(ax, x, 0.30, width, 0.15, title, body, color)
    for x1, x2 in [(0.23, 0.285), (0.455, 0.51), (0.68, 0.735)]:
        arrow(ax, (x1, 0.375), (x2, 0.375))
    arrow(ax, (0.89, 0.63), (0.835, 0.45), "loads best model", (0.92, 0.535))
    ax.text(
        0.5,
        0.13,
        "Outputs: model metadata and plots   |   file predictions   |   metrics grouped by scenario",
        ha="center",
        fontsize=10,
        color=NAVY,
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#f8fafc", "edgecolor": "#d0d5dd"},
    )
    fig.savefig(HERE / "flowchart.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_architecture():
    fig, ax = plt.subplots(figsize=(16, 7.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.94, "Implemented Spectrogram CNN (Multi-Output)", ha="center", fontsize=20, fontweight="bold", color=NAVY)
    ax.text(0.5, 0.885, "src/model.py: shared CNN backbone with separate device and location heads", ha="center", fontsize=10, color=GRAY)
    
    # Shared Backbone stages
    stages = [
        ("Input", "257 x 61 x 1\nspectrogram", BLUE),
        ("Noise", "GaussianNoise\n0.003", LIGHT_BLUE),
        ("Conv block 1", "32 filters\npool -> 128 x 30", GREEN),
        ("Conv block 2", "64 filters\npool -> 64 x 15", GREEN),
        ("Conv block 3", "128 filters\npool -> 32 x 7", GREEN),
        ("Conv block 4", "256 filters\n32 x 7 x 256", GREEN),
        ("Pooling", "avgmax\nvector: 512", GREEN),
    ]
    
    # Draw Shared Backbone (y centered around 0.425)
    for index, (title, body, color) in enumerate(stages):
        x = 0.02 + index * 0.095
        box(ax, x, 0.425, 0.08, 0.15, title, body, color)
        if index:
            arrow(ax, (x - 0.015, 0.50), (x, 0.50))
            
    # Draw Branch Splits from Pooling (x = 0.59 + 0.08 = 0.67)
    # 1. Device Branch
    box(ax, 0.70, 0.60, 0.11, 0.13, "Device Dense", "Dense(256)\nDropout(0.25)", ORANGE)
    box(ax, 0.84, 0.60, 0.11, 0.13, "Device Output", "Softmax\n6 classes", PURPLE)
    arrow(ax, (0.67, 0.52), (0.70, 0.665))
    arrow(ax, (0.81, 0.665), (0.84, 0.665))
    
    # 2. Location Branch (with StopGradient)
    box(ax, 0.70, 0.35, 0.11, 0.13, "StopGradient", "Blocks backward\ngradients", ORANGE)
    box(ax, 0.70, 0.15, 0.11, 0.13, "Location Dense", "Dense(128) ->\nDense(64)", ORANGE)
    box(ax, 0.84, 0.15, 0.11, 0.13, "Location Output", "Softmax\n3 classes", PURPLE)
    arrow(ax, (0.67, 0.48), (0.70, 0.415))
    arrow(ax, (0.755, 0.35), (0.755, 0.28))
    arrow(ax, (0.81, 0.215), (0.84, 0.215))

    ax.text(0.35, 0.34, "Each convolution block: Conv2D(3 x 3, same) + BatchNormalization + ReLU", ha="center", fontsize=9, color=NAVY)
    ax.text(0.35, 0.295, "Blocks 1-3 also apply MaxPooling2D(2 x 2)", ha="center", fontsize=8.5, color=GRAY)
    
    ax.text(0.15, 0.23, "Optimizer", ha="right", fontweight="bold", fontsize=9.5, color=NAVY)
    ax.text(0.17, 0.23, "Adam, lr = 3e-4", ha="left", fontsize=9.5, color=NAVY)
    
    ax.text(0.15, 0.17, "Loss weights", ha="right", fontweight="bold", fontsize=9.5, color=NAVY)
    ax.text(0.17, 0.17, "device = 1.0  |  location = 0.5 (categorical crossentropy)", ha="left", fontsize=9.5, color=NAVY)
    
    ax.text(0.15, 0.11, "Outputs", ha="right", fontweight="bold", fontsize=9.5, color=NAVY)
    ax.text(0.17, 0.11, "device (dooralarm | lora | microphone | mbus | sigfox | miwi)\nlocation (anotherroom | sameroom | upstairs)", ha="left", fontsize=9.5, color=NAVY)
    
    fig.savefig(HERE / "architecture_diagram.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    generate_flowchart()
    generate_architecture()
    print(f"Generated diagrams in {HERE}")
