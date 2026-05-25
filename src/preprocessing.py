"""Signal windowing and spectrogram preprocessing."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy import signal as sg

from config import SPECTROGRAM_CONFIG, WINDOW_SIZE


def segment_signal(data: np.ndarray, window_size: int = WINDOW_SIZE) -> np.ndarray:
    """Split a one-dimensional signal into non-overlapping fixed-size windows."""
    data = np.asarray(data).reshape(-1)
    num_samples = len(data) // window_size
    if num_samples == 0:
        return np.empty((0, window_size), dtype=data.dtype)

    return data[: num_samples * window_size].reshape(num_samples, window_size)


def sliding_windows(
    data: np.ndarray,
    window_size: int = WINDOW_SIZE,
    step: int = 1024,
) -> np.ndarray:
    """Split a signal into overlapping windows for inference-time voting."""
    data = np.asarray(data).reshape(-1)
    if len(data) < window_size:
        return np.empty((0, window_size), dtype=data.dtype)

    windows = [
        data[start : start + window_size]
        for start in range(0, len(data) - window_size + 1, step)
    ]
    return np.asarray(windows)


def normalize_windows(windows: np.ndarray) -> np.ndarray:
    """Normalize each RF window independently."""
    windows = np.asarray(windows)
    normalized = (windows - np.mean(windows, axis=1, keepdims=True)) / (
        np.std(windows, axis=1, keepdims=True) + 1e-8
    )
    if np.iscomplexobj(normalized):
        return normalized.astype(np.complex64)
    return normalized.astype(np.float32)


def create_spectrograms(
    windows: np.ndarray,
    spectrogram_config: dict[str, object] | None = None,
) -> np.ndarray:
    """Create normalized log-compressed spectrograms with a channel dimension."""
    config = spectrogram_config or SPECTROGRAM_CONFIG
    specs = []

    for window in windows:
        _, _, spectrum = sg.spectrogram(window, **config)
        spec = np.log1p(spectrum)
        spec = (spec - np.mean(spec)) / (np.std(spec) + 1e-8)
        specs.append(spec.astype(np.float32))

    if not specs:
        return np.empty((0, 0, 0, 1), dtype=np.float32)

    return np.asarray(specs, dtype=np.float32)[..., np.newaxis]


def windows_to_model_input(windows: np.ndarray) -> np.ndarray:
    """Apply the full preprocessing path used by both training and inference."""
    if len(windows) == 0:
        return np.empty((0, 0, 0, 1), dtype=np.float32)

    normalized = normalize_windows(windows)
    return create_spectrograms(normalized)


def class_distribution(labels: Iterable[int]) -> dict[int, int]:
    """Return a deterministic class-count dictionary."""
    unique, counts = np.unique(np.asarray(list(labels)), return_counts=True)
    return {int(label): int(count) for label, count in zip(unique, counts)}
