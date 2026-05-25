"""Dataset loading and splitting utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from config import CLASSES, DATA_FILES, SEED, TEST_DATA_FILES, WINDOW_SIZE
from preprocessing import segment_signal, windows_to_model_input


def class_file_candidates(
    class_name: str,
    file_map: dict[str, str] | None = None,
    file_template: str | None = None,
) -> list[str]:
    """Return supported file names for a class, in lookup order."""
    if file_map is not None:
        return [file_map[class_name]]
    if file_template is not None:
        return [file_template.format(class_name=class_name, class_label=class_name)]

    raw_name = DATA_FILES.get(class_name, f"{class_name}.npy")
    test_name = TEST_DATA_FILES.get(class_name, f"{class_name}_test.npy")
    return [raw_name, test_name]


def find_class_signal_path(
    data_dir: Path,
    class_name: str,
    file_map: dict[str, str] | None = None,
    file_template: str | None = None,
) -> Path:
    """Find a class signal file in a directory."""
    candidates = [
        data_dir / name
        for name in class_file_candidates(class_name, file_map, file_template)
    ]
    for path in candidates:
        if path.exists():
            return path

    expected = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Missing required data file for {class_name}. Expected one of: {expected}"
    )


def iter_class_signal_paths(
    data_dir: Path,
    classes: list[str] | None = None,
    file_map: dict[str, str] | None = None,
    file_template: str | None = None,
    allow_missing: bool = False,
) -> list[tuple[str, Path]]:
    """List labeled signal files available in a directory."""
    classes = classes or CLASSES
    paths: list[tuple[str, Path]] = []

    for class_name in classes:
        try:
            paths.append(
                (
                    class_name,
                    find_class_signal_path(
                        data_dir,
                        class_name,
                        file_map=file_map,
                        file_template=file_template,
                    ),
                )
            )
        except FileNotFoundError:
            if allow_missing:
                continue
            raise

    if not paths:
        raise FileNotFoundError(f"No labeled .npy files found in {data_dir}")

    return paths


def load_raw_signal(
    data_dir: Path,
    class_name: str,
    file_map: dict[str, str] | None = None,
    file_template: str | None = None,
) -> np.ndarray:
    """Load one raw class signal from disk."""
    path = find_class_signal_path(
        data_dir,
        class_name,
        file_map=file_map,
        file_template=file_template,
    )
    return np.load(path)


def build_dataset(
    data_dir: Path,
    classes: list[str] | None = None,
    window_size: int = WINDOW_SIZE,
    balance: bool = True,
    seed: int = SEED,
    file_map: dict[str, str] | None = None,
    file_template: str | None = None,
    allow_missing: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Load all class signals, segment them, create spectrograms, and label them."""
    classes = classes or CLASSES
    all_specs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for class_name, signal_path in iter_class_signal_paths(
        data_dir,
        classes=classes,
        file_map=file_map,
        file_template=file_template,
        allow_missing=allow_missing,
    ):
        idx = classes.index(class_name)
        raw = np.load(signal_path)
        windows = segment_signal(raw, window_size=window_size)
        if len(windows) == 0:
            raise ValueError(
                f"{class_name} has fewer than {window_size} samples and cannot be segmented"
            )

        specs = windows_to_model_input(windows)
        all_specs.append(specs)
        all_labels.append(np.full(len(specs), idx, dtype=np.int64))

    if balance:
        min_samples = min(len(specs) for specs in all_specs)
        all_specs = [specs[:min_samples] for specs in all_specs]
        all_labels = [labels[:min_samples] for labels in all_labels]

    x = np.concatenate(all_specs, axis=0)
    y = np.concatenate(all_labels, axis=0)

    return shuffle(x, y, random_state=seed)


def split_dataset(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.10,
    validation_size: float = 0.10,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create stratified train, validation, and test splits."""
    holdout_size = test_size + validation_size
    if not 0 < holdout_size < 1:
        raise ValueError("test_size + validation_size must be between 0 and 1")

    x_train, x_holdout, y_train, y_holdout = train_test_split(
        x,
        y,
        test_size=holdout_size,
        random_state=seed,
        stratify=y,
    )

    relative_test_size = test_size / holdout_size
    x_val, x_test, y_val, y_test = train_test_split(
        x_holdout,
        y_holdout,
        test_size=relative_test_size,
        random_state=seed,
        stratify=y_holdout,
    )

    return x_train, x_val, x_test, y_train, y_val, y_test
