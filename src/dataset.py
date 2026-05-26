"""Dataset loading and splitting utilities."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from config import CLASSES, DATA_FILES, SEED, TEST_DATA_FILES, WINDOW_SIZE
from preprocessing import segment_signal, windows_to_model_input


@dataclass(frozen=True)
class Recording:
    """One independently captured signal file and its experimental metadata."""

    path: Path
    class_name: str
    scenario: str = "unspecified"
    capture_id: str = ""
    split: str | None = None


def load_manifest(
    manifest_path: Path,
    classes: list[str] | None = None,
    split: str | None = None,
) -> list[Recording]:
    """Load recording assignments from a CSV manifest.

    Paths in the manifest are interpreted relative to the working directory,
    matching the command-line paths used throughout this project.
    """
    classes = classes or CLASSES
    required_columns = {"path", "class", "scenario", "capture_id", "split"}
    all_recordings: list[Recording] = []

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"Manifest {manifest_path} is missing columns: {missing_text}")

        for row_number, row in enumerate(reader, start=2):
            class_name = row["class"].strip()
            if class_name not in classes:
                raise ValueError(
                    f"Manifest {manifest_path} row {row_number} has unknown class: {class_name}"
                )

            row_split = row["split"].strip()
            path = Path(row["path"].strip())
            if not path.exists():
                raise FileNotFoundError(
                    f"Manifest {manifest_path} row {row_number} references missing file: {path}"
                )
            all_recordings.append(
                Recording(
                    path=path,
                    class_name=class_name,
                    scenario=row["scenario"].strip() or "unspecified",
                    capture_id=row["capture_id"].strip(),
                    split=row_split or None,
                )
            )

    assigned_paths: dict[Path, str | None] = {}
    for recording in all_recordings:
        normalized_path = recording.path.resolve()
        if normalized_path in assigned_paths:
            raise ValueError(
                f"Manifest {manifest_path} assigns recording more than once: {recording.path}"
            )
        assigned_paths[normalized_path] = recording.split

    recordings = [
        recording
        for recording in all_recordings
        if split is None or recording.split == split
    ]
    if not recordings:
        requested = f" for split={split}" if split else ""
        raise ValueError(f"Manifest {manifest_path} contains no recordings{requested}")

    return recordings


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


def legacy_recordings(
    data_dir: Path,
    classes: list[str] | None = None,
    file_map: dict[str, str] | None = None,
    file_template: str | None = None,
    allow_missing: bool = False,
) -> list[Recording]:
    """Represent one-file-per-class data directories as recording metadata."""
    return [
        Recording(path=path, class_name=class_name)
        for class_name, path in iter_class_signal_paths(
            data_dir,
            classes=classes,
            file_map=file_map,
            file_template=file_template,
            allow_missing=allow_missing,
        )
    ]


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
    data_dir: Path | None = None,
    classes: list[str] | None = None,
    window_size: int = WINDOW_SIZE,
    balance: bool = True,
    seed: int = SEED,
    file_map: dict[str, str] | None = None,
    file_template: str | None = None,
    allow_missing: bool = False,
    manifest_path: Path | None = None,
    split: str | None = None,
    max_windows_per_class: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build model inputs from legacy directories or a multi-recording manifest."""
    classes = classes or CLASSES
    if manifest_path is not None:
        recordings = load_manifest(manifest_path, classes=classes, split=split)
    elif data_dir is not None:
        recordings = legacy_recordings(
            data_dir,
            classes=classes,
            file_map=file_map,
            file_template=file_template,
            allow_missing=allow_missing,
        )
    else:
        raise ValueError("Provide data_dir or manifest_path")

    return build_dataset_from_recordings(
        recordings,
        classes=classes,
        window_size=window_size,
        balance=balance,
        seed=seed,
        allow_missing=allow_missing,
        max_windows_per_class=max_windows_per_class,
    )


def build_dataset_from_recordings(
    recordings: list[Recording],
    classes: list[str] | None = None,
    window_size: int = WINDOW_SIZE,
    balance: bool = True,
    seed: int = SEED,
    allow_missing: bool = False,
    max_windows_per_class: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create window inputs after recordings have already been assigned a split."""
    classes = classes or CLASSES
    if max_windows_per_class is not None and max_windows_per_class <= 0:
        raise ValueError("max_windows_per_class must be positive")

    windows_by_class: dict[str, list[np.ndarray]] = {class_name: [] for class_name in classes}
    windows_seen: dict[str, int] = {class_name: 0 for class_name in classes}
    rng = np.random.default_rng(seed)

    for recording in recordings:
        if recording.class_name not in classes:
            raise ValueError(f"Unknown recording class: {recording.class_name}")
        raw = np.load(recording.path)
        windows = segment_signal(raw, window_size=window_size)
        if len(windows) == 0:
            raise ValueError(
                f"{recording.path} has fewer than {window_size} samples and cannot be segmented"
            )
        if max_windows_per_class is None:
            windows_by_class[recording.class_name].append(windows)
            continue

        samples = windows_by_class[recording.class_name]
        for window in windows:
            windows_seen[recording.class_name] += 1
            seen = windows_seen[recording.class_name]
            if len(samples) < max_windows_per_class:
                samples.append(np.array(window, copy=True))
                continue
            candidate = int(rng.integers(0, seen))
            if candidate < max_windows_per_class:
                samples[candidate] = np.array(window, copy=True)

    missing = [name for name, windows in windows_by_class.items() if not windows]
    if missing and not allow_missing:
        raise ValueError(f"No recordings provided for classes: {', '.join(missing)}")

    class_windows = {
        class_name: (
            np.asarray(windows)
            if max_windows_per_class is not None
            else np.concatenate(windows, axis=0)
        )
        for class_name, windows in windows_by_class.items()
        if windows
    }
    balanced_limit = min(len(windows) for windows in class_windows.values()) if balance else None
    all_specs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for idx, class_name in enumerate(classes):
        if class_name not in class_windows:
            continue
        windows = class_windows[class_name]
        limit = balanced_limit if balance else len(windows)
        if max_windows_per_class is not None:
            limit = min(limit, max_windows_per_class)
        if len(windows) > limit:
            indices = np.sort(rng.choice(len(windows), size=limit, replace=False))
            windows = windows[indices]
        class_specs = windows_to_model_input(windows)
        all_specs.append(class_specs)
        all_labels.append(np.full(len(class_specs), idx, dtype=np.int64))

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
