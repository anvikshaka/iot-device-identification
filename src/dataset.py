"""Dataset loading and splitting utilities."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from config import CLASSES, DATA_FILES, LOCATIONS, SEED, TEST_DATA_FILES, WINDOW_SIZE
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
    locations: list[str] | None = None,
    window_size: int = WINDOW_SIZE,
    balance: bool = True,
    seed: int = SEED,
    file_map: dict[str, str] | None = None,
    file_template: str | None = None,
    allow_missing: bool = False,
    manifest_path: Path | None = None,
    split: str | None = None,
    max_windows_per_class: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build model inputs from legacy directories or a multi-recording manifest."""
    classes = classes or CLASSES
    locations = locations or LOCATIONS
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
        locations=locations,
        window_size=window_size,
        balance=balance,
        seed=seed,
        allow_missing=allow_missing,
        max_windows_per_class=max_windows_per_class,
    )


def build_dataset_from_recordings(
    recordings: list[Recording],
    classes: list[str] | None = None,
    locations: list[str] | None = None,
    window_size: int = WINDOW_SIZE,
    balance: bool = True,
    seed: int = SEED,
    allow_missing: bool = False,
    max_windows_per_class: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create window inputs after recordings have already been assigned a split."""
    classes = classes or CLASSES
    locations = locations or LOCATIONS
    if max_windows_per_class is not None and max_windows_per_class <= 0:
        raise ValueError("max_windows_per_class must be positive")

    # Group window entries by class name to maintain class-level balance
    WindowEntry = tuple[np.ndarray, int, int]
    entries_by_class: dict[str, list[WindowEntry]] = {class_name: [] for class_name in classes}
    windows_seen: dict[str, int] = {class_name: 0 for class_name in classes}
    rng = np.random.default_rng(seed)

    for recording in recordings:
        if recording.class_name not in classes:
            raise ValueError(f"Unknown recording class: {recording.class_name}")
        
        device_idx = classes.index(recording.class_name)
        scenario = recording.scenario
        # Map scenario to location index or -1 for unspecified
        if scenario in locations:
            location_idx = locations.index(scenario)
        else:
            location_idx = -1

        raw = np.load(recording.path)
        windows = segment_signal(raw, window_size=window_size)
        if len(windows) == 0:
            raise ValueError(
                f"{recording.path} has fewer than {window_size} samples and cannot be segmented"
            )

        if max_windows_per_class is None:
            for w in windows:
                entries_by_class[recording.class_name].append((w, device_idx, location_idx))
            continue

        samples = entries_by_class[recording.class_name]
        for window in windows:
            windows_seen[recording.class_name] += 1
            seen = windows_seen[recording.class_name]
            entry = (np.array(window, copy=True), device_idx, location_idx)
            if len(samples) < max_windows_per_class:
                samples.append(entry)
                continue
            candidate = int(rng.integers(0, seen))
            if candidate < max_windows_per_class:
                samples[candidate] = entry

    missing = [name for name, entries in entries_by_class.items() if not entries]
    if missing and not allow_missing:
        raise ValueError(f"No recordings provided for classes: {', '.join(missing)}")

    class_entries: dict[str, list[WindowEntry]] = {
        class_name: entries
        for class_name, entries in entries_by_class.items()
        if entries
    }
    balanced_limit = min(len(entries) for entries in class_entries.values()) if balance else None
    
    all_specs: list[np.ndarray] = []
    all_device_labels: list[np.ndarray] = []
    all_location_labels: list[np.ndarray] = []

    for class_name in classes:
        if class_name not in class_entries:
            continue
        entries = class_entries[class_name]
        limit = balanced_limit if balance else len(entries)
        if max_windows_per_class is not None:
            limit = min(limit, max_windows_per_class)
        if len(entries) > limit:
            indices = np.sort(rng.choice(len(entries), size=limit, replace=False))
            entries = [entries[i] for i in indices]

        windows_array = np.asarray([e[0] for e in entries])
        class_specs = windows_to_model_input(windows_array)
        all_specs.append(class_specs)
        all_device_labels.append(np.array([e[1] for e in entries], dtype=np.int64))
        all_location_labels.append(np.array([e[2] for e in entries], dtype=np.int64))

    x = np.concatenate(all_specs, axis=0)
    y_device = np.concatenate(all_device_labels, axis=0)
    y_location = np.concatenate(all_location_labels, axis=0)

    # Shuffle all three arrays synchronously
    return shuffle(x, y_device, y_location, random_state=seed)


def split_dataset(
    x: np.ndarray,
    y_device: np.ndarray,
    y_location: np.ndarray,
    test_size: float = 0.10,
    validation_size: float = 0.10,
    seed: int = SEED,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
]:
    """Create stratified train, validation, and test splits."""
    holdout_size = test_size + validation_size
    if not 0 < holdout_size < 1:
        raise ValueError("test_size + validation_size must be between 0 and 1")

    (
        x_train, x_holdout,
        y_dev_train, y_dev_holdout,
        y_loc_train, y_loc_holdout,
    ) = train_test_split(
        x,
        y_device,
        y_location,
        test_size=holdout_size,
        random_state=seed,
        stratify=y_device,
    )

    relative_test_size = test_size / holdout_size
    (
        x_val, x_test,
        y_dev_val, y_dev_test,
        y_loc_val, y_loc_test,
    ) = train_test_split(
        x_holdout,
        y_dev_holdout,
        y_loc_holdout,
        test_size=relative_test_size,
        random_state=seed,
        stratify=y_dev_holdout,
    )

    return (
        x_train, x_val, x_test,
        y_dev_train, y_dev_val, y_dev_test,
        y_loc_train, y_loc_val, y_loc_test,
    )
