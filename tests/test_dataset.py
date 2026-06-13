import csv
import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from dataset import build_dataset, load_manifest


def write_manifest(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "class", "scenario", "capture_id", "split"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_load_manifest_filters_recordings_by_split(tmp_path):
    train_file = tmp_path / "microphone_train.npy"
    test_file = tmp_path / "microphone_upstairs.npy"
    np.save(train_file, np.zeros(4096, dtype=np.float32))
    np.save(test_file, np.zeros(4096, dtype=np.float32))
    manifest = tmp_path / "recordings.csv"
    write_manifest(
        manifest,
        [
            {
                "path": train_file,
                "class": "microphone",
                "scenario": "baseline",
                "capture_id": "01",
                "split": "train",
            },
            {
                "path": test_file,
                "class": "microphone",
                "scenario": "upstairs",
                "capture_id": "02",
                "split": "test",
            },
        ],
    )

    recordings = load_manifest(manifest, classes=["microphone"], split="test")

    assert len(recordings) == 1
    assert recordings[0].path == test_file
    assert recordings[0].scenario == "upstairs"
    assert recordings[0].capture_id == "02"


def test_build_dataset_combines_recordings_then_balances_classes(tmp_path):
    microphone_a = tmp_path / "microphone_room.npy"
    microphone_b = tmp_path / "microphone_upstairs.npy"
    lora = tmp_path / "lora_room.npy"
    np.save(microphone_a, np.ones(4096, dtype=np.float32))
    np.save(microphone_b, np.ones(4096, dtype=np.float32))
    np.save(lora, np.ones(4096 * 3, dtype=np.float32))
    manifest = tmp_path / "recordings.csv"
    write_manifest(
        manifest,
        [
            {"path": microphone_a, "class": "microphone", "scenario": "sameroom", "capture_id": "01", "split": "train"},
            {"path": microphone_b, "class": "microphone", "scenario": "upstairs", "capture_id": "02", "split": "train"},
            {"path": lora, "class": "lora", "scenario": "sameroom", "capture_id": "01", "split": "train"},
        ],
    )

    x, y_dev, y_loc = build_dataset(
        manifest_path=manifest,
        split="train",
        classes=["microphone", "lora"],
        balance=True,
    )

    assert x.shape == (4, 257, 61, 1)
    assert np.bincount(y_dev).tolist() == [2, 2]


def test_manifest_rejects_a_capture_assigned_to_multiple_splits(tmp_path):
    recording = tmp_path / "microphone.npy"
    np.save(recording, np.zeros(4096, dtype=np.float32))
    manifest = tmp_path / "recordings.csv"
    write_manifest(
        manifest,
        [
            {"path": recording, "class": "microphone", "scenario": "sameroom", "capture_id": "01", "split": "train"},
            {"path": recording, "class": "microphone", "scenario": "sameroom", "capture_id": "01", "split": "test"},
        ],
    )

    with pytest.raises(ValueError, match="assigns recording more than once"):
        load_manifest(manifest, classes=["microphone"])


def test_focused_test_split_can_score_only_one_configured_class(tmp_path):
    microphone = tmp_path / "microphone_upstairs.npy"
    np.save(microphone, np.zeros(4096, dtype=np.float32))
    manifest = tmp_path / "recordings.csv"
    write_manifest(
        manifest,
        [
            {"path": microphone, "class": "microphone", "scenario": "upstairs", "capture_id": "01", "split": "test"},
        ],
    )

    x, y_dev, y_loc = build_dataset(
        manifest_path=manifest,
        split="test",
        classes=["microphone", "lora"],
        allow_missing=True,
    )

    assert x.shape[0] == 1
    assert y_dev.tolist() == [0]


def test_window_cap_is_applied_per_class_before_model_input_creation(tmp_path):
    microphone = tmp_path / "microphone.npy"
    lora = tmp_path / "lora.npy"
    np.save(microphone, np.ones(4096 * 3, dtype=np.float32))
    np.save(lora, np.ones(4096 * 3, dtype=np.float32))
    manifest = tmp_path / "recordings.csv"
    write_manifest(
        manifest,
        [
            {"path": microphone, "class": "microphone", "scenario": "sameroom", "capture_id": "01", "split": "train"},
            {"path": lora, "class": "lora", "scenario": "sameroom", "capture_id": "01", "split": "train"},
        ],
    )

    x, y_dev, y_loc = build_dataset(
        manifest_path=manifest,
        split="train",
        classes=["microphone", "lora"],
        max_windows_per_class=1,
    )

    assert x.shape == (2, 257, 61, 1)
    assert sorted(y_dev.tolist()) == [0, 1]
