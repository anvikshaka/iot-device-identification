import os
import sys
import csv
from pathlib import Path

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from dataset_builder import get_train_val_test_splits
from evaluate import run_evaluation
from spectrogram import create_spectrograms
from test import run_test, summarize_scenarios, wilson_confidence_interval


def test_legacy_spectrogram_export_delegates_to_canonical_transform():
    windows = np.random.default_rng(42).standard_normal((1, 4096))

    assert create_spectrograms(windows).shape == (1, 257, 61, 1)


def test_legacy_split_wrapper_remains_importable():
    x = np.zeros((60, 2, 2, 1), dtype=np.float32)
    y = np.repeat(np.arange(6), 10)

    train, validation, test, y_test = get_train_val_test_splits(x, y)

    assert train[0].shape[0] == 48
    assert validation[0].shape[0] == 6
    assert test[0].shape[0] == 6
    assert train[1].shape[1] == 6
    assert len(y_test) == 6


def test_small_external_score_has_wide_file_level_confidence_interval():
    low, high = wilson_confidence_interval(6, 6)

    assert 0.60 < low < 0.62
    assert high == 1.0


def test_window_evaluation_requires_an_explicit_role():
    with pytest.raises(ValueError, match="evaluation_role must be one of"):
        run_evaluation(
            model_path=Path("missing-model.h5"),
            metadata_path=Path("missing-metadata.json"),
            data_dir=Path("missing-data"),
            output_dir=Path("missing-output"),
        )


def test_scenario_summary_reports_file_level_results():
    rows = [
        {"scenario": "room", "correct": 1},
        {"scenario": "room", "correct": 0},
        {"scenario": "upstairs", "correct": 1},
    ]

    summary = summarize_scenarios(rows)

    assert summary["room"]["files"] == 2
    assert summary["room"]["accuracy"] == 0.5
    assert summary["upstairs"]["accuracy"] == 1.0


def test_manifest_file_test_can_require_complete_class_coverage(tmp_path):
    signal_path = tmp_path / "microphone.npy"
    np.save(signal_path, np.zeros(4096, dtype=np.float32))
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "class", "scenario", "capture_id", "split"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "path": signal_path,
                "class": "microphone",
                "scenario": "upstairs",
                "capture_id": "01",
                "split": "test",
            }
        )

    with pytest.raises(ValueError, match="has no recordings for classes"):
        run_test(
            model_path=Path("missing-model.h5"),
            metadata_path=Path("missing-metadata.json"),
            data_dir=None,
            output_dir=tmp_path / "output",
            manifest_path=manifest,
            allow_missing=False,
        )
