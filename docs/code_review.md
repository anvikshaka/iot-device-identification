# Code Review Remediation

Review scope: `src/`, `tests/`, `README.md`, and checked-in artifacts, reviewed
and remediated on 2026-05-25. The documentation and diagrams describe the
executable path through `train.py`, `evaluate.py`, `test.py`, and `predict.py`.

## Actions

### Resolved: Evaluation role is explicit

**Location:** `src/evaluate.py:27`, `src/evaluate.py:76-86`;
`results/evaluation/metrics.json`

`evaluate.py` defaults to `data/raw`, the same input directory used by
`train.py` to construct training windows. The checked-in `0.9900` accuracy is
therefore a diagnostic measurement on training-source recordings, not an
external validation score. Reporting it without that distinction would
overstate model generalization.

**Action:** `evaluate.py` now requires `--data-dir` or `--manifest` and
`--evaluation-role`, whose choices distinguish training-source diagnostics from
external window evaluation. Metrics include `evaluation_role` and
`evaluation_unit`. The checked-in raw-directory metric has been marked
`training-source-diagnostic`.

### Resolved: Model test targets the implemented API

**Location:** `tests/test_model.py:6-13`; `src/model.py:22-65`

The test imports `build_model`, while the implementation exports `build_cnn`
and requires `num_classes`. Importing the tested symbol raises
`ImportError: cannot import name 'build_model' from 'model'`.

**Action:** `tests/test_model.py` now calls
`build_cnn(input_shape, NUM_CLASSES)` and verifies loss and softmax output
alongside the expected output shape.

### Resolved: Legacy dataset API delegates to the canonical path

**Location:** `src/dataset_builder.py:6`; `src/config.py:8-30`

`dataset_builder.py` imports lowercase `classes` and `data_files`, but the
configuration exposes uppercase `CLASSES` and `DATA_FILES`. Importing
`dataset_builder` fails immediately. The active training pipeline uses
`src/dataset.py`, so this is currently dead code, but it makes the repository
appear to provide an unusable second dataset API.

**Action:** `dataset_builder.py` is retained as a compatibility wrapper for
notebook-era callers but uses `dataset.build_dataset()` and
`dataset.split_dataset()`. A compatibility test covers its legacy split
return signature.

### Resolved: Preprocessing paths are consolidated

**Location:** `src/preprocessing.py:51-79`; `src/spectrogram.py:4-30`;
`src/visualize.py:24-38`

Training and inference use `preprocessing.py`, while the unit test targets
`spectrogram.py`, and visualization uses different overlap settings. Tests can
remain green while the transformation actually supplied to the model changes.

**Action:** `spectrogram.py` is now a compatibility export of
`preprocessing.create_spectrograms()`. Tests target the canonical module and
cover `windows_to_model_input()`. `visualize.py` renders the normalized
spectrogram supplied to the CNN instead of computing a differently configured
display spectrogram.

### Mitigated: External result reports its limited statistical scope

**Location:** `src/test.py:88-146`; `results/test/metrics.json`;
`results/test/predictions.csv`

The checked-in file-level result is six correct predictions from six files.
Each prediction averages thousands of overlapping windows from one recording;
those windows are not independent test cases. The result is useful as a
pipeline check but is not strong evidence for a 100% accuracy claim.

**Action:** `test.py` reports `correct_files`, `evaluation_unit`, and a 95%
Wilson confidence interval over independent file predictions, together with an
interpretation note. The six-file checked-in result has a wide interval of
approximately `[0.610, 1.000]`. Gathering more independent captures remains a
data requirement rather than a code change.

### Resolved: Scenario datasets can be evaluated without capture leakage

**Action:** `dataset.py` now supports a manifest with `path`, `class`,
`scenario`, `capture_id`, and `split` fields. `train.py --manifest` builds
training, validation, and held-out datasets after entire recordings have been
assigned to their split. `test.py --manifest` performs one ensemble prediction
per held-out recording and reports scenario-level file metrics.

## Verification Notes

- `pytest` is now listed in `requirements.txt` as a development/test
  requirement.
- The updated test suite covers active model/preprocessing behavior,
  compatibility imports, manifest ingestion, class balancing across multiple
  recordings, and scenario-level file summaries.
- Diagram placeholders under `docs/` were zero-byte files before the
  documentation update; `docs/generate_diagrams.py` now regenerates them.
