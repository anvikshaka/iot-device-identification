# Methodology

## Problem Definition

The classifier maps raw RF signal recordings to six labels:
`dooralarm`, `lora`, `microphone`, `mbus`, `sigfox`, and `miwi`. The label
ordering is fixed in `src/config.py` and is preserved in model metadata and
probability output.

## Input Data

The recommended workflow accepts multiple independent NumPy recordings per
class through a CSV manifest. The local legacy dataset reviewed for this
documentation contains `10,000,000` `float32` samples in each raw and external
test file.

| Purpose | File pattern | Treatment |
| --- | --- | --- |
| Multi-recording training/evaluation | Manifest row: `path,class,scenario,capture_id,split` | Complete capture assigned to train, validation, or test before segmentation |
| Legacy training source | `data/raw/<class>.npy` | Non-overlapping windows randomly partitioned; compatibility mode only |
| Legacy external test source | `data/test/<class>_test.npy` | Whole-file prediction using overlapping inference windows |

Manifest validation rejects a signal file listed more than once, so the same
capture cannot accidentally appear in two experimental splits.
Training and validation require recordings for all configured classes. The
test split may contain only a targeted held-out device/scenario, such as
microphone recordings from a new room.

With `WINDOW_SIZE = 4096`, each local raw file supplies `2441` complete
training windows; remaining trailing samples are discarded. The balanced
six-class training-source dataset therefore contains `14,646` windows before
splitting.

## Signal Preprocessing

The active transformation is implemented in `src/preprocessing.py` and shared
by training and inference.

| Step | Configuration/behavior |
| --- | --- |
| Window length | `4096` samples |
| Training/evaluation segmentation | Non-overlapping windows |
| Whole-file inference segmentation | Overlapping windows, default step `1024` |
| Window normalization | Subtract the window mean and divide by its standard deviation plus `1e-8` |
| Spectrogram window | Hann |
| `nperseg` / `noverlap` / `nfft` | `256` / `192` / `512` |
| Spectrogram scaling | `spectrum` |
| Compression | `log1p(spectrum)` |
| Spectrogram normalization | Per-spectrogram zero mean and unit standard deviation |
| CNN input shape | `257 x 61 x 1` |

For inference on a file, the CNN output probabilities are averaged over all
overlapping windows. This provides one consolidated prediction and confidence
per signal file.

## Model and Training

The CNN consists of four convolutional feature-extraction blocks followed by a
fully connected classification head. Detailed layer shapes appear in
[architecture.md](architecture.md).

Default model configuration:

| Setting | Value |
| --- | --- |
| Classes | `6` |
| Batch size | `32` |
| Maximum epochs | `15` |
| Seed | `42` |
| Learning rate | `3e-4` |
| Optimizer | Adam |
| Loss | Categorical cross-entropy |
| Recommended split | Explicit recording-level `train` / `validation` / `test` assignments in a manifest |
| Legacy split | Stratified 80% train / 10% validation / 10% held-out windows |
| Balancing | Enabled by default; truncate each class to the minimum class window count |
| Large multi-capture sets | Use `--max-windows-per-class N` for bounded reservoir sampling per split/class before spectrogram conversion |

Training callbacks:

| Callback | Setting |
| --- | --- |
| `ReduceLROnPlateau` | Monitor validation loss, factor `0.5`, patience `3`, minimum LR `1e-6` |
| `EarlyStopping` | Monitor validation accuracy, patience `6`, restore best weights |
| `ModelCheckpoint` | Save the best validation-accuracy model to `models/best_iot_classifier.h5` |

## Evaluation Modes

The repository exposes three distinct evaluations; their metrics are not
interchangeable.

| Command/path | Unit measured | Appropriate interpretation |
| --- | --- | --- |
| `python src/train.py --manifest ...` held-out report | Windows from whole recordings assigned to `test` before segmentation | Internal model-training report without capture leakage |
| Legacy training held-out report in `models/` | Windows held out after random split of `data/raw` | Historical estimate; windows originate from the same source recordings as training |
| `python src/evaluate.py --data-dir data/raw --evaluation-role training-source-diagnostic` | Individual windows loaded from `data/raw` | Diagnostic scoring on the training-source recordings, not independent test performance |
| `python src/test.py --manifest ... --manifest-split test` | One consolidated prediction per held-out file, grouped by scenario | Recommended scenario evaluation |
| `python src/test.py --data-dir data/test` output in `results/test/` | One consolidated prediction per legacy external file | Historical external check; currently only six scored files |

## Recorded Results

The following figures are read from checked-in artifacts; reporting metadata
was augmented during the review remediation:

| Evaluation | Samples/files | Accuracy | Source artifact |
| --- | ---: | ---: | --- |
| Training held-out window split | `1465` windows | `0.9454` | `models/metadata.json` |
| Raw-directory window evaluation | `14,646` windows | `0.9900` | `results/evaluation/metrics.json` |
| External file-level ensemble test | `6` files | `1.0000` | `results/test/metrics.json` |

The external result means all six available class files were predicted
correctly. The file-level 95% Wilson interval is approximately `[0.610,
1.000]`, reflecting that one scored recording per class does not establish a
statistically robust 100% generalization rate.

## Reproducible Commands

```bash
# Recommended: train with independent recording assignments.
python src/train.py \
  --manifest data/recordings_manifest.csv \
  --output-dir models \
  --max-windows-per-class 500

# Recommended: one test prediction per held-out capture and scenario.
python src/test.py \
  --manifest data/recordings_manifest.csv \
  --manifest-split test \
  --model models/best_iot_classifier.h5 \
  --metadata models/metadata.json \
  --output-dir results/scenario_test

# Optional: require the test split to cover all six device classes.
python src/test.py \
  --manifest data/recordings_manifest.csv \
  --manifest-split test \
  --require-all-classes

# Legacy: train and generate model-held-out window reports.
python src/train.py --data-dir data/raw --output-dir models

# Diagnose predictions over all labeled raw-source windows.
python src/evaluate.py \
  --data-dir data/raw \
  --evaluation-role training-source-diagnostic \
  --output-dir results/evaluation

# Evaluate the separate test files at file level.
python src/test.py --data-dir data/test --output-dir results/test

# Run one-file inference.
python src/predict.py --input data/test/miwi_test.npy
```

## Methodological Limitations

- Randomly splitting windows from each continuous source recording can
  overestimate performance on new recordings or devices; use the manifest
  workflow to assign complete recordings before window generation.
- The raw-directory evaluation reuses training-source files and must not be
  cited as an independent validation metric.
- The external test report contains six file-level observations, so it is an
  operational smoke test rather than sufficient performance evidence.
- Compatibility modules now delegate to the active pipeline; notebook-only
  code remains historical context rather than an executable baseline.
