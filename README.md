# IoT Device Classification

Machine learning project to classify IoT devices based on their RF signals. The model takes raw RF signals, segments them, converts them to spectrograms, and trains a Convolutional Neural Network (CNN) to classify the devices into 6 categories:
- dooralarm
- lora
- microphone
- mbus
- sigfox
- miwi

## Directory Structure
- `data/`: Raw and test data (.npy format)
- `src/`: Source code for data loading, preprocessing, model architecture, training, and evaluation
- `notebooks/`: Original exploratory Jupyter notebooks
- `saved_models/`: Serialized Keras models
- `results/`: Training curves, confusion matrices, and logs
- `docs/`: Project documentation
- `tests/`: Unit tests for the pipeline

## Setup
```bash
pip install -r requirements.txt
```

## Running
To train the model:
```bash
python src/train.py
```

To evaluate the model:
```bash
python src/evaluate.py
```

To predict using ensemble sliding window on a test file:
```bash
python src/predict.py
```

To test using developed model on all test files:
```bash
python src/test.py
```