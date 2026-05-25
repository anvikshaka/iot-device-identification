import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf

from config import classes, data_files, NUM_CLASSES
from data_loader import load_data
from preprocessing import segment_signal, normalize_windows
from spectrogram import create_spectrograms

def build_dataset():
    all_specs = []
    all_labels = []

    for idx, cls in enumerate(classes):
        raw = load_data(data_files[cls])

        # Segment
        windows = segment_signal(raw)

        # Normalize each RF window
        windows = normalize_windows(windows)

        # Spectrograms
        specs = create_spectrograms(windows)

        all_specs.append(specs)
        all_labels.append(np.full(len(specs), idx))

    # Balance dataset (equal samples/class)
    min_samples = min(len(s) for s in all_specs)

    X = np.concatenate([s[:min_samples] for s in all_specs])
    y = np.concatenate([l[:min_samples] for l in all_labels])

    X, y = shuffle(X, y, random_state=42)
    return X, y

def get_train_val_test_splits(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    y_train_cat = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val_cat = tf.keras.utils.to_categorical(y_val, NUM_CLASSES)
    y_test_cat = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

    return (X_train, y_train_cat), (X_val, y_val_cat), (X_test, y_test_cat), y_test
