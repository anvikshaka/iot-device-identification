"""CNN model definition."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GaussianNoise,
    Input,
    MaxPooling2D,
    ReLU,
)
from tensorflow.keras.models import Sequential


def build_cnn(
    input_shape: tuple[int, int, int],
    num_classes: int,
    learning_rate: float = 3e-4,
) -> tf.keras.Model:
    """Build and compile the spectrogram CNN from the notebook."""
    model = Sequential(
        [
            Input(shape=input_shape),
            GaussianNoise(0.003),
            Conv2D(32, (3, 3), padding="same"),
            BatchNormalization(),
            ReLU(),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), padding="same"),
            BatchNormalization(),
            ReLU(),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), padding="same"),
            BatchNormalization(),
            ReLU(),
            MaxPooling2D((2, 2)),
            Conv2D(256, (3, 3), padding="same"),
            BatchNormalization(),
            ReLU(),
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.25),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

