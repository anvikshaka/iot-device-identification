"""CNN model definition."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GaussianNoise,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Input,
    MaxPooling2D,
    ReLU,
)


@tf.keras.utils.register_keras_serializable(package="iot")
class StopGradientLayer(tf.keras.layers.Layer):
    """Pass-through layer that blocks gradient flow."""

    def call(self, inputs):
        return tf.stop_gradient(inputs)


def build_cnn(
    input_shape: tuple[int, int, int],
    num_classes: int,
    num_locations: int = 3,
    learning_rate: float = 3e-4,
    pooling: str = "avg",
) -> tf.keras.Model:
    """Build and compile the spectrogram CNN with configurable pooling."""
    if pooling not in {"avg", "max", "avgmax", "flatten"}:
        raise ValueError(f"Unknown pooling type: {pooling}")

    inputs = Input(shape=input_shape)
    x = GaussianNoise(0.003)(inputs)
    x = Conv2D(32, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    if pooling == "flatten":
        x = Flatten()(x)
    elif pooling == "avg":
        x = GlobalAveragePooling2D()(x)
    elif pooling == "max":
        x = GlobalMaxPooling2D()(x)
    else:
        x = Concatenate()([GlobalAveragePooling2D()(x), GlobalMaxPooling2D()(x)])

    shared = Dense(256, activation="relu")(x)
    shared = Dropout(0.25)(shared)
    device_output = Dense(num_classes, activation="softmax", name="device")(shared)

    # Location branch with StopGradientLayer to protect device accuracy from location gradient corruption
    loc_input = StopGradientLayer()(x)
    loc_branch = Dense(128, activation="relu")(loc_input)
    loc_branch = Dropout(0.3)(loc_branch)
    loc_branch = Dense(64, activation="relu")(loc_branch)
    location_output = Dense(num_locations, activation="softmax", name="location")(loc_branch)

    model = tf.keras.Model(inputs=inputs, outputs=[device_output, location_output])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "device": "categorical_crossentropy",
            "location": "categorical_crossentropy",
        },
        loss_weights={"device": 1.0, "location": 0.5},
        metrics={
            "device": ["accuracy"],
            "location": ["accuracy"],
        },
    )
    return model
