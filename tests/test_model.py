import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from config import NUM_CLASSES, NUM_LOCATIONS
from model import build_cnn


def test_build_cnn_default_pooling():
    input_shape = (257, 61, 1)
    model = build_cnn(input_shape, NUM_CLASSES)  # should default to "avg"

    assert model is not None
    # Multi-output model: two output tensors [device, location]
    assert isinstance(model.output_shape, list)
    assert len(model.output_shape) == 2
    assert model.output_shape[0] == (None, NUM_CLASSES)
    assert model.output_shape[1] == (None, NUM_LOCATIONS)
    
    # Assert that the pooling layer is GlobalAveragePooling2D
    pooling_layers = [l for l in model.layers if "global_average_pooling" in l.name]
    assert len(pooling_layers) == 1


def test_build_cnn_pooling_modes():
    input_shape = (257, 61, 1)
    model_avg = build_cnn(input_shape, NUM_CLASSES, pooling="avg")
    model_max = build_cnn(input_shape, NUM_CLASSES, pooling="max")
    model_avgmax = build_cnn(input_shape, NUM_CLASSES, pooling="avgmax")
    model_flat = build_cnn(input_shape, NUM_CLASSES, pooling="flatten")

    for model in [model_avg, model_max, model_avgmax, model_flat]:
        assert isinstance(model.output_shape, list)
        assert len(model.output_shape) == 2
        assert model.output_shape[0] == (None, NUM_CLASSES)
        assert model.output_shape[1] == (None, NUM_LOCATIONS)

    # Check for respective pooling layers
    assert any("global_average_pooling" in l.name for l in model_avg.layers)
    assert any("global_max_pooling" in l.name for l in model_max.layers)
    assert any("concatenate" in l.name for l in model_avgmax.layers)
    assert any("flatten" in l.name for l in model_flat.layers)

    # Parameter size comparisons
    params_avg = model_avg.count_params()
    params_flat = model_flat.count_params()

    # The flat model should have significantly more parameters than the avg model
    assert params_flat > 14_000_000
    assert params_avg < 500_000
    assert params_flat > params_avg * 30


def test_build_cnn_output_heads_named_correctly():
    input_shape = (257, 61, 1)
    model = build_cnn(input_shape, NUM_CLASSES)

    # Check that the model has named output layers "device" and "location"
    layer_names = [l.name for l in model.layers]
    assert "device" in layer_names
    assert "location" in layer_names
