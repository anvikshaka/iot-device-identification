import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from model import build_model
from config import NUM_CLASSES

def test_build_model():
    # Mock input shape (height, width, channels)
    input_shape = (257, 61, 1)
    model = build_model(input_shape)
    
    assert model is not None
    assert model.output_shape == (None, NUM_CLASSES)
