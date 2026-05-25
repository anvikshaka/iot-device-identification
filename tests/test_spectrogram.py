import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from spectrogram import create_spectrograms

def test_create_spectrograms():
    windows = np.random.randn(3, 4096)  # 3 random windows
    specs = create_spectrograms(windows)
    
    assert len(specs.shape) == 4
    assert specs.shape[0] == 3
    assert specs.shape[3] == 1  # channel
