import numpy as np
import sys
import os

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocessing import segment_signal, normalize_windows

def test_segment_signal():
    data = np.arange(8192 + 10)  # 2 full windows + 10 extra
    windows = segment_signal(data, window_size=4096)
    
    assert windows.shape == (2, 4096)
    assert windows[0, 0] == 0
    assert windows[1, 0] == 4096

def test_normalize_windows():
    windows = np.random.randn(2, 4096) * 10 + 5
    normalized = normalize_windows(windows)
    
    # Check mean is close to 0
    assert np.allclose(np.mean(normalized, axis=1), 0, atol=1e-6)
    # Check std is close to 1
    assert np.allclose(np.std(normalized, axis=1), 1, atol=1e-6)
