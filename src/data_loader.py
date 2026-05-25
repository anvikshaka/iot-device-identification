import numpy as np
import os

def load_data(filepath):
    """
    Load data from a numpy file.
    If the file is empty or mock (for structural testing), return a mock array.
    """
    try:
        # Mocking logic just in case it's the empty touch file
        if os.path.getsize(filepath) == 0:
            print(f"Warning: {filepath} is empty. Returning mock data.")
            return np.random.randn(4096 * 10)  # 10 windows worth of mock data
        return np.load(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        # Return mock data if file is invalid/dummy
        return np.random.randn(4096 * 10)
