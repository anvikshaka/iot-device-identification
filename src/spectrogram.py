import numpy as np
from scipy import signal as sg

def create_spectrograms(windows):
    """
    Convert 1D RF windows into 2D spectrograms with log compression and normalization.
    """
    specs = []

    for w in windows:
        _, _, Sxx = sg.spectrogram(
            w,
            window="hann",
            nperseg=256,
            noverlap=192,
            nfft=512,
            scaling="spectrum"
        )

        # Log compression
        spec = np.log1p(Sxx)

        # Normalize spectrogram
        spec = (spec - np.mean(spec)) / (
            np.std(spec) + 1e-8
        )

        specs.append(spec)

    specs = np.array(specs)
    return specs[..., np.newaxis]
