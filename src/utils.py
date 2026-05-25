import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sg

def plot_training_curves(history, save_path=None):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Loss")
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_spectrogram(signal, title, save_path=None):
    f, t, Sxx = sg.spectrogram(signal, nperseg=256, noverlap=128)
    Sxx_log = np.log1p(Sxx)

    plt.figure(figsize=(5,4))
    plt.pcolormesh(t, f, Sxx_log, shading='gouraud')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
