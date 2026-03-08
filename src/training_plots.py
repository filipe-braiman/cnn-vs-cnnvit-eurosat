
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import torch

# HISTORY NORMALIZATION
def normalize_history(history, framework="keras"):
    """
    Converts Keras History object or dict into unified dictionary format:
    {
        "accuracy": [...],
        "val_accuracy": [...],
        "loss": [...],
        "val_loss": [...]
    }
    """

    if framework == "keras":
        # If it's a live History object
        if hasattr(history, "history"):
            hist = history.history
        else:
            # Already a JSON-loaded dict
            hist = history

    elif framework == "torch":
        hist = history

    else:
        raise ValueError("framework must be 'keras' or 'torch'")

    return {
        "accuracy": hist["accuracy"],
        "val_accuracy": hist["val_accuracy"],
        "loss": hist["loss"],
        "val_loss": hist["val_loss"],
    }

# GENERIC TRAINING CURVE PLOT
def plot_training_curves(history, title_prefix="Model"):

    plt.figure(figsize=(8,6))
    plt.plot(history["accuracy"], label="Train Accuracy")
    plt.plot(history["val_accuracy"], label="Val Accuracy")
    plt.legend()
    plt.grid(True)
    plt.title(f"{title_prefix} Accuracy")
    plt.show()

    plt.figure(figsize=(8,6))
    plt.plot(history["loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.legend()
    plt.grid(True)
    plt.title(f"{title_prefix} Loss")
    plt.show()
