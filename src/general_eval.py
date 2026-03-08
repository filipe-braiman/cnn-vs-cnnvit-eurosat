
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

# KERAS EVALUATION
def evaluate_keras_binary(model, val_ds, threshold=0.5, verbose=True):
    y_true = []
    y_probs = []

    for x_batch, y_batch in val_ds:
        probs = model.predict(x_batch, verbose=0)
        y_true.extend(y_batch.numpy().ravel())
        y_probs.extend(probs.ravel())

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    y_pred = (y_probs > threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_probs),
    }

    if verbose:
        print("\n===== FINAL KERAS VALIDATION METRICS =====")
        for k, v in metrics.items():
            print(f"{k.capitalize():<10}: {v:.4f}")

    return metrics


# PYTORCH EVALUATION
def evaluate_torch_binary(model, val_loader, device=None, threshold=0.5, verbose=True):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    y_true = []
    y_probs = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)

            y_true.extend(labels.cpu().numpy().ravel())
            y_probs.extend(probs.cpu().numpy().ravel())

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    y_pred = (y_probs > threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_probs),
    }

    if verbose:
        print("\n===== FINAL PYTORCH VALIDATION METRICS =====")
        for k, v in metrics.items():
            print(f"{k.capitalize():<10}: {v:.4f}")

    return metrics
