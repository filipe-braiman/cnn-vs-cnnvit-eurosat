
import numpy as np
import torch
import tensorflow as tf

# KERAS LOADING
# Rebuild architecture → Load weights only
def _load_keras_weights(model_builder, weights_path, input_shape=(64, 64, 3)):
    """
    Rebuild architecture and load weights only.
    Avoids serialization errors with custom layers.
    """
    model = model_builder(input_shape=input_shape)
    model.load_weights(weights_path)
    model.trainable = False  # ensure inference mode
    return model


def _evaluate_keras_model(model, dataset, threshold=0.5):
    y_true = []
    y_proba = []

    for images, labels in dataset:
        probs = model(images, training=False)  # force inference
        probs = probs.numpy().ravel()

        y_proba.extend(probs)
        y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_proba = np.array(y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    return y_true, y_proba, y_pred


# PYTORCH LOADING
def _load_torch_model(model_class, weights_path, device):
    model = model_class().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def _evaluate_torch_model(model, dataloader, device, threshold=0.5):
    y_true = []
    y_proba = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)

            y_proba.extend(probs.cpu().numpy().ravel())
            y_true.extend(labels.cpu().numpy())

    y_true = np.array(y_true)
    y_proba = np.array(y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    return y_true, y_proba, y_pred


# PUBLIC FUNCTION
def evaluate_models(
    cnn_keras_weights,
    hybrid_keras_weights,
    cnn_torch_weights,
    hybrid_torch_weights,
    keras_val_dataset,
    torch_val_loader,
    build_keras_cnn,
    build_keras_hybrid_model,
    CNNBaseline,
    TorchHybridModel,
    device=None
):
    """
    Fully robust unified evaluation.

    You only provide:
        - weight paths
        - dataset/dataloader
        - architecture builders
        - torch classes
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}

    # KERAS CNN
    cnn_keras = _load_keras_weights(
        build_keras_cnn,
        cnn_keras_weights
    )

    y_true, y_proba, y_pred = _evaluate_keras_model(
        cnn_keras,
        keras_val_dataset
    )

    results["CNN_Keras"] = {
        "y_true": y_true,
        "y_proba": y_proba,
        "y_pred": y_pred
    }

    # KERAS HYBRID
    hybrid_keras = _load_keras_weights(
        build_keras_hybrid_model,
        hybrid_keras_weights
    )

    y_true, y_proba, y_pred = _evaluate_keras_model(
        hybrid_keras,
        keras_val_dataset
    )

    results["Hybrid_Keras"] = {
        "y_true": y_true,
        "y_proba": y_proba,
        "y_pred": y_pred
    }

    # TORCH CNN
    cnn_torch = _load_torch_model(
        CNNBaseline,
        cnn_torch_weights,
        device
    )

    y_true, y_proba, y_pred = _evaluate_torch_model(
        cnn_torch,
        torch_val_loader,
        device
    )

    results["CNN_Torch"] = {
        "y_true": y_true,
        "y_proba": y_proba,
        "y_pred": y_pred
    }

    # TORCH HYBRID
    hybrid_torch = _load_torch_model(
        TorchHybridModel,
        hybrid_torch_weights,
        device
    )

    y_true, y_proba, y_pred = _evaluate_torch_model(
        hybrid_torch,
        torch_val_loader,
        device
    )

    results["Hybrid_Torch"] = {
        "y_true": y_true,
        "y_proba": y_proba,
        "y_pred": y_pred
    }

    return results
