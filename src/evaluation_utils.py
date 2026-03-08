
import json
import pandas as pd
from pathlib import Path



# Internal helper: clean GPU name
def _clean_gpu_name(gpu_name: str):
    """
    Cleans GPU name string (e.g., removes duplicates like 'Tesla T4\\nTesla T4').
    """
    if not isinstance(gpu_name, str):
        return gpu_name

    lines = [line.strip() for line in gpu_name.split("\n") if line.strip()]
    return lines[0] if lines else "Unknown"


# Load single metadata JSON
def load_metadata(metadata_path):
    """
    Loads a single metadata JSON file and returns a dictionary.
    """
    metadata_path = Path(metadata_path)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    metadata["gpu_name"] = _clean_gpu_name(metadata.get("gpu_name"))

    return metadata

# Infer architecture from model name
def _infer_architecture(model_name: str):
    """
    Infers architecture type from model name.
    """
    name = model_name.lower()

    if "hybrid" in name or "vit" in name:
        return "Hybrid"
    elif "cnn" in name:
        return "CNN"
    else:
        return "Unknown"

# Build unified metadata dataframe
def build_metadata_dataframe(metadata_paths, save_path=None):
    """
    Given a list of metadata JSON paths,
    returns a clean pandas DataFrame summarizing all experiments.
    """

    records = []

    for path in metadata_paths:
        meta = load_metadata(path)

        architecture = _infer_architecture(meta.get("model_name", ""))
        framework = meta.get("framework", "Unknown").capitalize()

        model_id = f"{architecture}_{framework}"

        record = {
            "model_id": model_id,
            "architecture": architecture,
            "framework": framework,
            "best_val_accuracy": meta.get("best_val_accuracy"),
            "best_val_loss": meta.get("best_val_loss"),
            "best_epoch": meta.get("best_epoch"),
            "epochs_ran": meta.get("epochs_ran"),
            "param_count": meta.get("param_count"),
            "batch_size": meta.get("batch_size"),
            "learning_rate": meta.get("learning_rate"),
            "train_dataset_size": meta.get("train_dataset_size"),
            "val_dataset_size": meta.get("val_dataset_size"),
            "total_training_time_sec": meta.get("total_training_time_sec"),
            "avg_time_per_epoch_sec": meta.get("avg_time_per_epoch_sec"),
            "model_size_mb": meta.get("model_size_mb"),
            "gpu_name": meta.get("gpu_name"),
            "seed_used": meta.get("seed_used"),
            "run_timestamp": meta.get("run_timestamp"),
        }

        records.append(record)

    df = pd.DataFrame(records)

    # Sort cleanly
    df = df.sort_values(by=["architecture", "framework"]).reset_index(drop=True)
    df = df.set_index("model_id").T
    df.columns.name = None

    # Optional save
    if save_path:
        df.to_csv(save_path, index=True)

    return df

#==============================================================================#

import json
import matplotlib.pyplot as plt
import numpy as np


def plot_framework_comparison(
    keras_metrics_path,
    torch_metrics_path,
    title="Model Comparison",
    save_path=None
):
    """
    Plots validation Accuracy, AUC and Loss for Keras vs PyTorch.
    Produces 6 lines in a single clear scientific plot.
    """
    
    plt.style.use("seaborn-v0_8-whitegrid")
    
    # Load metrics
    with open(keras_metrics_path, "r") as f:
        keras = json.load(f)

    with open(torch_metrics_path, "r") as f:
        torch = json.load(f)

    # Extract metrics
    k_acc = keras["val_accuracy"]
    k_auc = keras["val_auc"]
    k_loss = keras["val_loss"]

    t_acc = torch["val_accuracy"]
    t_auc = torch["val_auc"]
    t_loss = torch["val_loss"]

    k_epochs = len(k_acc)
    t_epochs = len(t_acc)

    k_x = np.arange(1, k_epochs + 1)
    t_x = np.arange(1, t_epochs + 1)

    # Plot
    plt.figure(figsize=(12, 7))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # Framework colors
    keras_color = "#1f77b4"   # blue
    torch_color = "#d62728"   # red

    # Accuracy & AUC (left axis)
    ax1.plot(k_x, k_acc, color=keras_color, linestyle="-",
             label=f"Keras Acc ({k_epochs} ep)")
    ax1.plot(k_x, k_auc, color=keras_color, linestyle="--",
             label=f"Keras AUC")

    ax1.plot(t_x, t_acc, color=torch_color, linestyle="-",
             label=f"PyTorch Acc ({t_epochs} ep)")
    ax1.plot(t_x, t_auc, color=torch_color, linestyle="--",
             label=f"PyTorch AUC")

    ax1.set_ylabel("Validation Accuracy / AUC", fontsize=12)
    ax1.set_ylim(0.5, 1.01)
    ax1.set_yticks(np.linspace(0.5, 1.0, 11))

    # Loss (right axis)
    ax2.plot(k_x, k_loss, color=keras_color, linestyle=":",
             label="Keras Loss")
    ax2.plot(t_x, t_loss, color=torch_color, linestyle=":",
             label="PyTorch Loss")

    ax2.set_ylabel("Validation Loss", fontsize=12)

    # X-axis
    max_epochs = max(k_epochs, t_epochs)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_xlim(1, max_epochs)
    ax1.set_xticks(np.arange(1, max_epochs + 1, 2))

    # Grid
    ax1.grid(True, which="both", linestyle="--", alpha=0.4)

    # Legend (combine both axes)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="best",
        fontsize=10
    )

    plt.title(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

#==============================================================================#

def plot_architecture_comparison(
    cnn_keras_path,
    cnn_torch_path,
    hybrid_keras_path,
    hybrid_torch_path,
    title="Architecture Comparison: CNN vs Hybrid",
    save_path=None
):
    import json
    import numpy as np
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")

    # Load
    def load(path):
        with open(path, "r") as f:
            return json.load(f)

    cnn_k = load(cnn_keras_path)
    cnn_t = load(cnn_torch_path)
    hyb_k = load(hybrid_keras_path)
    hyb_t = load(hybrid_torch_path)

    # Extract
    models = {
        "CNN_Keras": cnn_k,
        "CNN_Torch": cnn_t,
        "Hybrid_Keras": hyb_k,
        "Hybrid_Torch": hyb_t
    }

    # Colors per MODEL
    model_colors = {
        "CNN_Keras": "#1f77b4",     # blue
        "CNN_Torch": "#9467bd",     # purple
        "Hybrid_Keras": "#ff7f0e",  # orange
        "Hybrid_Torch": "#d62728"   # red
    }

    # Plot
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 10), sharex=True
    )

    for name, data in models.items():
        epochs = len(data["val_accuracy"])
        x = np.arange(1, epochs + 1)

        color = model_colors[name]
        lw = 1.0

        # Accuracy & AUC
        ax1.plot(
            x, data["val_accuracy"],
            color=color, linestyle="-",
            linewidth=lw,
            label=f"{name} Acc"
        )

        ax1.plot(
            x, data["val_auc"],
            color=color, linestyle="--",
            linewidth=lw,
            label=f"{name} AUC"
        )

        # Loss
        ax2.plot(
            x, data["val_loss"],
            color=color, linestyle="-",
            linewidth=lw,
            label=f"{name} Loss"
        )

    # Formatting
    max_epochs=max(x)
    ax1.set_ylabel("Validation Accuracy / AUC")
    ax1.set_ylim(0.5, 1.01)
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2.set_ylabel("Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.set_xlim(1, max_epochs)
    ax2.set_xticks(np.arange(1, max_epochs + 1, 2))

    ax1.legend(loc="lower right", fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)

    plt.suptitle(title, fontsize=15, weight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

#==============================================================================#

def plot_generalization_gap(
    cnn_keras_path,
    cnn_torch_path,
    hybrid_keras_path,
    hybrid_torch_path,
    metric="accuracy",
    title="Generalization Gap Over Epochs",
    save_path=None
):
    import json
    import numpy as np
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")

    # Load
    def load(path):
        with open(path, "r") as f:
            return json.load(f)

    cnn_k = load(cnn_keras_path)
    cnn_t = load(cnn_torch_path)
    hyb_k = load(hybrid_keras_path)
    hyb_t = load(hybrid_torch_path)

    # Select metric
    train_key = f"{metric}"
    val_key = f"val_{metric}"

    models = {
        "CNN_Keras": cnn_k,
        "CNN_Torch": cnn_t,
        "Hybrid_Keras": hyb_k,
        "Hybrid_Torch": hyb_t
    }

    # Color scheme
    colors = {
        "Keras": "#1f77b4",
        "Torch": "#d62728"
    }

    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)

    architecture_groups = {
        "CNN": ["CNN_Keras", "CNN_Torch"],
        "CNN-ViT Hybrid": ["Hybrid_Keras", "Hybrid_Torch"]
    }

    for ax, (arch, model_names) in zip(axes, architecture_groups.items()):

        for name in model_names:
            data = models[name]
            epochs = len(data[val_key])
            x = np.arange(1, epochs + 1)

            gap = np.array(data[train_key]) - np.array(data[val_key])

            framework = "Keras" if "Keras" in name else "Torch"

            ax.plot(
                x,
                gap,
                label=name,
                color=colors[framework],
                linewidth=2.5
            )

        max_epochs=max(x)
        ax.axhline(0, color="black", linewidth=1.2, linestyle="--")
        ax.set_title(f"{arch} Architecture", fontsize=13, weight="bold")
        ax.set_ylabel("Generalization Gap")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
        ax.set_xlim(1, max_epochs)
        ax.set_xticks(np.arange(1, max_epochs + 1, 2))

    axes[1].set_xlabel("Epoch")

    plt.suptitle(f"{title} ({metric.capitalize()})", fontsize=15, weight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

#==============================================================================#

# GENERALIZATION GAP TABLE
def compute_gap_at_best_epoch(
    cnn_keras_path,
    cnn_torch_path,
    hybrid_keras_path,
    hybrid_torch_path,
    save_path=None
):
    import json
    import numpy as np
    import pandas as pd

    # Load
    def load(path):
        with open(path, "r") as f:
            return json.load(f)

    models = {
        "CNN_Keras": load(cnn_keras_path),
        "CNN_Torch": load(cnn_torch_path),
        "Hybrid_Keras": load(hybrid_keras_path),
        "Hybrid_Torch": load(hybrid_torch_path)
    }

    results = []

    for name, data in models.items():

        train_loss = np.array(data["loss"])
        val_loss = np.array(data["val_loss"])
        train_acc = np.array(data["accuracy"])
        val_acc = np.array(data["val_accuracy"])

        # Best epoch = lowest validation loss
        best_epoch = int(np.argmin(val_loss))

        # Absolute gaps
        gap_loss = train_loss[best_epoch] - val_loss[best_epoch]
        gap_acc = train_acc[best_epoch] - val_acc[best_epoch]

        # Relative % gaps
        rel_gap_acc = (gap_acc / train_acc[best_epoch]) * 100
        rel_gap_loss = ((val_loss[best_epoch] - train_loss[best_epoch])
                        / val_loss[best_epoch]) * 100

        results.append([
            name,
            best_epoch + 1,
            round(gap_loss, 4),
            round(rel_gap_loss, 2),
            round(gap_acc, 4),
            round(rel_gap_acc, 2)
        ])

    df = pd.DataFrame(
        results,
        columns=[
            "Model",
            "Best Epoch",
            "Gap (Loss)",
            "Rel Gap % (Loss)",
            "Gap (Accuracy)",
            "Rel Gap % (Accuracy)"
        ]
    )
    df=df.set_index("Model")
    df.index.name = None
    
    # Optional save
    if save_path:
        df.to_csv(save_path, index=True)
        
    print("Gap at Best Epoch (Best = Lowest Validation Loss)\n")
    return df

#==============================================================================#

# FINAL METRICS TABLE (SECTION 5.1)
def build_final_metrics_table(
    cnn_keras_path,
    cnn_torch_path,
    hybrid_keras_path,
    hybrid_torch_path,
    save_path=None
):
    """
    Builds final comparison table at BEST EPOCH (lowest validation loss)
    for all four models.

    Automatically computes F1 if not present (e.g., Keras).
    """

    import json
    import numpy as np
    import pandas as pd

    def load(path):
        with open(path, "r") as f:
            return json.load(f)

    models = {
        "CNN_Keras": load(cnn_keras_path),
        "CNN_Torch": load(cnn_torch_path),
        "Hybrid_Keras": load(hybrid_keras_path),
        "Hybrid_Torch": load(hybrid_torch_path)
    }

    records = []

    for name, data in models.items():

        val_loss = np.array(data["val_loss"])
        best_epoch = int(np.argmin(val_loss))

        acc = float(data["val_accuracy"][best_epoch])
        prec = float(data["val_precision"][best_epoch])
        rec = float(data["val_recall"][best_epoch])
        auc = float(data["val_auc"][best_epoch])
        loss = float(data["val_loss"][best_epoch])

        # ---- Compute F1 if not available ----
        if "val_f1" in data:
            f1 = float(data["val_f1"][best_epoch])
        else:
            if (prec + rec) == 0:
                f1 = 0.0
            else:
                f1 = 2 * (prec * rec) / (prec + rec)

        record = {
            "Model": name,
            "Best Epoch": best_epoch + 1,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "AUC": round(auc, 4),
            "Loss": round(loss, 4),
        }

        records.append(record)

    df = pd.DataFrame(records)
    df = df.set_index("Model")
    df = df.sort_index()
    df.index.name = None
    
    if save_path:
        df.to_csv(save_path, index=True)

    return df

#==============================================================================#

# ROC CURVES PLOT
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

plt.style.use("seaborn-v0_8-whitegrid")

model_colors = {
    "CNN_Keras": "#1f77b4",      # blue
    "CNN_Torch": "#9467bd",      # purple
    "Hybrid_Keras": "#ff7f0e",   # orange
    "Hybrid_Torch": "#d62728"    # red
}

def compute_roc_from_results(results_dict):
    """
    Computes ROC curve data from evaluation results dictionary.
    Expects:
        results[model]["y_true"]
        results[model]["y_proba"]
    """
    roc_results = {}

    for model_name, data in results_dict.items():
        y_true = data["y_true"]
        y_proba = data["y_proba"]

        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        roc_results[model_name] = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": roc_auc
        }

        print(f"{model_name} AUC: {roc_auc:.4f}")

    return roc_results

def plot_roc_comparison(roc_results, save_path=None):
    plt.figure(figsize=(8, 6))

    for model_name, data in roc_results.items():
        plt.plot(
            data["fpr"],
            data["tpr"],
            label=f"{model_name} (AUC = {data['auc']:.4f})",
            color=model_colors.get(model_name, "black"),
            linewidth=2
        )

    # Random classifier baseline
    plt.plot(
        [0, 1], [0, 1],
        linestyle="--",
        color="gray",
        linewidth=1,
        label="Random Baseline"
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve Comparison of CNN and Hybrid Models", fontsize=14)

    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

#==============================================================================#

# CONFUSION MATRICES
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

plt.style.use("seaborn-v0_8-whitegrid")

def compute_confusion_matrices(results_dict):
    """
    Computes confusion matrices for all models
    using y_true and y_pred from evaluation results.
    """
    cm_results = {}

    for model_name, data in results_dict.items():
        y_true = data["y_true"]
        y_pred = data["y_pred"]

        cm = confusion_matrix(y_true, y_pred)

        cm_results[model_name] = cm

    return cm_results

def plot_confusion_matrix(cm, model_name, normalize=False, cmap="Blues"):
    """
    Plots a single confusion matrix.
    """

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)

    # Axis labels
    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ylabel="True Label",
        xlabel="Predicted Label",
        title=f"{model_name} Confusion Matrix"
    )

    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=0)

    # Annotate cells
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i,
                format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11
            )

    fig.tight_layout()
    plt.show()

def plot_confusion_matrix_grid(cm_results, normalize=False, save_path=None):
    """
    Plots a 2x2 grid of confusion matrices for all models.
    """

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    for idx, (model_name, cm) in enumerate(cm_results.items()):
        ax = axes[idx]

        if normalize:
            cm_display = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        else:
            cm_display = cm

        im = ax.imshow(cm_display, cmap="Blues")

        ax.set_title(model_name, fontsize=12)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Negative", "Positive"])
        ax.set_yticklabels(["Negative", "Positive"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        fmt = ".2f" if normalize else "d"
        thresh = cm_display.max() / 2.

        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i,
                    format(cm_display[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if cm_display[i, j] > thresh else "black",
                    fontsize=10
                )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

#==============================================================================#

# PARAM COUNT BAR CHART
import torch
import matplotlib.pyplot as plt

def compute_parameter_counts(
    build_keras_cnn,
    build_keras_hybrid_model,
    CNNBaseline,
    TorchHybridModel,
    input_shape=(64, 64, 3),
    device=None
):
    """
    Computes parameter counts directly from architecture definitions.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    param_counts = {}

    # KERAS CNN
    keras_cnn = build_keras_cnn(input_shape=input_shape)
    param_counts["CNN_Keras"] = keras_cnn.count_params()

    # KERAS HYBRID
    keras_hybrid = build_keras_hybrid_model(input_shape=input_shape)
    param_counts["Hybrid_Keras"] = keras_hybrid.count_params()

    # TORCH CNN
    torch_cnn = CNNBaseline().to(device)
    param_counts["CNN_Torch"] = sum(p.numel() for p in torch_cnn.parameters())

    # TORCH HYBRID
    torch_hybrid = TorchHybridModel().to(device)
    param_counts["Hybrid_Torch"] = sum(p.numel() for p in torch_hybrid.parameters())

    return param_counts

plt.style.use("seaborn-v0_8-whitegrid")

model_colors = {
    "CNN_Keras": "#1f77b4",
    "CNN_Torch": "#9467bd",
    "Hybrid_Keras": "#ff7f0e",
    "Hybrid_Torch": "#d62728"
}

def plot_parameter_comparison(param_counts, save_path=None):
    models = list(param_counts.keys())
    values = list(param_counts.values())

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        models,
        values,
        color=[model_colors[m] for m in models]
    )

    plt.ylabel("Number of Parameters")
    plt.title("Model Parameter Count Comparison")

    # Rotate labels slightly for readability
    plt.xticks(rotation=20)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height,
            f"{height:,}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

#==============================================================================#

# TRAINING TIME COMPARISON
import json
import matplotlib.pyplot as plt

def load_training_metadata(json_paths_dict):
    """
    json_paths_dict format:
    {
        "CNN_Keras": "path/to/json",
        "Hybrid_Keras": "path/to/json",
        ...
    }
    """
    
    json_paths = {
    "CNN_Keras": "/kaggle/input/datasets/filipebraiman/baseline-models-training-info/keras_cnn_baseline_metadata.json",
    "Hybrid_Keras": "/kaggle/input/datasets/filipebraiman/hybrid-models-training-info/keras_hybrid_metadata.json",
    "CNN_Torch": "/kaggle/input/datasets/filipebraiman/baseline-models-training-info/pytorch_cnn_baseline_metadata.json",
    "Hybrid_Torch": "/kaggle/input/datasets/filipebraiman/hybrid-models-training-info/pytorch_hybrid_metadata.json"
    }
    
    metadata = {}

    for model_name, path in json_paths_dict.items():
        with open(path, "r") as f:
            data = json.load(f)

        metadata[model_name] = data

    return metadata

def extract_training_times(metadata_dict):
    """
    Extract total training time in minutes for readability.
    """

    training_times = {}

    for model_name, data in metadata_dict.items():
        total_sec = data["total_training_time_sec"]
        total_min = total_sec / 60  # convert to minutes

        training_times[model_name] = total_min

        print(f"{model_name}: {total_min:.2f} minutes")

    return training_times

plt.style.use("seaborn-v0_8-whitegrid")

model_colors = {
    "CNN_Keras": "#1f77b4",
    "CNN_Torch": "#9467bd",
    "Hybrid_Keras": "#ff7f0e",
    "Hybrid_Torch": "#d62728"
}

def plot_training_time_comparison(training_times, save_path=None):
    models = list(training_times.keys())
    values = list(training_times.values())

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        models,
        values,
        color=[model_colors[m] for m in models]
    )

    plt.ylabel("Total Training Time (Minutes)")
    plt.title("Training Time Comparison Across Models")

    plt.xticks(rotation=20)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

#==============================================================================#

# MODEL SIZE BAR SIZE
import matplotlib.pyplot as plt

def extract_model_sizes(metadata_dict):
    """
    Extract model size in MB from metadata.
    """

    model_sizes = {}

    for model_name, data in metadata_dict.items():
        size_mb = data["model_size_mb"]
        model_sizes[model_name] = size_mb

        print(f"{model_name}: {size_mb:.2f} MB")

    return model_sizes

plt.style.use("seaborn-v0_8-whitegrid")

model_colors = {
    "CNN_Keras": "#1f77b4",
    "CNN_Torch": "#9467bd",
    "Hybrid_Keras": "#ff7f0e",
    "Hybrid_Torch": "#d62728"
}

def plot_model_size_comparison(model_sizes, save_path=None):
    models = list(model_sizes.keys())
    values = list(model_sizes.values())

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        models,
        values,
        color=[model_colors[m] for m in models]
    )

    plt.ylabel("Model Size (MB)")
    plt.title("Model Storage Size Comparison")

    plt.xticks(rotation=20)

    # Annotate bar values
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height,
            f"{height:.2f} MB",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

#==============================================================================#

# PERFORMANCE VS EFFICIENCY SCATTER PLOTS
import json
import matplotlib.pyplot as plt
import numpy as np

def extract_best_epoch_metrics(metadata_json_path, history_json_path):
    """
    Returns:
        best_val_auc
        best_val_accuracy
    """

    # Load metadata
    with open(metadata_json_path, "r") as f:
        metadata = json.load(f)

    # Load history
    with open(history_json_path, "r") as f:
        history = json.load(f)

    best_epoch = metadata["best_epoch"] - 1  # convert to 0-index

    best_val_auc = history["val_auc"][best_epoch]
    best_val_accuracy = history["val_accuracy"][best_epoch]

    return best_val_auc, best_val_accuracy


models_info = {
    "CNN_Keras": {
        "metadata": "/kaggle/input/datasets/filipebraiman/baseline-models-training-info/keras_cnn_baseline_metadata.json",
        "history": "/kaggle/input/datasets/filipebraiman/baseline-models-training-info/keras_cnn_baseline_history.json"
    },
    "Hybrid_Keras": {
        "metadata": "/kaggle/input/datasets/filipebraiman/hybrid-models-training-info/keras_hybrid_metadata.json",
        "history": "/kaggle/input/datasets/filipebraiman/hybrid-models-training-info/keras_hybrid_history.json"
    },
    "CNN_Torch": {
        "metadata": "/kaggle/input/datasets/filipebraiman/baseline-models-training-info/pytorch_cnn_baseline_metadata.json",
        "history": "/kaggle/input/datasets/filipebraiman/baseline-models-training-info/pytorch_cnn_baseline_history.json"
    },
    "Hybrid_Torch": {
        "metadata": "/kaggle/input/datasets/filipebraiman/hybrid-models-training-info/pytorch_hybrid_metadata.json",
        "history": "/kaggle/input/datasets/filipebraiman/hybrid-models-training-info/pytorch_hybrid_history.json"
    }
}


def build_tradeoff_metrics(models_info):
    """
    Returns dictionary with:
        AUC
        Accuracy
        Param Count
        Training Time (minutes)
        Model Size (MB)
    """

    tradeoff = {}

    for model_name, paths in models_info.items():

        with open(paths["metadata"], "r") as f:
            metadata = json.load(f)

        best_auc, best_acc = extract_best_epoch_metrics(
            paths["metadata"],
            paths["history"]
        )

        tradeoff[model_name] = {
            "auc": best_auc,
            "accuracy": best_acc,
            "params": metadata["param_count"],
            "training_time_min": metadata["total_training_time_sec"] / 60,
            "model_size_mb": metadata["model_size_mb"]
        }

    return tradeoff

plt.style.use("seaborn-v0_8-whitegrid")

model_colors = {
    "CNN_Keras": "#1f77b4",
    "CNN_Torch": "#9467bd",
    "Hybrid_Keras": "#ff7f0e",
    "Hybrid_Torch": "#d62728"
}


def plot_auc_vs_params(tradeoff_metrics, save_path=None):

    plt.figure(figsize=(8, 6))

    for model_name, data in tradeoff_metrics.items():

        bubble_size = data["training_time_min"] * 30

        plt.scatter(
            data["params"],
            data["auc"],
            s=bubble_size,
            color=model_colors[model_name],
            alpha=0.75,
            edgecolors="black"
        )

        plt.text(
            data["params"],
            data["auc"],
            f"    {model_name}",
            va="center"
        )

    plt.xlabel("Parameter Count")
    plt.ylabel("Validation AUC (Best Epoch)")
    plt.title("AUC vs Parameter Count (Bubble = Training Time)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_accuracy_vs_time(tradeoff_metrics, save_path=None):

    plt.figure(figsize=(8, 6))

    for model_name, data in tradeoff_metrics.items():

        plt.scatter(
            data["training_time_min"],
            data["accuracy"],
            s=200,
            color=model_colors[model_name],
            alpha=0.75,
            edgecolors="black"
        )

        plt.text(
            data["training_time_min"],
            data["accuracy"],
            f"    {model_name}",
            va="center"
        )

    plt.xlabel("Training Time (Minutes)")
    plt.ylabel("Validation Accuracy (Best Epoch)")
    plt.title("Accuracy vs Training Time")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_auc_vs_model_size(tradeoff_metrics, save_path=None):

    plt.figure(figsize=(8, 6))

    for model_name, data in tradeoff_metrics.items():

        bubble_size = data["training_time_min"] * 30

        plt.scatter(
            data["model_size_mb"],
            data["auc"],
            s=bubble_size,
            color=model_colors[model_name],
            alpha=0.75,
            edgecolors="black"
        )

        plt.text(
            data["model_size_mb"],
            data["auc"],
            f"    {model_name}",
            va="center"
        )

    plt.xlabel("Model Size (MB)")
    plt.ylabel("Validation AUC (Best Epoch)")
    plt.title("AUC vs Model Size (Bubble = Training Time)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
