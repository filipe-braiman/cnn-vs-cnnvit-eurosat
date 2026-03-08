
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import json
import os
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score
)


def train_torch_model(
    model,
    train_loader,
    val_loader,
    lr=1e-3,
    epochs=50,
    device=None,
    model_path="pytorch_cnn_baseline.pth",
    scheduler_factor=0.3,
    scheduler_patience=5,
    scheduler_min_lr=1e-6,
    early_stopping_patience=10,
    min_delta=1e-4,
    seed=None
):

    start_time = time.time()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    
    #Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-7)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=scheduler_min_lr
    )

    best_val_loss = float("inf")
    best_epoch = 0
    early_stop_counter = 0

    #History
    history = {
        "loss": [],
        "val_loss": [],
        "accuracy": [],
        "val_accuracy": [],
        "precision": [],
        "recall": [],
        "auc": [],
        "f1": [],
        "val_precision": [],
        "val_recall": [],
        "val_auc": [],
        "val_f1": []
    }

    for epoch in range(epochs):

        # Train
        model.train()

        train_losses = []
        train_preds = []
        train_targets = []
        train_probs = []

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            train_preds.extend(preds.detach().cpu().numpy().ravel())
            train_targets.extend(labels.detach().cpu().numpy().ravel())
            train_probs.extend(probs.detach().cpu().numpy().ravel())

        train_loss = np.mean(train_losses)
        train_acc = accuracy_score(train_targets, train_preds)
        train_precision = precision_score(train_targets, train_preds, zero_division=0)
        train_recall = recall_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds)
        train_auc = roc_auc_score(train_targets, train_probs)

        # Validation
        model.eval()

        val_losses = []
        val_preds = []
        val_targets = []
        val_probs = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_losses.append(loss.item())

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                val_preds.extend(preds.cpu().numpy().ravel())
                val_targets.extend(labels.cpu().numpy().ravel())
                val_probs.extend(probs.cpu().numpy().ravel())

        val_loss = np.mean(val_losses)
        val_acc = accuracy_score(val_targets, val_preds)
        val_precision = precision_score(val_targets, val_preds, zero_division=0)
        val_recall = recall_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds)
        val_auc = roc_auc_score(val_targets, val_probs)

        scheduler.step(val_loss)

        # Early Stopping and Save
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_path)
            print("Saved best model.")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # Store History
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
        history["precision"].append(train_precision)
        history["recall"].append(train_recall)
        history["auc"].append(train_auc)
        history["f1"].append(train_f1)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)
        history["val_auc"].append(val_auc)
        history["val_f1"].append(val_f1)

        # Print Epoch Summary
        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_acc:.4f} "
            f"Val AUC: {val_auc:.4f} "
            f"Val F1: {val_f1:.4f}"
        )

        if early_stop_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

    # Define metadata objects
    total_time = time.time() - start_time
    epochs_ran = len(history["loss"])
    avg_time_per_epoch = total_time / epochs_ran

    param_count = sum(p.numel() for p in model.parameters())
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)

    # Metada schema
    metadata = {
        "framework": "torch",
        "model_name": model_path,
        "total_training_time_sec": total_time,
        "avg_time_per_epoch_sec": avg_time_per_epoch,
        "epochs_ran": epochs_ran,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": history["val_accuracy"][best_epoch - 1] if best_epoch > 0 else None,
        "param_count": param_count,
        "batch_size": train_loader.batch_size if train_loader.batch_size is not None else None,
        "learning_rate": lr,
        "train_dataset_size": train_size,
        "val_dataset_size": val_size,
        "gpu_name": gpu_name,
        "seed_used": seed,
        "run_timestamp": datetime.now().isoformat(),
        "model_size_mb": model_size_mb
    }

    # Save history and metadata .json files
    history_path = model_path.replace(".pth", "_history.json")
    metadata_path = model_path.replace(".pth", "_metadata.json")

    with open(history_path, "w") as f:
        json.dump(history, f)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    print(f"Saved history → {history_path}")
    print(f"Saved metadata → {metadata_path}")

    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Loaded best model with val_loss =", best_val_loss)

    return model, history
