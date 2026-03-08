
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import time
import json
import os
import numpy as np
import subprocess
from datetime import datetime


def train_keras_model(model, train_ds, val_ds,
                      lr=1e-3,
                      epochs=50,
                      model_name="keras_cnn.keras",
                      seed=None):

    start_time = time.time()

    # Compiling: loss, optimzer and metrics
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ]
    )

    # Checkpoint, Early Stopping, and LR Reducing
    checkpoint_cb = ModelCheckpoint(
        filepath=model_name,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1
    )

    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        min_delta=1e-4,
        restore_best_weights=True
    )

    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint_cb, early_stop_cb, reduce_lr_cb],
        verbose=1
    )

    
    # Define metadata objects
    total_time = time.time() - start_time
    epochs_ran = len(history.history["loss"])
    avg_time_per_epoch = total_time / epochs_ran

    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    best_val_loss = float(np.min(history.history["val_loss"]))
    best_val_accuracy = float(
        history.history["val_accuracy"][best_epoch - 1]
    )

    param_count = model.count_params()
    model_size_mb = os.path.getsize(model_name) / (1024 * 1024)

    # GPU name extraction
    try:
        gpu_name = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
        ).decode("utf-8").strip()
    except:
        gpu_name = "CPU"

    # Proper batch size extraction from dataset
    try:
        for x, _ in train_ds.take(1):
            batch_size = int(x.shape[0])
    except:
        batch_size = None

    train_size = sum(1 for _ in train_ds.unbatch())
    val_size = sum(1 for _ in val_ds.unbatch())

    # Metadata schema
    metadata = {
        "framework": "keras",
        "model_name": model_name,
        "total_training_time_sec": total_time,
        "avg_time_per_epoch_sec": avg_time_per_epoch,
        "epochs_ran": epochs_ran,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
        "param_count": param_count,
        "batch_size": batch_size,
        "learning_rate": lr,
        "train_dataset_size": train_size,
        "val_dataset_size": val_size,
        "gpu_name": gpu_name,
        "seed_used": seed,
        "run_timestamp": datetime.now().isoformat(),
        "model_size_mb": model_size_mb
    }

    # Save history and metadata .json files
    history_path = model_name.replace(".keras", "_history.json")
    metadata_path = model_name.replace(".keras", "_metadata.json")

    with open(history_path, "w") as f:
        json.dump(history.history, f)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    print(f"Saved history → {history_path}")
    print(f"Saved metadata → {metadata_path}")

    return history
