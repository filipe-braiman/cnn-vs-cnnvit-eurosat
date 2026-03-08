
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import tensorflow as tf


# GLOBAL SEED

def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # TF
    tf.random.set_seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    try:
        tf.config.experimental.enable_op_determinism()
    except:
        pass


# BUILD BALANCED FILE LIST

def build_balanced_binary_dataset(
    base_dir: str,
    class_map: dict,
    samples_per_class: int,
    seed: int
):
    set_global_seed(seed)

    all_files = []
    all_labels = []

    for class_name, label in class_map.items():
        class_dir = os.path.join(base_dir, class_name)
        files = [os.path.join(class_dir, f) for f in os.listdir(class_dir)]
        files = random.sample(files, samples_per_class)

        all_files.extend(files)
        all_labels.extend([label] * samples_per_class)

    return np.array(all_files), np.array(all_labels)


# TRAIN / VAL SPLIT

def build_stratified_split(
    files,
    labels,
    val_split: float,
    seed: int
):
    return train_test_split(
        files,
        labels,
        test_size=val_split,
        stratify=labels,
        random_state=seed
    )


# KERAS PIPELINE

def build_keras_datasets(
    train_files,
    train_labels,
    val_files,
    val_labels,
    image_size=(64, 64),
    batch_size=128,
    seed=42
):

    AUTOTUNE = tf.data.AUTOTUNE

    def load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, image_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))

    train_ds = train_ds.map(load_image, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(load_image, num_parallel_calls=AUTOTUNE)

    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomZoom((-0.2, 0.2)),
        tf.keras.layers.RandomContrast(0.2),
    ])

    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=AUTOTUNE
    )

    train_ds = train_ds.shuffle(1000, seed=seed)\
                       .batch(batch_size)\
                       .prefetch(AUTOTUNE)

    val_ds = val_ds.batch(batch_size).prefetch(AUTOTUNE)

    return train_ds, val_ds


# TORCH PIPELINE

class EuroSATDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)


def build_torch_dataloaders(
    train_files,
    train_labels,
    val_files,
    val_labels,
    image_size=(64, 64),
    batch_size=128,
    seed=42,
    num_workers=2
):

    transform_train = transforms.Compose([
        transforms.Pad(10, padding_mode='reflect'),
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(72),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.8, 1.2)
        ),
        transforms.ColorJitter(contrast=0.2),
        transforms.ToTensor(),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    train_dataset = EuroSATDataset(train_files, train_labels, transform_train)
    val_dataset = EuroSATDataset(val_files, val_labels, transform_val)

    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
