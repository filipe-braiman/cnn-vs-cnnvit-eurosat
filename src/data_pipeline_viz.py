
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import torch

# RAW DATASET VISUALIZATION
def show_samples_per_class(
    file_array,
    label_array,
    class_map,
    samples_per_class=3,
    seed=None,
    figsize=(12, 6)
):
    """
    Displays random samples per class from file paths.
    """

    if seed is not None:
        random.seed(seed)

    inv_class_map = {v: k for k, v in class_map.items()}
    unique_labels = np.unique(label_array)

    plt.figure(figsize=figsize)
    plot_idx = 1

    for label in unique_labels:
        indices = np.where(label_array == label)[0]

        selected_indices = random.sample(
            list(indices),
            min(samples_per_class, len(indices))
        )

        for idx in selected_indices:
            img = Image.open(file_array[idx])

            plt.subplot(len(unique_labels), samples_per_class, plot_idx)
            plt.imshow(img)
            plt.title(inv_class_map[label])
            plt.axis("off")

            plot_idx += 1

    plt.tight_layout()
    plt.show()


# FRAMEWORK COMPARISON VISUALIZATION
def compare_framework_batches(
    keras_dataset,
    torch_dataloader,
    n_images=4,
    figsize=(12, 6)
):
    """
    Displays side-by-side comparison of one batch from:
    - Keras (channels last)
    - PyTorch (channels first)

    Helps visually inspect augmentation and tensor format differences.
    """

    import torch

    # Get one batch from each
    keras_images, _ = next(iter(keras_dataset))
    torch_images, _ = next(iter(torch_dataloader))

    # Ensure tensors are numpy for plotting
    keras_images = keras_images.numpy()
    torch_images = torch_images.detach().cpu()

    plt.figure(figsize=figsize)

    for i in range(n_images):

        # ---- KERAS IMAGE ----
        plt.subplot(2, n_images, i + 1)
        plt.imshow(keras_images[i])
        plt.title("Keras")
        plt.axis("off")

        # ---- TORCH IMAGE ----
        img_torch = torch_images[i].permute(1, 2, 0).numpy()

        plt.subplot(2, n_images, n_images + i + 1)
        plt.imshow(img_torch)
        plt.title("PyTorch")
        plt.axis("off")

    plt.suptitle("Framework Augmentation Comparison")
    plt.tight_layout()
    plt.show()

    # ---- Print structural info ----
    print("Keras batch shape :", keras_images.shape)
    print("PyTorch batch shape:", torch_images.shape)
    print("\nChannel format difference:")
    print("Keras  -> (B, H, W, C)")
    print("PyTorch-> (B, C, H, W)")
