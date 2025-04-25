"""
visualize.py - Visualization utilities for hierarchical object recognition

This module provides functions for visualizing training progress, model predictions,
confusion matrices, and hierarchical classification results.
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
import numpy as np

# Use non-interactive backend to avoid display issues
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from data_utils import CLASS_NAMES, SUPERCLASS_NAMES


def visualize_training_history(
    history: Union[tf.keras.callbacks.History, Dict[str, List[float]]],
    save_dir: str = "visualizations",
    show_plot: bool = False,
) -> str:
    """
    Visualize training history metrics.

    Args:
        history: Keras History object or dictionary containing training metrics
        save_dir: Directory to save visualizations
        show_plot: Whether to show the plot (not recommended in non-interactive environments)

    Returns:
        Path to the saved visualization file
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Convert History object to dictionary if needed
    if isinstance(history, tf.keras.callbacks.History):
        history_dict = history.history
    else:
        history_dict = history

    # Get available metrics
    metrics = [key for key in history_dict.keys() if not key.startswith("val_")]

    # Create a figure with multiple subplots
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)), sharex=True)

    # If there's only one metric, axes won't be an array
    if len(metrics) == 1:
        axes = [axes]

    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Plot training metric
        ax.plot(history_dict[metric], label=f"Training {metric}")

        # Plot validation metric if available
        val_metric = f"val_{metric}"
        if val_metric in history_dict:
            ax.plot(history_dict[val_metric], label=f"Validation {metric}")

        ax.set_title(f"{metric} during training")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True)

    # Set x-axis label on the bottom subplot
    axes[-1].set_xlabel("Epoch")

    # Adjust layout
    plt.tight_layout()

    # Save figure
    save_path = os.path.join(save_dir, "training_history.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # Show plot if requested
    if show_plot:
        plt.show()
        return save_path

    plt.close()
    return save_path


def visualize_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (10, 8),
    normalize: bool = True,
    show_plot: bool = False,
) -> Optional[plt.Figure]:
    """
    Visualize a confusion matrix.

    Args:
        confusion_matrix: Confusion matrix as a 2D numpy array
        class_names: List of class names for axis labels
        save_path: Path to save the visualization (if None, doesn't save)
        title: Plot title
        figsize: Figure size as (width, height) in inches
        normalize: Whether to normalize the confusion matrix
        show_plot: Whether to show the plot

    Returns:
        Matplotlib Figure object if show_plot is True, None otherwise
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Normalize if requested
    if normalize:
        confusion_matrix = (
            confusion_matrix.astype("float") / confusion_matrix.sum(axis=1)[:, np.newaxis]
        )
        fmt = ".2f"
    else:
        fmt = "d"

    # Create heatmap
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )

    # Set labels and title
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title(title)

    # Rotate axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)

    # Adjust layout
    plt.tight_layout()

    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # Show plot if requested
    if show_plot:
        plt.show()
        return fig

    plt.close()
    return None


def _prepare_sample_grid(
    x_data: np.ndarray,
    y_data: np.ndarray,
    class_names: List[str],
    num_samples: int = 9,
) -> Tuple[np.ndarray, List[str], int]:
    """
    Prepare sample grid data for visualization.

    Args:
        x_data: Image data as numpy array
        y_data: Labels as integers
        class_names: List of class names
        num_samples: Number of samples to display

    Returns:
        Tuple of (selected_indices, class_labels, grid_size)
    """
    # Determine grid size
    grid_size = int(np.ceil(np.sqrt(num_samples)))

    # Choose random indices
    num_samples = min(num_samples, len(x_data))
    indices = np.random.choice(len(x_data), num_samples, replace=False)

    # Prepare class labels
    class_labels = []
    for idx in indices:
        class_idx = y_data[idx]
        if isinstance(class_idx, np.ndarray) and len(class_idx.shape) > 0:
            class_idx = np.argmax(class_idx)
        class_labels.append(class_names[class_idx])

    return indices, class_labels, grid_size


def visualize_sample_images(
    x_data: np.ndarray,
    y_data: np.ndarray,
    class_names: List[str],
    num_samples: int = 9,
    save_path: Optional[str] = None,
    title: str = "Sample Images",
    figsize: Tuple[int, int] = (12, 12),
    show_plot: bool = False,
) -> Optional[plt.Figure]:
    """
    Visualize a grid of sample images from the dataset.

    Args:
        x_data: Image data as numpy array
        y_data: Labels as integers
        class_names: List of class names
        num_samples: Number of samples to display
        save_path: Path to save the visualization (if None, doesn't save)
        title: Plot title
        figsize: Figure size as (width, height) in inches
        show_plot: Whether to show the plot

    Returns:
        Matplotlib Figure object if show_plot is True, None otherwise
    """
    # Prepare data
    indices, class_labels, grid_size = _prepare_sample_grid(
        x_data, y_data, class_names, num_samples
    )

    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten()

    # Plot each image
    for i, idx in enumerate(indices):
        ax = axes[i]

        # Display image
        ax.imshow(x_data[idx])

        # Set title
        ax.set_title(class_labels[i])

        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide empty subplots
    for j in range(num_samples, len(axes)):
        axes[j].axis("off")

    # Set main title
    plt.suptitle(title, fontsize=16)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # Show plot if requested
    if show_plot:
        plt.show()
        return fig

    plt.close()
    return None


def visualize_predictions(
    x_test: np.ndarray,
    y_test_super: np.ndarray,
    y_test_class: np.ndarray,
    model: tf.keras.Model,
    num_samples: int = 5,
    save_dir: str = "visualizations",
) -> str:
    """
    Visualize model predictions on a few test samples.

    Args:
        x_test: Test images
        y_test_super: True superclass labels (one-hot encoded)
        y_test_class: True class labels (one-hot encoded)
        model: Trained model
        num_samples: Number of samples to visualize
        save_dir: Directory to save visualizations

    Returns:
        Path to the saved visualization file
    """
    os.makedirs(save_dir, exist_ok=True)

    # Convert one-hot encoded labels to class indices
    y_true_super = np.argmax(y_test_super, axis=1)
    y_true_class = np.argmax(y_test_class, axis=1)

    # Get model predictions
    predictions = model.predict(x_test[:num_samples])
    superclass_preds, class_preds = predictions

    # Convert predictions to class indices
    y_pred_super = np.argmax(superclass_preds, axis=1)
    y_pred_class = np.argmax(class_preds, axis=1)

    # Create figure
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4 * num_samples))

    # If there's only one sample, axes won't be an array
    if num_samples == 1:
        axes = [axes]

    # Plot each sample
    for i in range(num_samples):
        ax = axes[i]

        # Display image
        ax.imshow(x_test[i])

        # Get true and predicted labels
        true_super = SUPERCLASS_NAMES[y_true_super[i]]
        true_class = CLASS_NAMES[y_true_class[i]]

        pred_super = SUPERCLASS_NAMES[y_pred_super[i]]
        pred_class = CLASS_NAMES[y_pred_class[i]]

        # Determine if predictions are correct
        super_correct = y_true_super[i] == y_pred_super[i]
        class_correct = y_true_class[i] == y_pred_class[i]

        # Set title with color-coding
        super_color = "green" if super_correct else "red"
        class_color = "green" if class_correct else "red"

        title = f"True: {true_super} / {true_class}\n"
        title += "Pred: "
        title += rf"$\color{{{super_color}}}{{{pred_super}}}$ / "
        title += rf"$\color{{{class_color}}}{{{pred_class}}}$"

        ax.set_title(title)

        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])

    # Set main title
    plt.suptitle("Model Predictions", fontsize=16)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    # Save figure
    save_path = os.path.join(save_dir, "model_predictions.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return save_path


def _create_feature_model(model: tf.keras.Model, layer_name: str) -> tf.keras.Model:
    """
    Create a model that outputs the feature maps of a specific layer.

    Args:
        model: The main model
        layer_name: Name of the layer to extract features from

    Returns:
        A model that outputs the specified layer's output
    """
    try:
        layer = model.get_layer(layer_name)
        feature_model = tf.keras.Model(inputs=model.input, outputs=layer.output)
        return feature_model
    except ValueError:
        print(f"Layer '{layer_name}' not found in the model. Available layers:")
        for layer in model.layers:
            print(f"  - {layer.name}")
        return None


def visualize_feature_maps(
    model: tf.keras.Model,
    image: np.ndarray,
    layer_name: str,
    save_dir: str = "visualizations",
    max_features: int = 16,
) -> str:
    """
    Visualize feature maps of a specific layer for a given input image.

    Args:
        model: Trained model
        image: Input image (single image, not a batch)
        layer_name: Name of the layer to visualize
        save_dir: Directory to save visualizations
        max_features: Maximum number of feature maps to display

    Returns:
        Path to the saved visualization file
    """
    os.makedirs(save_dir, exist_ok=True)

    # Create a model that outputs the feature maps of the specified layer
    feature_model = _create_feature_model(model, layer_name)
    if feature_model is None:
        return ""

    # Expand dimensions to create a batch with a single image
    image_batch = np.expand_dims(image, axis=0)

    # Get feature maps
    feature_maps = feature_model.predict(image_batch)[0]

    # Determine grid size
    num_features = min(feature_maps.shape[-1], max_features)
    grid_size = int(np.ceil(np.sqrt(num_features)))

    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()

    # Plot each feature map
    for i in range(num_features):
        ax = axes[i]

        # Get feature map
        feature_map = feature_maps[:, :, i]

        # Display feature map
        im = ax.imshow(feature_map, cmap="viridis")

        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Add feature number
        ax.set_title(f"Feature {i+1}")

    # Hide empty subplots
    for j in range(num_features, len(axes)):
        axes[j].axis("off")

    # Add colorbar
    plt.colorbar(im, ax=axes, shrink=0.7)

    # Set main title
    plt.suptitle(f"Feature Maps - Layer: {layer_name}", fontsize=16)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save figure
    save_path = os.path.join(save_dir, f"feature_maps_{layer_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return save_path
