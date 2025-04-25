"""
data_utils.py - Dataset handling for hierarchical object recognition

This module handles loading, preprocessing, and organizing the CIFAR-10 dataset
into a hierarchical structure for multi-level classification.
"""

import os
from typing import Any, Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# CIFAR-10 class names
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Define hierarchical categorization of CIFAR-10 classes
# Superclasses: vehicles, animals
SUPERCLASS_MAPPING = {
    "vehicles": ["airplane", "automobile", "ship", "truck"],
    "animals": ["bird", "cat", "deer", "dog", "frog", "horse"],
}

# Create an inverse mapping from class to superclass
CLASS_TO_SUPERCLASS = {}
for superclass, classes in SUPERCLASS_MAPPING.items():
    for class_name in classes:
        CLASS_TO_SUPERCLASS[class_name] = superclass

# Extract unique superclass names
SUPERCLASS_NAMES = list(SUPERCLASS_MAPPING.keys())


def load_cifar10_data(
    data_dir: str = "data",
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load CIFAR-10 dataset.

    Args:
        data_dir: Directory where the dataset will be stored

    Returns:
        Tuple of (train_data, test_data) where each is a tuple of (images, labels)
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    try:
        # Load CIFAR-10 dataset
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        # Normalize pixel values to [0, 1]
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

        # Reshape labels to 1D
        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)

        return (x_train, y_train), (x_test, y_test)

    except Exception as e:
        print(f"Error loading CIFAR-10 dataset: {e}")
        raise


def get_class_name(label_index: int) -> str:
    """
    Convert a class index to its name.

    Args:
        label_index: Index of the class (0-9 for CIFAR-10)

    Returns:
        String name of the class
    """
    if not 0 <= label_index < len(CLASS_NAMES):
        raise ValueError(
            f"Invalid label index: {label_index}. Must be between 0 and {len(CLASS_NAMES)-1}"
        )

    return CLASS_NAMES[label_index]


def get_superclass_name(class_name: str) -> str:
    """
    Get the superclass name for a given class name.

    Args:
        class_name: Name of the class

    Returns:
        Name of the superclass

    Raises:
        ValueError: If class_name is not a valid CIFAR-10 class
    """
    if class_name not in CLASS_TO_SUPERCLASS:
        raise ValueError(f"Invalid class name: {class_name}. Must be one of {CLASS_NAMES}")

    return CLASS_TO_SUPERCLASS[class_name]


def get_superclass_index(class_name: str) -> int:
    """
    Get the superclass index for a given class name.

    Args:
        class_name: Name of the class

    Returns:
        Index of the superclass
    """
    superclass_name = get_superclass_name(class_name)
    return SUPERCLASS_NAMES.index(superclass_name)


def prepare_hierarchical_data(
    x_data: np.ndarray, y_data: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare hierarchical labels for the dataset.

    Args:
        x_data: Input images (normalized)
        y_data: Original class labels (integers)

    Returns:
        Tuple of (x_data, y_superclass, y_class) where:
            x_data: Input images
            y_superclass: One-hot encoded superclass labels
            y_class: One-hot encoded class labels
    """
    # Convert class indices to class names
    class_names = [get_class_name(label) for label in y_data]

    # Map class names to superclass indices
    superclass_indices = [get_superclass_index(class_name) for class_name in class_names]

    # Convert to numpy arrays
    superclass_indices = np.array(superclass_indices)

    # Create one-hot encodings
    y_superclass = to_categorical(superclass_indices, num_classes=len(SUPERCLASS_NAMES))
    y_class = to_categorical(y_data, num_classes=len(CLASS_NAMES))

    return x_data, y_superclass, y_class


def create_data_generators(
    x_train: np.ndarray,
    y_train_super: np.ndarray,
    y_train_class: np.ndarray,
    batch_size: int = 32,
    augment: bool = True,
) -> tf.keras.preprocessing.image.ImageDataGenerator:
    """
    Create data generators for training with optional augmentation.

    Args:
        x_train: Training images
        y_train_super: Superclass labels (one-hot encoded)
        y_train_class: Class labels (one-hot encoded)
        batch_size: Batch size for training
        augment: Whether to apply data augmentation

    Returns:
        Data generator that yields (inputs, [superclass_targets, class_targets])
    """
    if augment:
        # Create data generator with augmentation
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
        )
    else:
        # Create data generator without augmentation
        datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    # Prepare the data generator
    generator = datagen.flow(
        x=x_train,
        y={
            "superclass_output": y_train_super,
            "class_output": y_train_class,
        },
        batch_size=batch_size,
    )

    return generator


def get_dataset_stats() -> Dict[str, Any]:
    """
    Get statistics about the hierarchical dataset.

    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        "num_classes": len(CLASS_NAMES),
        "num_superclasses": len(SUPERCLASS_NAMES),
        "classes": CLASS_NAMES,
        "superclasses": SUPERCLASS_NAMES,
        "hierarchy": SUPERCLASS_MAPPING,  # Fixed unnecessary comprehension
    }

    return stats
