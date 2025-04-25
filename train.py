"""
train.py - Training procedures for hierarchical object recognition

This module contains functions for training the hierarchical classification model,
including callbacks and training loop implementation.
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

from data_utils import create_data_generators


def create_callbacks(
    model_path: str,
    patience: int = 10,
    log_dir: Optional[str] = None,
) -> List[tf.keras.callbacks.Callback]:
    """
    Create a list of callbacks for model training.

    Args:
        model_path: Path to save the best model
        patience: Patience for early stopping and LR reduction
        log_dir: Directory for TensorBoard logs

    Returns:
        List of Keras callbacks
    """
    # Create directory for the model if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # ModelCheckpoint to save the best model
    checkpoint = ModelCheckpoint(
        filepath=model_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        mode="min",
        verbose=1,
    )

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )

    # Reduce learning rate when a metric has stopped improving
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=patience // 2,
        min_lr=1e-6,
        verbose=1,
    )

    # Base callbacks list
    callbacks = [checkpoint, early_stopping, reduce_lr]

    # Add TensorBoard callback if log_dir is provided
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

        # Create a unique log directory based on timestamp
        log_dir = os.path.join(log_dir, f"run_{int(time.time())}")

        tensorboard = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq="epoch",
        )

        callbacks.append(tensorboard)

    # Add CSV logger to record training metrics
    csv_log_path = os.path.join(os.path.dirname(model_path), "training_log.csv")
    csv_logger = CSVLogger(csv_log_path, append=True, separator=",")
    callbacks.append(csv_logger)

    return callbacks


# Split long function to reduce local variables


def _prepare_training_data(
    train_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    validation_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
    use_augmentation: bool,
) -> Tuple[tf.keras.utils.Sequence, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Prepare training data and generators.

    Args:
        train_data: Tuple of (x_train, y_train_super, y_train_class)
        validation_data: Tuple of (x_val, y_val_super, y_val_class)
        batch_size: Batch size for training
        use_augmentation: Whether to use data augmentation

    Returns:
        Tuple of (train_generator, validation_x, validation_y)
    """
    # Unpack data
    x_train, y_train_super, y_train_class = train_data
    x_val, y_val_super, y_val_class = validation_data

    # Create data generators for training
    if use_augmentation:
        train_generator = create_data_generators(
            x_train,
            y_train_super,
            y_train_class,
            batch_size=batch_size,
            augment=True,
        )
    else:
        train_generator = None

    # Prepare validation data format
    validation_x = x_val
    validation_y = {
        "superclass_output": y_val_super,
        "class_output": y_val_class,
    }

    # Prepare training data format (when not using generator)
    training_y = {
        "superclass_output": y_train_super,
        "class_output": y_train_class,
    }

    return train_generator, training_y, validation_y


def train_model(
    model: tf.keras.Model,
    train_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    validation_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    batch_size: int = 32,
    epochs: int = 50,
    model_path: str = "models/hierarchical_model.h5",
    use_augmentation: bool = True,
    log_dir: Optional[str] = "logs/",
    class_weights: Optional[Dict[str, Dict[int, float]]] = None,
) -> tf.keras.callbacks.History:
    """
    Train the hierarchical classification model.

    Args:
        model: The model to train
        train_data: Tuple of (x_train, y_train_super, y_train_class)
        validation_data: Tuple of (x_val, y_val_super, y_val_class)
        batch_size: Batch size for training
        epochs: Number of training epochs
        model_path: Path to save the best model
        use_augmentation: Whether to use data augmentation
        log_dir: Directory for TensorBoard logs
        class_weights: Optional class weights for handling imbalanced data

    Returns:
        Training history object
    """
    # Create callbacks
    callbacks = create_callbacks(model_path, patience=10, log_dir=log_dir)

    # Prepare data
    train_generator, training_y, validation_y = _prepare_training_data(
        train_data, validation_data, batch_size, use_augmentation
    )

    # Unpack data for clarity
    x_train, _, _ = train_data
    x_val, _, _ = validation_data

    # Train using the appropriate method
    if use_augmentation and train_generator is not None:
        # Train using the generator
        history = model.fit(
            train_generator,
            epochs=epochs,
            steps_per_epoch=len(x_train) // batch_size,
            validation_data=(x_val, validation_y),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1,
        )
    else:
        # Train without augmentation
        history = model.fit(
            x=x_train,
            y=training_y,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, validation_y),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1,
        )

    # Save training history to JSON
    save_training_history(
        history, os.path.join(os.path.dirname(model_path), "training_history.json")
    )

    return history


def calculate_class_weights(
    y_train: np.ndarray, y_train_super: np.ndarray
) -> Dict[str, Dict[int, float]]:
    """
    Calculate class weights to handle class imbalance.

    Args:
        y_train: Original class labels (integers)
        y_train_super: Superclass labels (one-hot encoded)

    Returns:
        Dictionary of class weights for each output
    """
    # Calculate class weights for fine-grained classes
    class_counts = np.bincount(y_train)
    n_samples = len(y_train)

    class_weights = {}
    for i, count in enumerate(class_counts):
        # More weight for less frequent classes
        class_weights[i] = n_samples / (len(class_counts) * count)

    # Calculate weights for superclasses
    superclass_indices = np.argmax(y_train_super, axis=1)
    superclass_counts = np.bincount(superclass_indices)

    superclass_weights = {}
    for i, count in enumerate(superclass_counts):
        superclass_weights[i] = n_samples / (len(superclass_counts) * count)

    # Return weights for both outputs
    return {
        "superclass_output": superclass_weights,
        "class_output": class_weights,
    }


def save_training_history(history: tf.keras.callbacks.History, file_path: str) -> None:
    """
    Save training history to a JSON file.

    Args:
        history: Keras training history object
        file_path: Path to save the history
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Convert history.history to a serializable format
    # (numpy arrays aren't directly JSON serializable)
    history_dict = {}
    for key, values in history.history.items():
        history_dict[key] = [float(val) for val in values]

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(history_dict, f, indent=4)
        print(f"Training history saved to {file_path}")
    except IOError as e:
        print(f"Error saving training history: {e}")


def load_training_history(file_path: str) -> Dict[str, List[float]]:
    """
    Load training history from a JSON file.

    Args:
        file_path: Path to the saved history file

    Returns:
        Dictionary containing training history

    Raises:
        FileNotFoundError: If the history file doesn't exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Training history file not found: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            history_dict = json.load(f)
        # Explicitly cast to satisfy mypy
        return {k: [float(v) for v in vals] for k, vals in history_dict.items()}
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error loading training history: {e}")
        raise
