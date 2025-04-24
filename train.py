"""
train.py - Training procedures for hierarchical object recognition

This module contains functions for training the hierarchical classification model,
including callbacks and training loop implementation.
"""

import os
import tensorflow as tf
import numpy as np
import time
import json
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)


def create_callbacks(model_name="hierarchical_model",
                     checkpoints_dir="checkpoints",
                     logs_dir="logs"):
    """
    Create a list of callbacks for model training.

    Args:
        model_name: Name of the model for saving files
        checkpoints_dir: Directory to save model checkpoints
        logs_dir: Directory to save training logs

    Returns:
        A list of Keras callbacks
    """
    # Create directories if they don't exist
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Current time for unique filenames
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Model checkpoint to save the best model
    checkpoint_filepath = os.path.join(
        checkpoints_dir,
        f"{model_name}_{timestamp}_best.h5"
    )
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        save_weights_only=False,
        verbose=1
    )

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    # Reduce learning rate when validation loss plateaus
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    # TensorBoard for visualization
    tensorboard = TensorBoard(
        log_dir=os.path.join(logs_dir, f"{model_name}_{timestamp}"),
        histogram_freq=1,
        write_graph=True
    )

    # CSV Logger to save training metrics
    csv_log_file = os.path.join(
        logs_dir,
        f"{model_name}_{timestamp}_training_log.csv"
    )
    csv_logger = CSVLogger(
        csv_log_file,
        append=True,
        separator=','
    )

    # Return all callbacks
    return [
        model_checkpoint,
        early_stopping,
        reduce_lr,
        tensorboard,
        csv_logger
    ]


class HierarchicalLearningRateScheduler(tf.keras.callbacks.Callback):
    """
    Custom learning rate scheduler that adjusts learning rate based on
    both superclass and class accuracy.
    """

    def __init__(self, initial_lr=0.001, decay_factor=0.75,
                 superclass_threshold=0.9, class_threshold=0.85):
        super(HierarchicalLearningRateScheduler, self).__init__()
        self.initial_lr = initial_lr
        self.decay_factor = decay_factor
        self.superclass_threshold = superclass_threshold
        self.class_threshold = class_threshold

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Get current validation accuracies
        superclass_acc = logs.get('val_superclass_output_accuracy', 0)
        class_acc = logs.get('val_class_output_accuracy', 0)

        # Get current learning rate
        current_lr = tf.keras.backend.get_value(self.model.optimizer.lr)

        # Reduce learning rate if both accuracies are above their thresholds
        if superclass_acc > self.superclass_threshold and class_acc > self.class_threshold:
            new_lr = current_lr * self.decay_factor
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
            print(f"\nEpoch {epoch + 1}: Both accuracies above thresholds. "
                  f"Reducing learning rate from {current_lr:.6f} to {new_lr:.6f}")


def train_model(model,
                train_generator,
                val_generator,
                steps_per_epoch,
                validation_steps,
                epochs=50,
                callbacks=None,
                model_name="hierarchical_model",
                save_dir="models"):
    """
    Train the hierarchical model.

    Args:
        model: Compiled Keras model
        train_generator: Generator for training data
        val_generator: Generator for validation data
        steps_per_epoch: Number of batches per epoch
        validation_steps: Number of validation batches
        epochs: Number of training epochs
        callbacks: List of Keras callbacks
        model_name: Name of the model for saving
        save_dir: Directory to save the final model

    Returns:
        Training history and path to the saved model
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Add default callbacks if none provided
    if callbacks is None:
        callbacks = create_callbacks(model_name=model_name)

    # Train the model
    print(f"Starting training for {epochs} epochs...")
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )

    # Save the final model
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(save_dir, f"{model_name}_{timestamp}_final.h5")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Save training history
    history_path = os.path.join(save_dir, f"{model_name}_{timestamp}_history.json")
    with open(history_path, 'w') as f:
        # Convert numpy values to Python native types for JSON serialization
        history_dict = {}
        for key, value in history.history.items():
            history_dict[key] = [float(x) for x in value]
        json.dump(history_dict, f, indent=4)
    print(f"Training history saved to {history_path}")

    return history, model_path


def load_trained_model(model_path):
    """
    Load a trained model from file.

    Args:
        model_path: Path to the saved model file

    Returns:
        Loaded Keras model
    """
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    return model


def create_training_config(epochs=50,
                           batch_size=64,
                           learning_rate=0.001,
                           superclass_weight=0.3,
                           data_augmentation=True,
                           early_stopping_patience=15,
                           model_type="cnn"):
    """
    Create a configuration dictionary for training parameters.

    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        superclass_weight: Weight for superclass loss
        data_augmentation: Whether to use data augmentation
        early_stopping_patience: Number of epochs with no improvement before stopping
        model_type: Type of model ("cnn" or "resnet")

    Returns:
        Dictionary of training parameters
    """
    config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "superclass_weight": superclass_weight,
        "data_augmentation": data_augmentation,
        "early_stopping_patience": early_stopping_patience,
        "model_type": model_type,
        "timestamp": time.strftime("%Y%m%d-%H%M%S")
    }

    return config


def save_training_config(config, save_dir="models"):
    """
    Save training configuration to a file.

    Args:
        config: Dictionary of training parameters
        save_dir: Directory to save the configuration

    Returns:
        Path to the saved configuration file
    """
    os.makedirs(save_dir, exist_ok=True)

    config_path = os.path.join(
        save_dir,
        f"training_config_{config['timestamp']}.json"
    )

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"Training configuration saved to {config_path}")
    return config_path


if __name__ == "__main__":
    # Test creating callbacks
    callbacks = create_callbacks()
    print(f"Created {len(callbacks)} callbacks")

    # Test creating and saving a training configuration
    config = create_training_config()
    config_path = save_training_config(config)
    print(f"Configuration saved to {config_path}")