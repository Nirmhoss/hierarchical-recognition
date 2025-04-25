"""
main.py - Main entry point for hierarchical object recognition system

This script integrates all components of the hierarchical object recognition system
and provides a command-line interface for training, evaluating, and making predictions.
"""

import argparse
import os
import sys
from typing import Optional

import tensorflow as tf

from data_utils import load_cifar10_data, prepare_hierarchical_data
from evaluate import evaluate_model
from model import build_hierarchical_model, compile_model
from train import train_model
from visualize import visualize_predictions, visualize_training_history


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the application.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Hierarchical Object Recognition System")

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "evaluate", "predict"],
        help="Operation mode: train, evaluate, or predict",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to store/load dataset",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="models/hierarchical_model.h5",
        help="Path to save/load model",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training/evaluation",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer",
    )

    parser.add_argument(
        "--superclass-weight",
        type=float,
        default=0.3,
        help="Weight for superclass loss (0-1)",
    )

    parser.add_argument(
        "--image-path",
        type=str,
        help="Path to image for prediction (predict mode only)",
    )

    return parser.parse_args()


def load_trained_model(model_path: str) -> tf.keras.Model:
    """
    Load a trained model from file with error handling.

    Args:
        model_path: Path to the saved model file

    Returns:
        Loaded Keras model

    Raises:
        FileNotFoundError: If the model file does not exist
        IOError: If the model file cannot be read
        ImportError: If the model cannot be imported
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        print(f"Loading model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        return model
    except (IOError, ImportError) as e:
        print(f"Error loading model: {e}")
        raise


def _prepare_data_for_training(data_dir: str) -> tuple:
    """Helper function to load and prepare data for training."""
    print("Loading and preparing data...")
    (x_train, y_train), (x_test, y_test) = load_cifar10_data(data_dir)

    train_data = prepare_hierarchical_data(x_train, y_train)
    test_data = prepare_hierarchical_data(x_test, y_test)

    return train_data, test_data


def train_and_save_model(
    data_dir: str,
    model_path: str,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    superclass_weight: float,
) -> tf.keras.Model:
    """
    Train and save a hierarchical recognition model.

    Args:
        data_dir: Directory to store/load the dataset
        model_path: Path to save the trained model
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate for the optimizer
        superclass_weight: Weight for the superclass loss

    Returns:
        Trained Keras model
    """
    # Load and prepare data
    train_data, test_data = _prepare_data_for_training(data_dir)

    # Unpack data for clarity
    x_train, y_train_super, y_train_class = train_data

    # Create model
    print("Building model...")
    model = build_hierarchical_model(input_shape=x_train.shape[1:])
    model = compile_model(
        model,
        learning_rate=learning_rate,
        superclass_weight=superclass_weight,
    )

    # Create directories for model and logs
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Train model
    print(f"Training model for {epochs} epochs...")
    history = train_model(
        model=model,
        train_data=train_data,
        validation_data=test_data,
        batch_size=batch_size,
        epochs=epochs,
        model_path=model_path,
    )

    # Visualize training history
    visualize_training_history(history, save_dir="results")

    return model


def main() -> None:
    """Main entry point for the application."""
    # Parse command line arguments
    args = parse_arguments()

    # Create necessary directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    try:
        if args.mode == "train":
            # Train and save model
            model = train_and_save_model(
                data_dir=args.data_dir,
                model_path=args.model_path,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                superclass_weight=args.superclass_weight,
            )

            # Evaluate the trained model
            test_data = prepare_hierarchical_data(*load_cifar10_data(args.data_dir)[1])
            evaluate_model(
                model=model,
                test_data=test_data,
                batch_size=args.batch_size,
                save_dir="results",
            )

        elif args.mode == "evaluate":
            # Load model
            model = load_trained_model(args.model_path)

            # Evaluate model
            test_data = prepare_hierarchical_data(*load_cifar10_data(args.data_dir)[1])
            evaluate_model(
                model=model,
                test_data=test_data,
                batch_size=args.batch_size,
                save_dir="results",
            )

            # Visualize some predictions
            x_test, y_test_super, y_test_class = test_data
            visualize_predictions(
                x_test=x_test[:10],
                y_test_super=y_test_super[:10],
                y_test_class=y_test_class[:10],
                model=model,
                save_dir="results",
            )

        elif args.mode == "predict":
            if not args.image_path:
                print("Error: --image-path is required for predict mode")
                sys.exit(1)

            # Check if image exists
            if not os.path.exists(args.image_path):
                print(f"Error: Image not found at {args.image_path}")
                sys.exit(1)

            # Load model
            model = load_trained_model(args.model_path)

            # TODO: Implement single image prediction
            print(f"Prediction for {args.image_path} not yet implemented")
    except Exception as e:
        # Use a specific exception handler rather than broad exception
        # This is a reasonable use case for a CLI application's main function
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
