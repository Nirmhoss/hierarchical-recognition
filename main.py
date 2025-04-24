"""
main.py - Main entry point for hierarchical object recognition system

This script integrates all components of the hierarchical object recognition system
and provides a command-line interface for training, evaluating, and making predictions.
"""

import os
import argparse
import tensorflow as tf
import numpy as np
import time
import json

# Import project modules
from data_utils import (
    load_data, create_data_generators, preprocess_image,
    CLASS_NAMES, SUPERCLASS_NAMES
)
from model import (
    create_hierarchical_model, create_hierarchical_resnet, compile_model
)
from train import (
    train_model, load_trained_model, create_callbacks,
    create_training_config, save_training_config
)
from evaluate import (
    evaluate_model, save_evaluation_results,
    generate_confusion_matrices, analyze_errors
)
from visualize import (
    plot_training_history, plot_confusion_matrices,
    visualize_predictions, plot_prediction_probabilities,
    plot_class_accuracies, plot_hierarchical_tree
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Hierarchical Object Recognition System'
    )

    # Main operation modes
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--predict', action='store_true', help='Make predictions on new images')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--superclass_weight', type=float, default=0.3,
                        help='Weight for superclass loss (between 0 and 1)')
    parser.add_argument('--no_data_augmentation', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--model_type', type=str, default='cnn', choices=['cnn', 'resnet'],
                        help='Type of model to use')

    # Paths
    parser.add_argument('--model_path', type=str, help='Path to saved model')
    parser.add_argument('--image_path', type=str, help='Path to image for prediction')
    parser.add_argument('--history_path', type=str, help='Path to training history JSON file')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save outputs')

    # Parse arguments
    args = parser.parse_args()

    # Validate arguments
    if args.predict and not (args.model_path or args.image_path):
        parser.error("--predict requires --model_path and --image_path")

    if args.evaluate and not args.model_path:
        parser.error("--evaluate requires --model_path")

    if args.visualize and not (args.model_path or args.history_path):
        parser.error("--visualize requires --model_path or --history_path")

    return args


def main():
    """Main entry point for the application."""
    # Parse command line arguments
    args = parse_arguments()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine operation mode
    if args.train:
        train(args)
    elif args.evaluate:
        evaluate(args)
    elif args.predict:
        predict(args)
    elif args.visualize:
        visualize(args)
    else:
        # Default to training if no operation specified
        print("No operation specified, defaulting to training.")
        train(args)


def train(args):
    """Train the hierarchical model."""
    print("Loading data...")

    # Load and preprocess the data
    (x_train, y_train_super, y_train_class,
     x_val, y_val_super, y_val_class,
     x_test, y_test_super, y_test_class) = load_data()

    # Create data generators
    data_augmentation = not args.no_data_augmentation
    train_generator, val_generator = create_data_generators(
        x_train, y_train_super, y_train_class,
        x_val, y_val_super, y_val_class,
        batch_size=args.batch_size,
        augment=data_augmentation
    )

    # Determine steps per epoch
    steps_per_epoch = len(x_train) // args.batch_size
    validation_steps = len(x_val) // args.batch_size

    # Create model
    print(f"Creating {args.model_type} model...")
    if args.model_type == 'cnn':
        model = create_hierarchical_model()
    else:  # resnet
        model = create_hierarchical_resnet()

    # Compile model
    model = compile_model(
        model,
        learning_rate=args.learning_rate,
        superclass_weight=args.superclass_weight
    )

    # Print model summary
    model.summary()

    # Create callbacks
    callbacks = create_callbacks(
        model_name=f"hierarchical_{args.model_type}",
        checkpoints_dir=os.path.join(args.output_dir, "checkpoints"),
        logs_dir=os.path.join(args.output_dir, "logs")
    )

    # Create and save training configuration
    config = create_training_config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        superclass_weight=args.superclass_weight,
        data_augmentation=data_augmentation,
        model_type=args.model_type
    )
    save_training_config(
        config,
        save_dir=os.path.join(args.output_dir, "configs")
    )

    # Train the model
    history, model_path = train_model(
        model,
        train_generator,
        val_generator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=args.epochs,
        callbacks=callbacks,
        model_name=f"hierarchical_{args.model_type}",
        save_dir=os.path.join(args.output_dir, "models")
    )

    # Evaluate the model
    print("\nEvaluating model on test set...")
    results = evaluate_model(model, x_test, y_test_super, y_test_class)

    # Save evaluation results
    save_evaluation_results(
        results,
        save_dir=os.path.join(args.output_dir, "evaluation")
    )

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_training_history(
        history,
        save_dir=os.path.join(args.output_dir, "visualizations")
    )

    # Convert labels to indices for confusion matrices
    y_true_super_idx = np.argmax(y_test_super, axis=1)
    y_true_class_idx = np.argmax(y_test_class, axis=1)
    y_pred_super, y_pred_class = model.predict(x_test)
    y_pred_super_idx = np.argmax(y_pred_super, axis=1)
    y_pred_class_idx = np.argmax(y_pred_class, axis=1)

    plot_confusion_matrices(
        y_true_super_idx, y_true_class_idx,
        y_pred_super_idx, y_pred_class_idx,
        save_dir=os.path.join(args.output_dir, "visualizations")
    )

    visualize_predictions(
        x_test, y_test_super, y_test_class,
        model,
        num_samples=10,
        save_dir=os.path.join(args.output_dir, "visualizations")
    )

    plot_class_accuracies(
        results,
        save_dir=os.path.join(args.output_dir, "visualizations")
    )

    plot_hierarchical_tree(
        results,
        save_dir=os.path.join(args.output_dir, "visualizations")
    )

    print(f"\nTraining complete. Model saved to {model_path}")


def evaluate(args):
    """Evaluate a trained model."""
    print(f"Loading model from {args.model_path}...")
    model = load_trained_model(args.model_path)

    print("Loading test data...")
    _, _, _, _, _, _, x_test, y_test_super, y_test_class = load_data()

    # Evaluate the model
    results = evaluate_model(model, x_test, y_test_super, y_test_class)

    # Save evaluation results
    results_path = save_evaluation_results(
        results,
        save_dir=os.path.join(args.output_dir, "evaluation")
    )

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Convert labels to indices for confusion matrices
    y_true_super_idx = np.argmax(y_test_super, axis=1)
    y_true_class_idx = np.argmax(y_test_class, axis=1)
    y_pred_super, y_pred_class = model.predict(x_test)
    y_pred_super_idx = np.argmax(y_pred_super, axis=1)
    y_pred_class_idx = np.argmax(y_pred_class, axis=1)

    plot_confusion_matrices(
        y_true_super_idx, y_true_class_idx,
        y_pred_super_idx, y_pred_class_idx,
        save_dir=os.path.join(args.output_dir, "visualizations")
    )

    visualize_predictions(
        x_test, y_test_super, y_test_class,
        model,
        num_samples=10,
        save_dir=os.path.join(args.output_dir, "visualizations")
    )

    plot_class_accuracies(
        results,
        save_dir=os.path.join(args.output_dir, "visualizations")
    )

    plot_hierarchical_tree(
        results,
        save_dir=os.path.join(args.output_dir, "visualizations")
    )

    print(f"\nEvaluation complete. Results saved to {results_path}")


def predict(args):
    """Make predictions on new images."""
    print(f"Loading model from {args.model_path}...")
    model = load_trained_model(args.model_path)

    # Process the image
    print(f"Processing image {args.image_path}...")
    img_array = preprocess_image(args.image_path)

    # Make prediction
    y_pred_super, y_pred_class = model.predict(img_array)

    # Get the predicted indices
    pred_superclass_idx = np.argmax(y_pred_super[0])
    pred_class_idx = np.argmax(y_pred_class[0])

    # Get the predicted labels
    pred_superclass = SUPERCLASS_NAMES[pred_superclass_idx]
    pred_class = CLASS_NAMES[pred_class_idx]

    # Get the prediction probabilities
    pred_superclass_prob = y_pred_super[0][pred_superclass_idx]
    pred_class_prob = y_pred_class[0][pred_class_idx]

    # Print prediction
    print("\nPrediction Results:")
    print(f"Superclass: {pred_superclass} ({pred_superclass_prob:.4f})")
    print(f"Class: {pred_class} ({pred_class_prob:.4f})")

    # Visualize prediction
    print("\nGenerating visualization...")
    plot_path = plot_prediction_probabilities(
        args.image_path,
        model,
        preprocess_image,
        save_dir=os.path.join(args.output_dir, "predictions")
    )

    print(f"\nPrediction visualization saved to {plot_path}")


def visualize(args):
    """Generate visualizations from a trained model or history file."""
    # Check if history file is provided
    if args.history_path:
        print(f"Loading training history from {args.history_path}...")
        plot_paths = plot_training_history(
            args.history_path,
            save_dir=os.path.join(args.output_dir, "visualizations")
        )
        print(f"Training history visualizations saved to {', '.join(plot_paths.values())}")

    # Check if model is provided
    if args.model_path:
        print(f"Loading model from {args.model_path}...")
        model = load_trained_model(args.model_path)

        print("Loading test data...")
        _, _, _, _, _, _, x_test, y_test_super, y_test_class = load_data()

        # Generate visualizations for the model and test data
        print("Generating model visualizations...")

        # Make predictions
        y_pred_super, y_pred_class = model.predict(x_test)

        # Convert labels to indices for confusion matrices
        y_true_super_idx = np.argmax(y_test_super, axis=1)
        y_true_class_idx = np.argmax(y_test_class, axis=1)
        y_pred_super_idx = np.argmax(y_pred_super, axis=1)
        y_pred_class_idx = np.argmax(y_pred_class, axis=1)

        # Generate confusion matrices
        cm_paths = plot_confusion_matrices(
            y_true_super_idx, y_true_class_idx,
            y_pred_super_idx, y_pred_class_idx,
            save_dir=os.path.join(args.output_dir, "visualizations")
        )
        print(f"Confusion matrices saved to {', '.join(cm_paths.values())}")

        # Visualize predictions
        pred_path = visualize_predictions(
            x_test, y_test_super, y_test_class,
            model,
            num_samples=10,
            save_dir=os.path.join(args.output_dir, "visualizations")
        )
        print(f"Prediction visualization saved to {pred_path}")

        # Evaluate the model to get results for class accuracies
        results = evaluate_model(model, x_test, y_test_super, y_test_class)

        # Plot class accuracies
        acc_paths = plot_class_accuracies(
            results,
            save_dir=os.path.join(args.output_dir, "visualizations")
        )
        print(f"Class accuracy visualizations saved to {', '.join(acc_paths.values())}")

        # Plot hierarchical tree
        tree_path = plot_hierarchical_tree(
            results,
            save_dir=os.path.join(args.output_dir, "visualizations")
        )
        print(f"Hierarchical tree visualization saved to {tree_path}")


if __name__ == "__main__":
    main()