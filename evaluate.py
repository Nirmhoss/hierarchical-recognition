"""
evaluate.py - Evaluation metrics for hierarchical object recognition

This module contains functions for evaluating the hierarchical classification model,
including metrics specific to hierarchical classification performance.
"""
import datetime
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import pandas as pd
import json

from data_utils import CLASS_NAMES, SUPERCLASS_NAMES, CLASS_TO_SUPERCLASS


def evaluate_model(model, x_test, y_test_super, y_test_class):
    """
    Evaluate the model on test data with various metrics.

    Args:
        model: Trained Keras model
        x_test: Test images
        y_test_super: One-hot encoded superclass labels
        y_test_class: One-hot encoded class labels

    Returns:
        Dictionary of evaluation metrics
    """
    print("Evaluating model...")

    # Get model predictions
    y_pred_super, y_pred_class = model.predict(x_test)

    # Convert one-hot encoded labels to class indices
    y_true_super = np.argmax(y_test_super, axis=1)
    y_true_class = np.argmax(y_test_class, axis=1)
    y_pred_super_idx = np.argmax(y_pred_super, axis=1)
    y_pred_class_idx = np.argmax(y_pred_class, axis=1)

    # Calculate basic metrics
    superclass_accuracy = accuracy_score(y_true_super, y_pred_super_idx)
    superclass_precision = precision_score(y_true_super, y_pred_super_idx, average='weighted')
    superclass_recall = recall_score(y_true_super, y_pred_super_idx, average='weighted')
    superclass_f1 = f1_score(y_true_super, y_pred_super_idx, average='weighted')

    class_accuracy = accuracy_score(y_true_class, y_pred_class_idx)
    class_precision = precision_score(y_true_class, y_pred_class_idx, average='weighted')
    class_recall = recall_score(y_true_class, y_pred_class_idx, average='weighted')
    class_f1 = f1_score(y_true_class, y_pred_class_idx, average='weighted')

    # Calculate hierarchical metrics
    hierarchical_accuracy = calculate_hierarchical_accuracy(
        y_true_super, y_true_class, y_pred_super_idx, y_pred_class_idx
    )

    # Calculate consistency metric (does the predicted class belong to the predicted superclass?)
    consistency_score = calculate_consistency(y_pred_super_idx, y_pred_class_idx)

    # Calculate average precision and recall per level
    superclass_precision_per_class = precision_score(
        y_true_super, y_pred_super_idx, average=None
    )
    class_precision_per_class = precision_score(
        y_true_class, y_pred_class_idx, average=None
    )

    # Create results dictionary
    results = {
        'superclass_accuracy': float(superclass_accuracy),
        'superclass_precision': float(superclass_precision),
        'superclass_recall': float(superclass_recall),
        'superclass_f1': float(superclass_f1),
        'class_accuracy': float(class_accuracy),
        'class_precision': float(class_precision),
        'class_recall': float(class_recall),
        'class_f1': float(class_f1),
        'hierarchical_accuracy': float(hierarchical_accuracy),
        'consistency_score': float(consistency_score),
        'superclass_precision_per_class': {
            SUPERCLASS_NAMES[i]: float(superclass_precision_per_class[i])
            for i in range(len(SUPERCLASS_NAMES))
        },
        'class_precision_per_class': {
            CLASS_NAMES[i]: float(class_precision_per_class[i])
            for i in range(len(CLASS_NAMES))
        }
    }

    # Print results
    print("\nEvaluation Results:")
    print(f"Superclass Accuracy: {superclass_accuracy:.4f}")
    print(f"Superclass F1 Score: {superclass_f1:.4f}")
    print(f"Class Accuracy: {class_accuracy:.4f}")
    print(f"Class F1 Score: {class_f1:.4f}")
    print(f"Hierarchical Accuracy: {hierarchical_accuracy:.4f}")
    print(f"Prediction Consistency: {consistency_score:.4f}")

    return results


def calculate_hierarchical_accuracy(y_true_super, y_true_class, y_pred_super, y_pred_class):
    """
    Calculate hierarchical accuracy, which requires both levels to be correct.

    Args:
        y_true_super: True superclass indices
        y_true_class: True class indices
        y_pred_super: Predicted superclass indices
        y_pred_class: Predicted class indices

    Returns:
        Hierarchical accuracy score
    """
    # A prediction is correct only if both superclass and class are correct
    correct_predictions = np.logical_and(
        y_true_super == y_pred_super,
        y_true_class == y_pred_class
    )

    hierarchical_accuracy = np.mean(correct_predictions)
    return hierarchical_accuracy


def calculate_consistency(y_pred_super, y_pred_class):
    """
    Calculate whether predictions are consistent with the hierarchy.
    A prediction is consistent if the predicted class belongs to the predicted superclass.

    Args:
        y_pred_super: Predicted superclass indices
        y_pred_class: Predicted class indices

    Returns:
        Consistency score (percentage of consistent predictions)
    """
    consistencies = []

    for i in range(len(y_pred_class)):
        pred_class = y_pred_class[i]
        pred_super = y_pred_super[i]

        # Check if the predicted class belongs to the predicted superclass
        true_super_of_pred_class = CLASS_TO_SUPERCLASS[pred_class]
        is_consistent = (true_super_of_pred_class == pred_super)

        consistencies.append(is_consistent)

    consistency_score = np.mean(consistencies)
    return consistency_score


def generate_confusion_matrices(y_true_super, y_true_class, y_pred_super, y_pred_class):
    """
    Generate confusion matrices for both hierarchical levels.

    Args:
        y_true_super: True superclass indices
        y_true_class: True class indices
        y_pred_super: Predicted superclass indices
        y_pred_class: Predicted class indices

    Returns:
        Tuple of (superclass_cm, class_cm)
    """
    # Generate confusion matrices
    superclass_cm = confusion_matrix(y_true_super, y_pred_super)
    class_cm = confusion_matrix(y_true_class, y_pred_class)

    # Convert to pandas DataFrames for better visualization
    superclass_cm_df = pd.DataFrame(
        superclass_cm,
        index=SUPERCLASS_NAMES,
        columns=SUPERCLASS_NAMES
    )

    class_cm_df = pd.DataFrame(
        class_cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    )

    return superclass_cm_df, class_cm_df


def analyze_errors(y_true_super, y_true_class, y_pred_super, y_pred_class):
    """
    Analyze prediction errors to identify challenging cases.

    Args:
        y_true_super: True superclass indices
        y_true_class: True class indices
        y_pred_super: Predicted superclass indices
        y_pred_class: Predicted class indices

    Returns:
        Dictionary with error analysis results
    """
    error_analysis = {}

    # Find samples with errors
    superclass_errors = (y_true_super != y_pred_super)
    class_errors = (y_true_class != y_pred_class)

    # Calculate error types
    only_superclass_errors = np.logical_and(superclass_errors, ~class_errors)
    only_class_errors = np.logical_and(~superclass_errors, class_errors)
    both_level_errors = np.logical_and(superclass_errors, class_errors)

    # Calculate error rates
    error_analysis['total_samples'] = len(y_true_class)
    error_analysis['correct_predictions'] = int(np.sum(~np.logical_or(superclass_errors, class_errors)))
    error_analysis['only_superclass_errors'] = int(np.sum(only_superclass_errors))
    error_analysis['only_class_errors'] = int(np.sum(only_class_errors))
    error_analysis['both_level_errors'] = int(np.sum(both_level_errors))

    # Calculate percentages
    total = error_analysis['total_samples']
    error_analysis['accuracy_percentage'] = (error_analysis['correct_predictions'] / total) * 100
    error_analysis['only_superclass_errors_percentage'] = (error_analysis['only_superclass_errors'] / total) * 100
    error_analysis['only_class_errors_percentage'] = (error_analysis['only_class_errors'] / total) * 100
    error_analysis['both_level_errors_percentage'] = (error_analysis['both_level_errors'] / total) * 100

    # Find the most confused class pairs
    class_error_indices = np.where(class_errors)[0]
    confused_pairs = {}

    for idx in class_error_indices:
        true_class = CLASS_NAMES[y_true_class[idx]]
        pred_class = CLASS_NAMES[y_pred_class[idx]]
        pair = f"{true_class} → {pred_class}"

        if pair in confused_pairs:
            confused_pairs[pair] += 1
        else:
            confused_pairs[pair] = 1

    # Sort pairs by frequency
    sorted_pairs = sorted(confused_pairs.items(), key=lambda x: x[1], reverse=True)
    error_analysis['most_confused_classes'] = dict(sorted_pairs[:10])  # Top 10 confused pairs

    return error_analysis


def save_evaluation_results(results, save_dir="evaluation"):
    """
    Save evaluation results to a JSON file.

    Args:
        results: Dictionary of evaluation results
        save_dir: Directory to save results

    Returns:
        Path to the saved results file
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save as JSON
    timestamp = results.get('timestamp', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    results_path = os.path.join(save_dir, f"evaluation_results_{timestamp}.json")

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Evaluation results saved to {results_path}")
    return results_path


def evaluate_model_from_predictions(y_true_super, y_true_class, y_pred_super, y_pred_class):
    """
    Evaluate model performance from pre-computed predictions.
    Useful for evaluating without needing to rerun the model.

    Args:
        y_true_super: True superclass indices
        y_true_class: True class indices
        y_pred_super: Predicted superclass indices (or probabilities)
        y_pred_class: Predicted class indices (or probabilities)

    Returns:
        Dictionary of evaluation metrics
    """
    # Convert probabilities to class indices if needed
    if y_pred_super.ndim > 1 and y_pred_super.shape[1] > 1:
        y_pred_super = np.argmax(y_pred_super, axis=1)

    if y_pred_class.ndim > 1 and y_pred_class.shape[1] > 1:
        y_pred_class = np.argmax(y_pred_class, axis=1)

    # Convert one-hot encoded labels to class indices if needed
    if y_true_super.ndim > 1 and y_true_super.shape[1] > 1:
        y_true_super = np.argmax(y_true_super, axis=1)

    if y_true_class.ndim > 1 and y_true_class.shape[1] > 1:
        y_true_class = np.argmax(y_true_class, axis=1)

    # Calculate metrics
    results = {}

    # Basic metrics
    results['superclass_accuracy'] = float(accuracy_score(y_true_super, y_pred_super))
    results['class_accuracy'] = float(accuracy_score(y_true_class, y_pred_class))

    # Hierarchical metrics
    results['hierarchical_accuracy'] = float(calculate_hierarchical_accuracy(
        y_true_super, y_true_class, y_pred_super, y_pred_class
    ))

    # Consistency
    results['consistency_score'] = float(calculate_consistency(y_pred_super, y_pred_class))

    # Generate classification reports
    superclass_report = classification_report(
        y_true_super, y_pred_super,
        target_names=SUPERCLASS_NAMES,
        output_dict=True
    )

    class_report = classification_report(
        y_true_class, y_pred_class,
        target_names=CLASS_NAMES,
        output_dict=True
    )

    results['superclass_report'] = superclass_report
    results['class_report'] = class_report

    # Error analysis
    results['error_analysis'] = analyze_errors(
        y_true_super, y_true_class, y_pred_super, y_pred_class
    )

    # Add timestamp
    results['timestamp'] = tf.datetime.now().strftime("%Y%m%d-%H%M%S")

    return results


if __name__ == "__main__":
    # Test the error analysis function with simulated data
    num_samples = 1000

    # Simulated data with a mix of correct and incorrect predictions
    y_true_super = np.random.randint(0, len(SUPERCLASS_NAMES), size=num_samples)
    y_true_class = np.array([
        np.random.choice([i for i, v in enumerate(CLASS_TO_SUPERCLASS.values())
                          if v == superclass])
        for superclass in y_true_super
    ])

    # Create predictions with controlled error rates
    y_pred_super = np.copy(y_true_super)
    y_pred_class = np.copy(y_true_class)

    # Introduce errors at superclass level (20% error rate)
    error_indices = np.random.choice(
        num_samples, size=int(num_samples * 0.2), replace=False
    )
    for idx in error_indices:
        y_pred_super[idx] = (y_true_super[idx] + 1) % len(SUPERCLASS_NAMES)

    # Introduce errors at class level (30% error rate)
    error_indices = np.random.choice(
        num_samples, size=int(num_samples * 0.3), replace=False
    )
    for idx in error_indices:
        # Get a different class from the same superclass
        current_superclass = CLASS_TO_SUPERCLASS[y_true_class[idx]]
        superclass_classes = [
            i for i, v in CLASS_TO_SUPERCLASS.items() if v == current_superclass
        ]
        valid_alternatives = [c for c in superclass_classes if c != y_true_class[idx]]

        if valid_alternatives:
            y_pred_class[idx] = np.random.choice(valid_alternatives)
        else:
            # If no alternatives in same superclass, pick a class from a different superclass
            y_pred_class[idx] = (y_true_class[idx] + 1) % len(CLASS_NAMES)

    # Run the error analysis
    error_analysis = analyze_errors(y_true_super, y_true_class, y_pred_super, y_pred_class)

    # Print results
    print("Error Analysis Test Results:")
    for key, value in error_analysis.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for subkey, subvalue in value.items():
                print(f"  {subkey}: {subvalue}")
        else:
            print(f"{key}: {value}")