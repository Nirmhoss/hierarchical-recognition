"""
evaluate.py - Evaluation metrics for hierarchical object recognition

This module contains functions for evaluating the hierarchical classification model,
including metrics specific to hierarchical classification performance.
"""

import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from data_utils import CLASS_NAMES, SUPERCLASS_NAMES


def _extract_predictions(
    model: tf.keras.Model,
    x_test: np.ndarray,
    y_test_super: np.ndarray,
    y_test_class: np.ndarray,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract predictions and convert to class indices.

    Args:
        model: The model to evaluate
        x_test: Test images
        y_test_super: True superclass labels (one-hot encoded)
        y_test_class: True class labels (one-hot encoded)
        batch_size: Batch size for evaluation

    Returns:
        Tuple of (y_true_super, y_pred_super, y_true_class, y_pred_class)
    """
    # Get model predictions
    predictions = model.predict(x_test, batch_size=batch_size)
    superclass_preds, class_preds = predictions

    # Convert one-hot encoded labels and predictions to class indices
    y_true_super = np.argmax(y_test_super, axis=1)
    y_pred_super = np.argmax(superclass_preds, axis=1)

    y_true_class = np.argmax(y_test_class, axis=1)
    y_pred_class = np.argmax(class_preds, axis=1)

    return y_true_super, y_pred_super, y_true_class, y_pred_class


def evaluate_model(
    model: tf.keras.Model,
    test_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    batch_size: int = 32,
    save_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate the hierarchical classification model on test data.

    Args:
        model: The model to evaluate
        test_data: Tuple of (x_test, y_test_super, y_test_class)
        batch_size: Batch size for evaluation
        save_dir: Directory to save evaluation results

    Returns:
        Dictionary containing evaluation metrics
    """
    # Unpack test data
    x_test, y_test_super, y_test_class = test_data

    # Evaluate model
    print("Evaluating model...")
    results = model.evaluate(
        x=x_test,
        y={"superclass_output": y_test_super, "class_output": y_test_class},
        batch_size=batch_size,
        verbose=1,
    )

    # Get metric names
    metric_names = model.metrics_names

    # Create results dictionary
    metrics = {metric_names[i]: results[i] for i in range(len(metric_names))}

    # Extract predictions
    y_true_super, y_pred_super, y_true_class, y_pred_class = _extract_predictions(
        model, x_test, y_test_super, y_test_class, batch_size
    )

    # Calculate additional metrics
    metrics.update(
        calculate_detailed_metrics(
            y_true_super=y_true_super,
            y_pred_super=y_pred_super,
            y_true_class=y_true_class,
            y_pred_class=y_pred_class,
        )
    )

    # Calculate hierarchical accuracy
    hierarchical_acc = calculate_hierarchical_accuracy(
        y_true_super=y_true_super,
        y_pred_super=y_pred_super,
        y_true_class=y_true_class,
        y_pred_class=y_pred_class,
    )
    metrics["hierarchical_accuracy"] = hierarchical_acc

    # Save evaluation results if save_dir is provided
    if save_dir:
        save_evaluation_results(metrics, save_dir)

        # Also save confusion matrices
        save_confusion_matrices(
            y_true_super=y_true_super,
            y_pred_super=y_pred_super,
            y_true_class=y_true_class,
            y_pred_class=y_pred_class,
            save_dir=save_dir,
        )

    return metrics


def calculate_detailed_metrics(
    y_true_super: np.ndarray,
    y_pred_super: np.ndarray,
    y_true_class: np.ndarray,
    y_pred_class: np.ndarray,
) -> Dict[str, Any]:
    """
    Calculate detailed classification metrics for both superclass and class predictions.

    Args:
        y_true_super: True superclass labels (indices)
        y_pred_super: Predicted superclass labels (indices)
        y_true_class: True class labels (indices)
        y_pred_class: Predicted class labels (indices)

    Returns:
        Dictionary containing detailed metrics
    """
    # Calculate metrics for superclass predictions
    super_precision, super_recall, super_f1, _ = precision_recall_fscore_support(
        y_true_super,
        y_pred_super,
        average="weighted",
    )

    super_accuracy = accuracy_score(y_true_super, y_pred_super)

    # Calculate metrics for class predictions
    class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
        y_true_class,
        y_pred_class,
        average="weighted",
    )

    class_accuracy = accuracy_score(y_true_class, y_pred_class)

    # Create detailed metrics dictionary
    detailed_metrics = {
        "superclass_precision": float(super_precision),
        "superclass_recall": float(super_recall),
        "superclass_f1": float(super_f1),
        "superclass_accuracy": float(super_accuracy),
        "class_precision": float(class_precision),
        "class_recall": float(class_recall),
        "class_f1": float(class_f1),
        "class_accuracy": float(class_accuracy),
    }

    return detailed_metrics


def calculate_hierarchical_accuracy(
    y_true_super: np.ndarray,
    y_pred_super: np.ndarray,
    y_true_class: np.ndarray,
    y_pred_class: np.ndarray,
) -> float:
    """
    Calculate hierarchical accuracy - a prediction is correct only if both
    superclass and class predictions are correct.

    Args:
        y_true_super: True superclass labels (indices)
        y_pred_super: Predicted superclass labels (indices)
        y_true_class: True class labels (indices)
        y_pred_class: Predicted class labels (indices)

    Returns:
        Hierarchical accuracy score
    """
    # A prediction is hierarchically correct if both superclass and class are correct
    superclass_correct = y_true_super == y_pred_super
    class_correct = y_true_class == y_pred_class

    # Both predictions need to be correct
    hierarchically_correct = np.logical_and(superclass_correct, class_correct)

    # Calculate accuracy
    hierarchical_accuracy = np.mean(hierarchically_correct)

    return float(hierarchical_accuracy)


def save_evaluation_results(
    metrics: Dict[str, Any],
    save_dir: str,
) -> None:
    """
    Save evaluation metrics to a JSON file.

    Args:
        metrics: Dictionary of evaluation metrics
        save_dir: Directory to save the results
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save metrics to JSON file
    metrics_path = os.path.join(save_dir, "evaluation_metrics.json")

    try:
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        print(f"Evaluation metrics saved to {metrics_path}")
    except IOError as e:
        print(f"Error saving evaluation metrics: {e}")


def save_confusion_matrices(
    y_true_super: np.ndarray,
    y_pred_super: np.ndarray,
    y_true_class: np.ndarray,
    y_pred_class: np.ndarray,
    save_dir: str,
) -> None:
    """
    Calculate and save confusion matrices for superclass and class predictions.

    Args:
        y_true_super: True superclass labels (indices)
        y_pred_super: Predicted superclass labels (indices)
        y_true_class: True class labels (indices)
        y_pred_class: Predicted class labels (indices)
        save_dir: Directory to save the confusion matrices
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Calculate confusion matrices
    superclass_cm = confusion_matrix(y_true_super, y_pred_super)
    class_cm = confusion_matrix(y_true_class, y_pred_class)

    # Save superclass confusion matrix
    superclass_cm_path = os.path.join(save_dir, "superclass_confusion_matrix.npy")
    superclass_cm_json_path = os.path.join(save_dir, "superclass_confusion_matrix.json")

    # Save class confusion matrix
    class_cm_path = os.path.join(save_dir, "class_confusion_matrix.npy")
    class_cm_json_path = os.path.join(save_dir, "class_confusion_matrix.json")

    try:
        # Save as NumPy files
        np.save(superclass_cm_path, superclass_cm)
        np.save(class_cm_path, class_cm)

        # Save as JSON files with labels
        superclass_cm_dict = {
            "matrix": superclass_cm.tolist(),
            "labels": SUPERCLASS_NAMES,
        }

        class_cm_dict = {
            "matrix": class_cm.tolist(),
            "labels": CLASS_NAMES,
        }

        with open(superclass_cm_json_path, "w", encoding="utf-8") as f:
            json.dump(superclass_cm_dict, f, indent=4)

        with open(class_cm_json_path, "w", encoding="utf-8") as f:
            json.dump(class_cm_dict, f, indent=4)

        print(f"Confusion matrices saved to {save_dir}")
    except IOError as e:
        print(f"Error saving confusion matrices: {e}")


def generate_classification_reports(
    y_true_super: np.ndarray,
    y_pred_super: np.ndarray,
    y_true_class: np.ndarray,
    y_pred_class: np.ndarray,
    save_dir: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Generate detailed classification reports for superclass and class predictions.

    Args:
        y_true_super: True superclass labels (indices)
        y_pred_super: Predicted superclass labels (indices)
        y_true_class: True class labels (indices)
        y_pred_class: Predicted class labels (indices)
        save_dir: Optional directory to save the reports

    Returns:
        Tuple of (superclass_report, class_report) strings
    """
    # Generate classification reports
    superclass_report = classification_report(
        y_true_super,
        y_pred_super,
        target_names=SUPERCLASS_NAMES,
    )

    class_report = classification_report(
        y_true_class,
        y_pred_class,
        target_names=CLASS_NAMES,
    )

    # Save reports if save_dir is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        superclass_report_path = os.path.join(save_dir, "superclass_classification_report.txt")
        class_report_path = os.path.join(save_dir, "class_classification_report.txt")

        try:
            with open(superclass_report_path, "w", encoding="utf-8") as f:
                f.write(superclass_report)

            with open(class_report_path, "w", encoding="utf-8") as f:
                f.write(class_report)

            print(f"Classification reports saved to {save_dir}")
        except IOError as e:
            print(f"Error saving classification reports: {e}")

    return superclass_report, class_report
