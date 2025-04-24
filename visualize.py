"""
visualize.py - Visualization utilities for hierarchical object recognition

This module provides functions for visualizing training progress, model predictions,
confusion matrices, and hierarchical classification results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

from data_utils import CLASS_NAMES, SUPERCLASS_NAMES, CLASS_TO_SUPERCLASS


def plot_training_history(history, save_dir="visualizations"):
    """
    Plot training and validation metrics.

    Args:
        history: Training history dictionary or path to history JSON file
        save_dir: Directory to save plots

    Returns:
        Dictionary of saved plot file paths
    """
    os.makedirs(save_dir, exist_ok=True)

    # Convert Keras History object to dictionary if needed
    if hasattr(history, 'history'):
        history = history.history
    # Load history if a file path is provided
    elif isinstance(history, str):
        with open(history, 'r') as f:
            history = json.load(f)

    # Load history if a file path is provided
    if isinstance(history, str):
        with open(history, 'r') as f:
            history = json.load(f)

    saved_plots = {}

    # Plot loss
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Overall Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot superclass and class losses
    plt.subplot(2, 1, 2)
    plt.plot(history['superclass_output_loss'], label='Superclass Loss')
    plt.plot(history['val_superclass_output_loss'], label='Val Superclass Loss')
    plt.plot(history['class_output_loss'], label='Class Loss')
    plt.plot(history['val_class_output_loss'], label='Val Class Loss')
    plt.title('Component Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save loss plot
    loss_plot_path = os.path.join(save_dir, 'loss_history.png')
    plt.savefig(loss_plot_path)
    saved_plots['loss'] = loss_plot_path
    plt.close()

    # Plot accuracy
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(history['superclass_output_accuracy'], label='Superclass Accuracy')
    plt.plot(history['val_superclass_output_accuracy'], label='Val Superclass Accuracy')
    plt.title('Superclass Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(history['class_output_accuracy'], label='Class Accuracy')
    plt.plot(history['val_class_output_accuracy'], label='Val Class Accuracy')
    plt.title('Class Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save accuracy plot
    acc_plot_path = os.path.join(save_dir, 'accuracy_history.png')
    plt.savefig(acc_plot_path)
    saved_plots['accuracy'] = acc_plot_path
    plt.close()

    # Plot Top-K accuracy if available
    if 'superclass_output_top2_accuracy' in history:
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(history['superclass_output_top2_accuracy'], label='Superclass Top-2 Accuracy')
        plt.plot(history['val_superclass_output_top2_accuracy'], label='Val Superclass Top-2 Accuracy')
        plt.title('Superclass Top-K Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(history['class_output_top3_accuracy'], label='Class Top-3 Accuracy')
        plt.plot(history['val_class_output_top3_accuracy'], label='Val Class Top-3 Accuracy')
        plt.title('Class Top-K Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save Top-K accuracy plot
        topk_plot_path = os.path.join(save_dir, 'topk_accuracy_history.png')
        plt.savefig(topk_plot_path)
        saved_plots['topk_accuracy'] = topk_plot_path
        plt.close()

    # Print summary of training progress
    print("\nTraining History Summary:")
    print(f"Initial Training Loss: {history['loss'][0]:.4f}")
    print(f"Final Training Loss: {history['loss'][-1]:.4f}")
    print(f"Initial Validation Loss: {history['val_loss'][0]:.4f}")
    print(f"Final Validation Loss: {history['val_loss'][-1]:.4f}")

    print(f"Initial Superclass Accuracy: {history['superclass_output_accuracy'][0]:.4f}")
    print(f"Final Superclass Accuracy: {history['superclass_output_accuracy'][-1]:.4f}")
    print(f"Initial Class Accuracy: {history['class_output_accuracy'][0]:.4f}")
    print(f"Final Class Accuracy: {history['class_output_accuracy'][-1]:.4f}")

    return saved_plots


def plot_confusion_matrices(y_true_super, y_true_class, y_pred_super, y_pred_class, save_dir="visualizations"):
    """
    Plot confusion matrices for both hierarchical levels.

    Args:
        y_true_super: True superclass indices
        y_true_class: True class indices
        y_pred_super: Predicted superclass indices
        y_pred_class: Predicted class indices
        save_dir: Directory to save plots

    Returns:
        Dictionary of saved plot file paths
    """
    os.makedirs(save_dir, exist_ok=True)
    saved_plots = {}

    # Convert one-hot encoded labels to class indices if needed
    if y_true_super.ndim > 1 and y_true_super.shape[1] > 1:
        y_true_super = np.argmax(y_true_super, axis=1)
    if y_true_class.ndim > 1 and y_true_class.shape[1] > 1:
        y_true_class = np.argmax(y_true_class, axis=1)

    # Convert probabilities to class indices if needed
    if y_pred_super.ndim > 1 and y_pred_super.shape[1] > 1:
        y_pred_super = np.argmax(y_pred_super, axis=1)
    if y_pred_class.ndim > 1 and y_pred_class.shape[1] > 1:
        y_pred_class = np.argmax(y_pred_class, axis=1)

    # Compute confusion matrices
    superclass_cm = confusion_matrix(y_true_super, y_pred_super)
    class_cm = confusion_matrix(y_true_class, y_pred_class)

    # Normalize by row (true labels)
    superclass_cm_norm = superclass_cm.astype('float') / superclass_cm.sum(axis=1)[:, np.newaxis]
    class_cm_norm = class_cm.astype('float') / class_cm.sum(axis=1)[:, np.newaxis]

    # Plot superclass confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        superclass_cm_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=SUPERCLASS_NAMES,
        yticklabels=SUPERCLASS_NAMES
    )
    plt.title('Superclass Confusion Matrix (Normalized)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()

    # Save superclass confusion matrix
    superclass_cm_path = os.path.join(save_dir, 'superclass_confusion_matrix.png')
    plt.savefig(superclass_cm_path)
    saved_plots['superclass_cm'] = superclass_cm_path
    plt.close()

    # Plot class confusion matrix
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        class_cm_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.title('Class Confusion Matrix (Normalized)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()

    # Save class confusion matrix
    class_cm_path = os.path.join(save_dir, 'class_confusion_matrix.png')
    plt.savefig(class_cm_path)
    saved_plots['class_cm'] = class_cm_path
    plt.close()

    # Create a hierarchical confusion matrix
    hierarchical_cm = pd.DataFrame(0,
                                   index=CLASS_NAMES,
                                   columns=CLASS_NAMES)

    for i, true_class in enumerate(CLASS_NAMES):
        true_indices = np.where(y_true_class == i)[0]
        if len(true_indices) == 0:
            continue

        for j, pred_class in enumerate(CLASS_NAMES):
            count = np.sum(y_pred_class[true_indices] == j)
            hierarchical_cm.at[true_class, pred_class] = count

    # Normalize by row
    hierarchical_cm = hierarchical_cm.div(hierarchical_cm.sum(axis=1), axis=0)

    # Create a custom colormap that highlights the hierarchy
    # Group by superclass
    sorted_classes = []
    for superclass in range(len(SUPERCLASS_NAMES)):
        for class_idx, class_name in enumerate(CLASS_NAMES):
            if CLASS_TO_SUPERCLASS[class_idx] == superclass:
                sorted_classes.append(class_name)

    # Reindex the confusion matrix
    hierarchical_cm = hierarchical_cm.reindex(index=sorted_classes, columns=sorted_classes)

    # Plot hierarchical confusion matrix
    plt.figure(figsize=(16, 14))

    # Add grid lines to separate superclasses
    ax = plt.gca()

    # Count classes per superclass to determine grid positions
    classes_per_superclass = {}
    for superclass in range(len(SUPERCLASS_NAMES)):
        classes_per_superclass[superclass] = sum(1 for v in CLASS_TO_SUPERCLASS.values() if v == superclass)

    # Draw lines at superclass boundaries
    pos = 0
    for superclass in range(len(SUPERCLASS_NAMES) - 1):
        pos += classes_per_superclass[superclass]
        ax.axhline(y=pos, color='black', linestyle='-', linewidth=2)
        ax.axvline(x=pos, color='black', linestyle='-', linewidth=2)

    # Plot the heatmap
    sns.heatmap(
        hierarchical_cm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        ax=ax
    )

    plt.title('Hierarchical Confusion Matrix (Grouped by Superclass)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()

    # Save hierarchical confusion matrix
    hierarchical_cm_path = os.path.join(save_dir, 'hierarchical_confusion_matrix.png')
    plt.savefig(hierarchical_cm_path)
    saved_plots['hierarchical_cm'] = hierarchical_cm_path
    plt.close()

    return saved_plots


def visualize_predictions(x_test, y_test_super, y_test_class, model, num_samples=5, save_dir="visualizations"):
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

    # Get random sample indices
    indices = np.random.choice(len(x_test), num_samples, replace=False)

    # Make predictions
    x_samples = x_test[indices]
    y_pred_super, y_pred_class = model.predict(x_samples)

    # Convert predictions to class indices
    y_pred_super_idx = np.argmax(y_pred_super, axis=1)
    y_pred_class_idx = np.argmax(y_pred_class, axis=1)

    # Create figure
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 3 * num_samples))

    for i in range(num_samples):
        # Get current sample
        img = x_samples[i]

        # Get true and predicted labels
        true_superclass = SUPERCLASS_NAMES[y_true_super[indices[i]]]
        true_class = CLASS_NAMES[y_true_class[indices[i]]]
        pred_superclass = SUPERCLASS_NAMES[y_pred_super_idx[i]]
        pred_class = CLASS_NAMES[y_pred_class_idx[i]]

        # Display image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"True: {true_class} ({true_superclass})")
        axes[i, 0].axis('off')

        # Create bar chart with both superclass and class predictions
        # Superclass predictions
        superclass_probs = y_pred_super[i]
        axes[i, 1].barh(
            y=np.arange(len(SUPERCLASS_NAMES)) + 0.5,
            width=superclass_probs,
            height=0.4,
            color='skyblue',
            label='Superclass'
        )

        # Mark correct superclass
        if y_pred_super_idx[i] == y_true_super[indices[i]]:
            axes[i, 1].barh(
                y=y_true_super[indices[i]] + 0.5,
                width=superclass_probs[y_true_super[indices[i]]],
                height=0.4,
                color='green'
            )
        else:
            axes[i, 1].barh(
                y=y_true_super[indices[i]] + 0.5,
                width=superclass_probs[y_true_super[indices[i]]],
                height=0.4,
                color='yellow'
            )
            axes[i, 1].barh(
                y=y_pred_super_idx[i] + 0.5,
                width=superclass_probs[y_pred_super_idx[i]],
                height=0.4,
                color='red'
            )

        # Add class prediction for the predicted superclass
        # Find classes belonging to the predicted superclass
        superclass_classes = [
            j for j, v in CLASS_TO_SUPERCLASS.items() if v == y_pred_super_idx[i]
        ]

        # Get probabilities for those classes
        class_probs = y_pred_class[i][superclass_classes]

        # Normalize to sum to 1
        class_probs = class_probs / np.sum(class_probs)

        # Class names for this superclass
        class_names = [CLASS_NAMES[j] for j in superclass_classes]

        # Add text to describe predictions
        pred_correct = (y_pred_super_idx[i] == y_true_super[indices[i]]) and (
                    y_pred_class_idx[i] == y_true_class[indices[i]])
        color = 'green' if pred_correct else 'red'
        axes[i, 1].set_title(f"Prediction: {pred_class} ({pred_superclass})", color=color)

        # Set y-axis labels to superclass names
        axes[i, 1].set_yticks(np.arange(len(SUPERCLASS_NAMES)) + 0.5)
        axes[i, 1].set_yticklabels(SUPERCLASS_NAMES)

        # Set x-axis limits
        axes[i, 1].set_xlim(0, 1)

        # Add a grid
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    save_path = os.path.join(save_dir, 'prediction_visualization.png')
    plt.savefig(save_path)
    plt.close()

    return save_path


def plot_prediction_probabilities(image_path, model, preprocess_func, save_dir="visualizations"):
    """
    Visualize prediction probabilities for a single image.

    Args:
        image_path: Path to the image file
        model: Trained model
        preprocess_func: Function to preprocess the image
        save_dir: Directory to save visualization

    Returns:
        Path to the saved visualization file
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load and preprocess the image
    img_array = preprocess_func(image_path)

    # Make predictions
    y_pred_super, y_pred_class = model.predict(img_array)

    # Get the predicted indices
    pred_superclass_idx = np.argmax(y_pred_super[0])
    pred_class_idx = np.argmax(y_pred_class[0])

    # Get the predicted labels
    pred_superclass = SUPERCLASS_NAMES[pred_superclass_idx]
    pred_class = CLASS_NAMES[pred_class_idx]

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Display image
    axes[0].imshow(img_array[0])
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    # Plot superclass probabilities
    superclass_indices = np.arange(len(SUPERCLASS_NAMES))
    axes[1].bar(superclass_indices, y_pred_super[0], color='skyblue')
    axes[1].set_xticks(superclass_indices)
    axes[1].set_xticklabels(SUPERCLASS_NAMES, rotation=45, ha='right')
    axes[1].set_title(
        f"Superclass Probabilities\nPredicted: {pred_superclass} ({y_pred_super[0][pred_superclass_idx]:.4f})")
    axes[1].grid(True, alpha=0.3)

    # Highlight the class probabilities of the predicted superclass
    # Find classes belonging to the predicted superclass
    superclass_classes = [
        i for i, v in CLASS_TO_SUPERCLASS.items() if v == pred_superclass_idx
    ]

    # Sort classes by prediction probability
    sorted_classes = sorted(superclass_classes, key=lambda i: y_pred_class[0][i], reverse=True)

    # Get class names and probabilities
    sorted_class_names = [CLASS_NAMES[i] for i in sorted_classes]
    sorted_class_probs = [y_pred_class[0][i] for i in sorted_classes]

    # Plot class probabilities
    class_indices = np.arange(len(sorted_classes))
    axes[2].bar(class_indices, sorted_class_probs, color='lightgreen')
    axes[2].set_xticks(class_indices)
    axes[2].set_xticklabels(sorted_class_names, rotation=45, ha='right')
    axes[2].set_title(
        f"Class Probabilities (within {pred_superclass})\nPredicted: {pred_class} ({y_pred_class[0][pred_class_idx]:.4f})")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    img_name = os.path.basename(image_path).split('.')[0]
    save_path = os.path.join(save_dir, f'prediction_{img_name}.png')
    plt.savefig(save_path)
    plt.close()

    return save_path


def plot_class_accuracies(evaluation_results, save_dir="visualizations"):
    """
    Plot the accuracies for each class and superclass.

    Args:
        evaluation_results: Dictionary of evaluation results
        save_dir: Directory to save plots

    Returns:
        Dictionary of saved plot file paths
    """
    os.makedirs(save_dir, exist_ok=True)
    saved_plots = {}

    # Extract class precision from evaluation results
    if 'superclass_precision_per_class' in evaluation_results:
        superclass_precision = evaluation_results['superclass_precision_per_class']
    else:
        superclass_report = evaluation_results.get('superclass_report', {})
        superclass_precision = {
            name: superclass_report.get(name, {}).get('precision', 0)
            for name in SUPERCLASS_NAMES
        }

    # Extract class precision from evaluation results
    if 'class_precision_per_class' in evaluation_results:
        class_precision = evaluation_results['class_precision_per_class']
    else:
        class_report = evaluation_results.get('class_report', {})
        class_precision = {
            name: class_report.get(name, {}).get('precision', 0)
            for name in CLASS_NAMES
        }

    # Plot superclass precision
    plt.figure(figsize=(10, 6))
    superclass_names = list(superclass_precision.keys())
    superclass_values = list(superclass_precision.values())

    # Sort by precision value
    sorted_indices = np.argsort(superclass_values)
    sorted_names = [superclass_names[i] for i in sorted_indices]
    sorted_values = [superclass_values[i] for i in sorted_indices]

    # Plot as horizontal bar chart
    bars = plt.barh(sorted_names, sorted_values, color='skyblue')

    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.01,  # x position
            bar.get_y() + bar.get_height() / 2,  # y position
            f'{width:.2f}',  # label
            va='center'  # vertical alignment
        )

    plt.title('Precision by Superclass')
    plt.xlabel('Precision')
    plt.xlim(0, 1.1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save superclass precision plot
    superclass_plot_path = os.path.join(save_dir, 'superclass_precision.png')
    plt.savefig(superclass_plot_path)
    saved_plots['superclass_precision'] = superclass_plot_path
    plt.close()

    # Plot class precision grouped by superclass
    plt.figure(figsize=(14, 10))

    # Group classes by superclass
    grouped_classes = {}
    for class_idx, class_name in enumerate(CLASS_NAMES):
        superclass_idx = CLASS_TO_SUPERCLASS[class_idx]
        superclass_name = SUPERCLASS_NAMES[superclass_idx]

        if superclass_name not in grouped_classes:
            grouped_classes[superclass_name] = []

        grouped_classes[superclass_name].append((class_name, class_precision[class_name]))

    # Sort by superclass, then by precision within superclass
    all_classes = []
    all_precision = []
    all_colors = []

    color_map = plt.cm.tab10
    for i, (superclass_name, classes) in enumerate(grouped_classes.items()):
        # Sort classes within superclass by precision
        sorted_classes = sorted(classes, key=lambda x: x[1])

        # Add to lists
        for class_name, precision in sorted_classes:
            all_classes.append(class_name)
            all_precision.append(precision)
            all_colors.append(color_map(i % 10))

    # Plot as horizontal bar chart
    bars = plt.barh(all_classes, all_precision, color=all_colors)

    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{width:.2f}',
            va='center'
        )

    # Add legend for superclasses
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_map(i % 10), label=superclass_name)
        for i, superclass_name in enumerate(grouped_classes.keys())
    ]
    plt.legend(handles=legend_elements, loc='lower right')

    plt.title('Precision by Class (Grouped by Superclass)')
    plt.xlabel('Precision')
    plt.xlim(0, 1.1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save class precision plot
    class_plot_path = os.path.join(save_dir, 'class_precision.png')
    plt.savefig(class_plot_path)
    saved_plots['class_precision'] = class_plot_path
    plt.close()

    return saved_plots


def plot_hierarchical_tree(evaluation_results=None, save_dir="visualizations"):
    """
    Plot a hierarchical tree visualization showing the relationship
    between superclasses and classes.

    Args:
        evaluation_results: Optional dictionary of evaluation results
        save_dir: Directory to save plots

    Returns:
        Path to the saved plot file
    """
    os.makedirs(save_dir, exist_ok=True)

    # Create a new figure with a larger size
    plt.figure(figsize=(16, 10))

    # Create a mapping from superclass to classes
    superclass_to_classes = {}
    for class_idx, superclass_idx in CLASS_TO_SUPERCLASS.items():
        if superclass_idx not in superclass_to_classes:
            superclass_to_classes[superclass_idx] = []
        superclass_to_classes[superclass_idx].append(class_idx)

    # Count the number of superclasses and classes
    num_superclasses = len(SUPERCLASS_NAMES)
    num_classes = len(CLASS_NAMES)

    # Get precision values from evaluation results if provided
    superclass_precision = {}
    class_precision = {}

    if evaluation_results:
        if 'superclass_precision_per_class' in evaluation_results:
            superclass_precision = evaluation_results['superclass_precision_per_class']
        else:
            superclass_report = evaluation_results.get('superclass_report', {})
            superclass_precision = {
                name: superclass_report.get(name, {}).get('precision', 0)
                for name in SUPERCLASS_NAMES
            }

        if 'class_precision_per_class' in evaluation_results:
            class_precision = evaluation_results['class_precision_per_class']
        else:
            class_report = evaluation_results.get('class_report', {})
            class_precision = {
                name: class_report.get(name, {}).get('precision', 0)
                for name in CLASS_NAMES
            }

    # Set up the hierarchical tree layout
    tree_height = 10
    tree_width = 16
    root_y = tree_height - 1  # Root at the top

    # Add the root node
    plt.scatter(tree_width / 2, root_y, s=100, color='black')
    plt.text(tree_width / 2, root_y + 0.2, "Root", ha='center', fontsize=12)

    # Place superclass nodes horizontally spaced
    superclass_x = np.linspace(1, tree_width - 1, num_superclasses)
    superclass_y = root_y - 2

    # Draw edges from root to superclasses
    for i in range(num_superclasses):
        plt.plot([tree_width / 2, superclass_x[i]], [root_y, superclass_y], 'k-', alpha=0.5)

    # Draw superclass nodes
    for i in range(num_superclasses):
        superclass_name = SUPERCLASS_NAMES[i]

        # Use precision for node color if available
        if superclass_name in superclass_precision:
            precision = superclass_precision[superclass_name]
            node_color = plt.cm.RdYlGn(precision)
            node_label = f"{superclass_name}\n({precision:.2f})"
        else:
            node_color = 'skyblue'
            node_label = superclass_name

        plt.scatter(superclass_x[i], superclass_y, s=300, color=node_color, edgecolor='black')
        plt.text(superclass_x[i], superclass_y, node_label, ha='center', va='center', fontsize=10)

        # Get classes for this superclass
        classes = superclass_to_classes.get(i, [])
        num_classes_in_superclass = len(classes)

        if num_classes_in_superclass > 0:
            # Place class nodes for this superclass
            class_width = tree_width / (num_superclasses)
            class_spacing = class_width / (num_classes_in_superclass + 1)
            class_x = np.array([superclass_x[i] - class_width / 2 + j * class_spacing
                                for j in range(1, num_classes_in_superclass + 1)])
            class_y = superclass_y - 2

            # Draw edges from superclass to classes
            for j in range(num_classes_in_superclass):
                plt.plot([superclass_x[i], class_x[j]], [superclass_y, class_y], 'k-', alpha=0.5)

            # Draw class nodes
            for j in range(num_classes_in_superclass):
                class_idx = classes[j]
                class_name = CLASS_NAMES[class_idx]

                # Use precision for node color if available
                if class_name in class_precision:
                    precision = class_precision[class_name]
                    node_color = plt.cm.RdYlGn(precision)
                    node_label = f"{class_name}\n({precision:.2f})"
                else:
                    node_color = 'lightgreen'
                    node_label = class_name

                plt.scatter(class_x[j], class_y, s=200, color=node_color, edgecolor='black')
                plt.text(class_x[j], class_y, node_label, ha='center', va='center', fontsize=8)

    # Set plot limits and remove axes
    plt.xlim(0, tree_width)
    plt.ylim(0, tree_height)
    plt.axis('off')

    # Add title
    if evaluation_results:
        plt.title('Hierarchical Classification Structure with Precision Scores', fontsize=14)
    else:
        plt.title('Hierarchical Classification Structure', fontsize=14)

    # Save the plot
    tree_plot_path = os.path.join(save_dir, 'hierarchical_tree.png')
    plt.savefig(tree_plot_path, bbox_inches='tight')
    plt.close()

    return tree_plot_path


if __name__ == "__main__":
    # Test the hierarchical tree visualization
    tree_path = plot_hierarchical_tree()
    print(f"Hierarchical tree visualization saved to {tree_path}")