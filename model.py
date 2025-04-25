"""
model.py - Neural network architecture for hierarchical object recognition

This module defines the CNN-based model architecture with two output heads:
one for superclass (coarse) classification and one for class (fine-grained) classification.
"""

from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from data_utils import CLASS_NAMES, SUPERCLASS_NAMES


def build_hierarchical_model(
    input_shape: Tuple[int, int, int] = (32, 32, 3),
    base_filters: int = 32,
    dropout_rate: float = 0.5,
    l2_reg: float = 0.001,
) -> tf.keras.Model:
    """
    Build a CNN model with hierarchical classification heads.

    This model has two output heads:
    1. Superclass classification (coarse-grained)
    2. Class classification (fine-grained)

    Args:
        input_shape: Shape of input images (height, width, channels)
        base_filters: Number of base filters for convolutional layers
        dropout_rate: Dropout rate for regularization
        l2_reg: L2 regularization factor

    Returns:
        Keras Model with hierarchical classification architecture
    """
    # Input layer
    inputs = Input(shape=input_shape, name="input")

    # Convolutional blocks
    x = Conv2D(
        base_filters,
        (3, 3),
        padding="same",
        kernel_regularizer=l2(l2_reg),
        name="conv1_1",
    )(inputs)
    x = BatchNormalization(name="bn1_1")(x)
    x = Activation("relu", name="relu1_1")(x)

    x = Conv2D(
        base_filters,
        (3, 3),
        padding="same",
        kernel_regularizer=l2(l2_reg),
        name="conv1_2",
    )(x)
    x = BatchNormalization(name="bn1_2")(x)
    x = Activation("relu", name="relu1_2")(x)
    x = MaxPooling2D((2, 2), name="pool1")(x)

    # Second block - double the filters
    x = Conv2D(
        base_filters * 2,
        (3, 3),
        padding="same",
        kernel_regularizer=l2(l2_reg),
        name="conv2_1",
    )(x)
    x = BatchNormalization(name="bn2_1")(x)
    x = Activation("relu", name="relu2_1")(x)

    x = Conv2D(
        base_filters * 2,
        (3, 3),
        padding="same",
        kernel_regularizer=l2(l2_reg),
        name="conv2_2",
    )(x)
    x = BatchNormalization(name="bn2_2")(x)
    x = Activation("relu", name="relu2_2")(x)
    x = MaxPooling2D((2, 2), name="pool2")(x)

    # Third block - double the filters again
    x = Conv2D(
        base_filters * 4,
        (3, 3),
        padding="same",
        kernel_regularizer=l2(l2_reg),
        name="conv3_1",
    )(x)
    x = BatchNormalization(name="bn3_1")(x)
    x = Activation("relu", name="relu3_1")(x)

    x = Conv2D(
        base_filters * 4,
        (3, 3),
        padding="same",
        kernel_regularizer=l2(l2_reg),
        name="conv3_2",
    )(x)
    x = BatchNormalization(name="bn3_2")(x)
    x = Activation("relu", name="relu3_2")(x)

    # Global average pooling
    x = GlobalAveragePooling2D(name="gap")(x)

    # Dropout for regularization
    x = Dropout(dropout_rate, name="dropout")(x)

    # Shared dense layer
    shared_features = Dense(
        256,
        activation="relu",
        kernel_regularizer=l2(l2_reg),
        name="shared_dense",
    )(x)

    # Superclass classification head
    superclass_output = Dense(
        len(SUPERCLASS_NAMES),
        activation="softmax",
        name="superclass_output",
        kernel_regularizer=l2(l2_reg),
    )(shared_features)

    # Class classification head
    class_output = Dense(
        len(CLASS_NAMES),
        activation="softmax",
        name="class_output",
        kernel_regularizer=l2(l2_reg),
    )(shared_features)

    # Create model with two outputs
    model = Model(
        inputs=inputs,
        outputs=[superclass_output, class_output],
        name="hierarchical_cnn",
    )

    return model


def compile_model(
    model: tf.keras.Model,
    learning_rate: float = 0.001,
    superclass_weight: float = 0.3,
) -> tf.keras.Model:
    """
    Compile the model with appropriate loss functions and metrics.

    Args:
        model: Keras Model to compile
        learning_rate: Learning rate for the optimizer
        superclass_weight: Weight for the superclass loss (between 0 and 1)
                         The class loss weight will be (1 - superclass_weight)

    Returns:
        Compiled Keras Model
    """
    # Define loss functions
    losses = {
        "superclass_output": "categorical_crossentropy",
        "class_output": "categorical_crossentropy",
    }

    # Define loss weights for multi-task learning
    loss_weights = {
        "superclass_output": superclass_weight,
        "class_output": 1.0 - superclass_weight,
    }

    # Define metrics
    metrics = {
        "superclass_output": [
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_accuracy"),
        ],
        "class_output": [
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_accuracy"),
        ],
    }

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics,
    )

    return model


def get_model_summary(model: tf.keras.Model) -> str:
    """
    Get a string representation of the model architecture.

    Args:
        model: Keras Model

    Returns:
        String containing the model summary
    """
    # Import io here to avoid unused import at module level
    import io

    summary_buffer = io.StringIO()
    model.summary(print_fn=lambda x: summary_buffer.write(x + "\n"))
    summary_string = summary_buffer.getvalue()
    summary_buffer.close()

    return summary_string
