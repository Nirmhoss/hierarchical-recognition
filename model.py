"""
model.py - Neural network architecture for hierarchical object recognition

This module defines the CNN-based model architecture with two output heads:
one for superclass (coarse) classification and one for class (fine-grained) classification.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization,
    Dense, Dropout, Flatten, Activation, GlobalAveragePooling2D
)
from tensorflow.keras.regularizers import l2

from data_utils import CLASS_NAMES, SUPERCLASS_NAMES


def create_hierarchical_model(input_shape=(32, 32, 3),
                              num_superclasses=len(SUPERCLASS_NAMES),
                              num_classes=len(CLASS_NAMES),
                              l2_reg=0.001):
    """
    Create a hierarchical CNN model with two output heads.

    Args:
        input_shape: Shape of input images (height, width, channels)
        num_superclasses: Number of superclasses (coarse level)
        num_classes: Number of classes (fine-grained level)
        l2_reg: L2 regularization factor

    Returns:
        A Keras Model with two outputs: one for superclass, one for class
    """
    # Input layer
    inputs = Input(shape=input_shape, name='input')

    # Base CNN layers
    x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(l2_reg))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x)

    # Shared features
    shared_features = GlobalAveragePooling2D()(x)

    # Superclass branch
    superclass_features = Dense(128, kernel_regularizer=l2(l2_reg))(shared_features)
    superclass_features = BatchNormalization()(superclass_features)
    superclass_features = Activation('relu')(superclass_features)
    superclass_features = Dropout(0.5)(superclass_features)
    superclass_output = Dense(num_superclasses, activation='softmax', name='superclass_output')(superclass_features)

    # Class branch (with superclass information)
    # We concatenate shared features with superclass prediction to help with class prediction
    combined_features = tf.keras.layers.concatenate([shared_features, superclass_output])

    class_features = Dense(256, kernel_regularizer=l2(l2_reg))(combined_features)
    class_features = BatchNormalization()(class_features)
    class_features = Activation('relu')(class_features)
    class_features = Dropout(0.5)(class_features)
    class_output = Dense(num_classes, activation='softmax', name='class_output')(class_features)

    # Create model with two outputs
    model = Model(inputs=inputs, outputs=[superclass_output, class_output])

    return model


def create_hierarchical_resnet(input_shape=(32, 32, 3),
                               num_superclasses=len(SUPERCLASS_NAMES),
                               num_classes=len(CLASS_NAMES)):
    """
    Create a hierarchical model based on a pre-trained ResNet50.
    This is an alternative model that leverages transfer learning.

    Args:
        input_shape: Shape of input images
        num_superclasses: Number of superclasses (coarse level)
        num_classes: Number of classes (fine-grained level)

    Returns:
        A Keras Model with two outputs
    """
    # Load ResNet50 without the top layers
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )

    # Freeze the base model layers
    base_model.trainable = False

    # Add custom layers on top
    inputs = Input(shape=input_shape)

    # Preprocess input
    x = tf.keras.applications.resnet50.preprocess_input(inputs)

    # Run through the base model
    x = base_model(x, training=False)

    # Global average pooling
    x = GlobalAveragePooling2D()(x)

    # Superclass branch
    superclass_features = Dense(128)(x)
    superclass_features = BatchNormalization()(superclass_features)
    superclass_features = Activation('relu')(superclass_features)
    superclass_features = Dropout(0.5)(superclass_features)
    superclass_output = Dense(num_superclasses, activation='softmax', name='superclass_output')(superclass_features)

    # Class branch (with superclass information)
    combined_features = tf.keras.layers.concatenate([x, superclass_output])

    class_features = Dense(256)(combined_features)
    class_features = BatchNormalization()(class_features)
    class_features = Activation('relu')(class_features)
    class_features = Dropout(0.5)(class_features)
    class_output = Dense(num_classes, activation='softmax', name='class_output')(class_features)

    # Create model with two outputs
    model = Model(inputs=inputs, outputs=[superclass_output, class_output])

    return model


def compile_model(model, learning_rate=0.001, superclass_weight=0.3):
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
        'superclass_output': 'categorical_crossentropy',
        'class_output': 'categorical_crossentropy'
    }

    # Define loss weights for multi-task learning
    loss_weights = {
        'superclass_output': superclass_weight,
        'class_output': 1.0 - superclass_weight
    }

    # Define metrics
    metrics = {
        'superclass_output': ['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_accuracy')],
        'class_output': ['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')]
    }

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics
    )

    return model


if __name__ == "__main__":
    # Test the model creation functions
    model = create_hierarchical_model()
    model = compile_model(model)

    model.summary()

    # Print model outputs
    print(f"\nModel inputs: {model.input_shape}")
    print(f"Model outputs:")
    for output in model.outputs:
        print(f"  - {output.name}: {output.shape}")