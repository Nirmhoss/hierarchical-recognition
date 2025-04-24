"""
data_utils.py - Dataset handling for hierarchical object recognition

This module handles loading, preprocessing, and organizing the CIFAR-10 dataset
into a hierarchical structure for multi-level classification.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Define the hierarchical structure for CIFAR-10
# Original CIFAR-10 classes:
# 0: airplane, 1: automobile, 2: bird, 3: cat, 4: deer,
# 5: dog, 6: frog, 7: horse, 8: ship, 9: truck

# Mapping from class indices to class names
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Mapping from class indices to superclass indices
CLASS_TO_SUPERCLASS = {
    0: 0,  # airplane -> vehicle
    1: 0,  # automobile -> vehicle
    2: 1,  # bird -> animal
    3: 1,  # cat -> animal
    4: 1,  # deer -> animal
    5: 1,  # dog -> animal
    6: 1,  # frog -> animal
    7: 1,  # horse -> animal
    8: 0,  # ship -> vehicle
    9: 0,  # truck -> vehicle
}

# Superclass names
SUPERCLASS_NAMES = ['vehicle', 'animal']


def load_data(validation_split=0.1, normalize=True, one_hot=True):
    """
    Load and preprocess the CIFAR-10 dataset with hierarchical labels.

    Args:
        validation_split: Fraction of training data to use for validation
        normalize: Whether to normalize pixel values to [0, 1]
        one_hot: Whether to convert labels to one-hot encoding

    Returns:
        A tuple of (x_train, y_train_super, y_train_class,
                    x_val, y_val_super, y_val_class,
                    x_test, y_test_super, y_test_class)
    """
    # Load CIFAR-10 dataset
    (x_train_all, y_train_all), (x_test, y_test_class) = cifar10.load_data()

    # Create superclass labels
    y_train_super_all = np.array([CLASS_TO_SUPERCLASS[y[0]] for y in y_train_all])
    y_test_super = np.array([CLASS_TO_SUPERCLASS[y[0]] for y in y_test_class])

    # Reshape label arrays
    y_train_class_all = y_train_all.reshape(-1)
    y_test_class = y_test_class.reshape(-1)

    # Split training data into train and validation sets
    x_train, x_val, y_train_class, y_val_class, y_train_super, y_val_super = train_test_split(
        x_train_all, y_train_class_all, y_train_super_all,
        test_size=validation_split,
        stratify=y_train_class_all,
        random_state=42
    )

    # Normalize pixel values to [0, 1]
    if normalize:
        x_train = x_train.astype('float32') / 255.0
        x_val = x_val.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

    # Convert labels to one-hot encoding
    if one_hot:
        y_train_class = to_categorical(y_train_class, num_classes=len(CLASS_NAMES))
        y_val_class = to_categorical(y_val_class, num_classes=len(CLASS_NAMES))
        y_test_class = to_categorical(y_test_class, num_classes=len(CLASS_NAMES))

        y_train_super = to_categorical(y_train_super, num_classes=len(SUPERCLASS_NAMES))
        y_val_super = to_categorical(y_val_super, num_classes=len(SUPERCLASS_NAMES))
        y_test_super = to_categorical(y_test_super, num_classes=len(SUPERCLASS_NAMES))

    return (x_train, y_train_super, y_train_class,
            x_val, y_val_super, y_val_class,
            x_test, y_test_super, y_test_class)


# First, define the hierarchical_generator function as a standalone function
def hierarchical_generator(x, y_super, y_class, datagen, batch_size):
    """Custom generator that yields both superclass and class labels."""
    num_samples = len(x)
    sample_indices = np.arange(num_samples)

    while True:
        # Shuffle at the start of each epoch
        np.random.shuffle(sample_indices)

        # Create batches from shuffled indices
        for start_idx in range(0, num_samples, batch_size):
            # Get batch indices
            batch_indices = sample_indices[start_idx:min(start_idx + batch_size, num_samples)]

            # Extract samples and labels
            x_batch = x[batch_indices].copy()
            y_super_batch = y_super[batch_indices].copy()
            y_class_batch = y_class[batch_indices].copy()

            # Apply data augmentation if provided
            if datagen is not None:
                for i in range(len(x_batch)):
                    x_batch[i] = datagen.random_transform(x_batch[i])
                    if hasattr(datagen, 'standardize'):
                        x_batch[i] = datagen.standardize(x_batch[i])

            # Yield the batch
            yield x_batch, {'superclass_output': y_super_batch, 'class_output': y_class_batch}


# Then define the create_data_generators function that calls it
def create_data_generators(x_train, y_train_super, y_train_class,
                           x_val, y_val_super, y_val_class,
                           batch_size=32, augment=True):
    """
    Create data generators for training and validation with optional augmentation.

    Args:
        x_train: Training images
        y_train_super: Training superclass labels
        y_train_class: Training class labels
        x_val: Validation images
        y_val_super: Validation superclass labels
        y_val_class: Validation class labels
        batch_size: Batch size for training
        augment: Whether to apply data augmentation

    Returns:
        A tuple of (train_generator, val_generator)
    """
    if augment:
        # Data augmentation for training
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
        )
    else:
        train_datagen = None

    # No augmentation for validation
    val_datagen = None

    # Create generators using our custom hierarchical_generator function
    train_generator = hierarchical_generator(
        x_train, y_train_super, y_train_class,
        train_datagen, batch_size
    )

    val_generator = hierarchical_generator(
        x_val, y_val_super, y_val_class,
        val_datagen, batch_size
    )

    return train_generator, val_generator


def preprocess_image(image_path, target_size=(32, 32)):
    """
    Preprocess a single image for prediction.

    Args:
        image_path: Path to the image file
        target_size: Target size for the image (height, width)

    Returns:
        Preprocessed image as a numpy array of shape (1, height, width, channels)
    """
    from PIL import Image
    import numpy as np

    # Load image
    img = Image.open(image_path)

    # Resize to target size
    img = img.resize(target_size, Image.LANCZOS)

    # Convert to numpy array
    img_array = np.array(img)

    # Ensure image has 3 channels (RGB)
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack([img_array, img_array, img_array], axis=-1)
    elif img_array.shape[2] > 3:  # RGBA or similar
        img_array = img_array[:, :, :3]

    # Normalize to [0, 1]
    img_array = img_array.astype('float32') / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def get_class_distribution():
    """
    Get the distribution of classes and superclasses in the CIFAR-10 dataset.

    Returns:
        A tuple of (class_counts, superclass_counts)
    """
    _, (_, y_test) = cifar10.load_data()

    # Get class distribution
    class_counts = {}
    for i in range(len(CLASS_NAMES)):
        class_counts[CLASS_NAMES[i]] = np.sum(y_test == i)

    # Get superclass distribution
    superclass_counts = {}
    for i in range(len(SUPERCLASS_NAMES)):
        superclass_counts[SUPERCLASS_NAMES[i]] = 0

    for cls, superclass in CLASS_TO_SUPERCLASS.items():
        superclass_name = SUPERCLASS_NAMES[superclass]
        class_count = np.sum(y_test == cls)
        superclass_counts[superclass_name] += class_count

    return class_counts, superclass_counts


if __name__ == "__main__":
    # Test the data loading function
    (x_train, y_train_super, y_train_class,
     x_val, y_val_super, y_val_class,
     x_test, y_test_super, y_test_class) = load_data()

    print(f"Training set: {x_train.shape[0]} samples")
    print(f"Validation set: {x_val.shape[0]} samples")
    print(f"Test set: {x_test.shape[0]} samples")
    print(f"Number of classes: {len(CLASS_NAMES)}")
    print(f"Number of superclasses: {len(SUPERCLASS_NAMES)}")

    # Print the hierarchical structure
    print("\nHierarchical structure:")
    for superclass_idx, superclass in enumerate(SUPERCLASS_NAMES):
        print(f"Superclass {superclass_idx}: {superclass}")
        for class_idx, class_name in enumerate(CLASS_NAMES):
            if CLASS_TO_SUPERCLASS[class_idx] == superclass_idx:
                print(f"  - Class {class_idx}: {class_name}")