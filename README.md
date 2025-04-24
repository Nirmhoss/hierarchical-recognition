# Hierarchical Object Recognition System

This project implements a hierarchical object recognition system using the CIFAR-10 dataset. It demonstrates how to perform classification at both a coarse level (superclass) and a fine-grained level (class) using a CNN-based deep learning approach.

## Hierarchical Structure

For this project, we define a hierarchical structure for CIFAR-10 by grouping the 10 classes into logical superclasses:

- **Vehicles**: airplane, automobile, ship, truck
- **Animals**: bird, cat, deer, dog, frog, horse

This allows us to demonstrate the hierarchical classification approach, predicting both the superclass (e.g., "Animal") and the specific class (e.g., "Dog").

## Features

- Multi-level hierarchical classification
- Custom CNN architecture with dual outputs
- Weighted loss function to balance superclass and class predictions
- Comprehensive evaluation metrics for both hierarchy levels
- Visualization tools for model performance and predictions
- Support for inference on new images

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/hierarchical-recognition.git
cd hierarchical-recognition
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model with default parameters:

```
python main.py --train
```

Optional arguments:
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size for training (default: 64)
- `--learning_rate`: Initial learning rate (default: 0.001)
- `--superclass_weight`: Weight for superclass loss (default: 0.3)
- `--data_augmentation`: Enable data augmentation (default: True)

### Evaluating the Model

To evaluate a trained model:

```
python main.py --evaluate
```

### Making Predictions

To make predictions on new images:

```
python main.py --predict --image_path path/to/image.jpg
```

### Visualizing Results

To generate visualizations from a trained model:

```
python main.py --visualize
```

## Project Structure

- `data_utils.py`: Dataset handling and preprocessing
- `model.py`: Neural network architecture definition
- `train.py`: Training procedures and loss functions
- `evaluate.py`: Evaluation metrics for hierarchical classification
- `visualize.py`: Visualization of training progress and predictions
- `main.py`: Main entry point integrating all components

## Results

After training, you can expect:
- Superclass accuracy: ~90-95%
- Class accuracy: ~80-85%

Example confusion matrices and visualizations will be saved to the `results/` directory.

## Extending the Project

This code can be extended to work with CIFAR-100 or other datasets with more complex hierarchical structures by modifying the class mappings in `data_utils.py`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.