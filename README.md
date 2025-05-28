# Chest X-Ray Pneumonia Detection using CNN

This project implements a Convolutional Neural Network (CNN) using transfer learning to detect pneumonia from chest X-ray images. The model is built using PyTorch and leverages a pre-trained ResNet-18 architecture.

## Project Overview

The system is designed to classify chest X-ray images into two categories:
- NORMAL: Healthy chest X-rays
- PNEUMONIA: Chest X-rays showing signs of pneumonia

## Technical Details

- **Framework**: PyTorch
- **Base Model**: ResNet-18 (pre-trained)
- **Input Size**: 224x224 pixels
- **Data Augmentation**: Includes random horizontal flips, rotations, affine transformations, and color jittering
- **Training Strategy**: Transfer learning with frozen base layers and custom classifier head

## Requirements

```
torch
torchvision
numpy
matplotlib
scikit-learn
seaborn
```

## Project Structure

The project expects the following directory structure:
```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── val/
    ├── NORMAL/
    └── PNEUMONIA/
```

## Model Architecture

The model uses a pre-trained ResNet-18 with:
- Frozen base layers
- Custom classifier head with:
  - Linear layer (512 units)
  - ReLU activation
  - Dropout (0.3)
  - Final classification layer (2 units)

## Training

The model is trained with:
- Batch size: 32
- Optimizer: Adam
- Learning rate: 0.001
- Learning rate scheduler: ReduceLROnPlateau
- Loss function: Cross Entropy Loss
- Number of epochs: 10

## Evaluation Metrics

The model's performance is evaluated using:
- Accuracy
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)
- ROC Curve and AUC Score

## Output Visualizations

The training process generates several visualization files:
- `training_history.png`: Training and validation loss/accuracy curves
- `confusion_matrix.png`: Confusion matrix heatmap
- `roc_curve.png`: ROC curve with AUC score

## Usage

1. Prepare your data in the required directory structure
2. Run the Jupyter notebook `final_cnn.ipynb`
3. The model will train and generate performance visualizations
4. Results and model performance metrics will be displayed at the end

## Performance Monitoring

The training process includes:
- Real-time loss and accuracy tracking
- Learning rate adjustment based on validation loss
- Best model checkpoint saving
- Comprehensive evaluation on test set

## Notes

- The code uses CUDA if available, otherwise falls back to CPU
- Data is normalized using ImageNet statistics
- The model implements various data augmentation techniques to prevent overfitting 