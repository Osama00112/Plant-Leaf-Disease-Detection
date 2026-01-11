# VGG16 Plant Classification with GradCAM Explainability

A comprehensive deep learning pipeline for plant species classification using transfer learning with VGG16, featuring complete training/evaluation workflows and model explainability through activation analysis.

## Research Overview

This project implements a robust plant classification system that:
- Uses **VGG16 transfer learning** for efficient training
- Includes **comprehensive evaluation metrics** with confusion matrices
- Provides **model explainability** through multi-layer activation analysis
- Features **production-ready model saving/loading**
- Achieves **professional-grade visualizations** and reporting

## Key Features

- **Transfer Learning**: Pre-trained VGG16 with frozen features and custom classifier
- **Mixed Precision Training**: Efficient GPU utilization with automatic mixed precision
- **Checkpoint System**: Resume training from interruptions
- **Comprehensive Evaluation**: Training curves, confusion matrix, classification reports
- **Model Explainability**: Multi-layer activation analysis showing decision-making process
- **Production Ready**: Complete model serialization and inference pipeline

## Project Structure
```
Plant-Classification-VGG16/
├── plant_classification_complete.ipynb    # Main Colab notebook
├── README.md                              # This file
├── results/                               # Generated outputs
│   ├── training_curves.png                  # Loss & accuracy plots
│   ├── confusion_matrix.png                 # Classification matrix
│   ├── activation_analysis.png              # Model explanations
│   └── evaluation_report.txt                # Detailed metrics
└── models/                                # Saved models
    └── vgg16_plant_classifier.pth            # Trained model file
```
## Dataset Requirements
### Dataset Structure
This project requires a plant dataset with the following structure:
```
📁 Plant/
├── 📁 Background Removed/
│   ├── 📁 Class1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── 📁 Class2/
│   │   ├── image1.jpg
│   │   └── ...
│   └── 📁 Class3/
│       └── ...
└── 📁 Image Augmentation/
    ├── 📁 Class1/
    ├── 📁 Class2/
    └── 📁 Class3/
```
### Dataset Setup Instructions
**Step 1: Download Dataset**
- Download the plant dataset from: [https://data.mendeley.com/datasets/5g238dv4ht/1]
- Alternative: Use any plant classification dataset with the above folder structure
**Step 2: Upload to Google Drive**
1. Create a folder named `Plant` in your Google Drive root directory
2. Upload both `Background Removed` and `Image Augmentation` folders
3. Ensure the folder structure matches the requirement above
**Step 3: Verify Setup**
- Your Google Drive should have: `My Drive/Plant/Background Removed/` and `My Drive/Plant/Image Augmentation/`
- Each subfolder should contain plant images of the respective class
### Example Dataset Stats
- **Classes**: 4 plant species (customizable)
- **Images per class**: 200+ recommended
- **Image format**: JPG/PNG
- **Input size**: Automatically resized to 224×224
- **Data split**: 80% training, 20% testing
##  Quick Start
### Prerequisites
- Google Colab account
- Google Drive with dataset uploaded
- Basic Python/PyTorch knowledge
### Step-by-Step Execution
**1. Open in Google Colab**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NCwHiX_dyb1w1srECZ4ukHRrZNELKPnj?usp=sharing)
**2. Setup Runtime**
```python
# Select GPU runtime in Colab
Runtime → Change runtime type → Hardware accelerator → GPU
```
**3. Mount Google Drive**
```python
# This will be done automatically in the notebook
from google.colab import drive
drive.mount('/content/drive')
```
**4. Run All Cells**
- Execute the notebook cells sequentially
- The complete pipeline will run automatically

**5. View Results**
- Training curves and metrics will display inline
- All results saved to `My Drive/Plant/results/`
- Trained model saved to `My Drive/Plant/models/`
  
## Technical Implementation
### Model Architecture
- **Base Model**: VGG16 pre-trained on ImageNet
- **Frozen Layers**: All convolutional features (transfer learning)
- **Trainable Layers**: Final classifier adapted for plant classes
- **Optimization**: Adam optimizer with learning rate 0.001
### Training Configuration
```python
# Key parameters
BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 10
LEARNING_RATE = 0.001
```
### Data Augmentation
- Resize to 224×224
- ImageNet normalization
- Combined original + augmented datasets for robust training
## Results & Evaluation
### Generated Outputs
**1. Training Metrics**
- Real-time loss and accuracy tracking
- Professional training curves visualization
- Epoch-by-epoch progress monitoring
**2. Model Evaluation**
- Comprehensive confusion matrix
- Per-class precision, recall, F1-scores
- Overall test accuracy and loss
**3. Model Explainability**
- Multi-layer activation analysis
- Feature map visualizations
- Decision-making process insights
### Performance Expectations
- **Training Time**: ~10-15 minutes (Colab GPU)
- **Expected Accuracy**: 85-95% (depends on dataset quality)
- **Memory Usage**: ~2-4GB GPU memory
## Model Explainability
### Activation Analysis
The notebook includes sophisticated model explainability through:
- **Multi-layer feature visualization**: Shows what different network layers detect
- **Activation heatmaps**: Highlights important image regions
- **Decision process insights**: Understand model focus areas
### Example Output
```python
# Generated visualizations show:
- Original plant images
- Layer-wise feature activations  
- Heatmap overlays showing model attention
- Prediction confidence scores
```


