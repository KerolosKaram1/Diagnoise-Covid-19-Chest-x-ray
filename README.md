# Diagnoise-Covid-19-Chest-x-ray
Here’s a template for a README file for your GitHub repository based on the code provided. You can customize it further based on your specific needs and project details.

---

# Chest X-Ray Classification for Pneumonia and COVID-19

This repository contains a Jupyter Notebook for classifying chest X-ray images into three categories: PNEUMONIA, NORMAL, and COVID-19. The notebook leverages transfer learning using the DenseNet-169 model and performs data augmentation to enhance model performance.

## Table of Contents

1. [Overview](#overview)
2. [Setup](#setup)
3. [Data](#data)
4. [Model](#model)
5. [Training](#training)
6. [Evaluation](#evaluation)


## Overview

The goal of this project is to develop a deep learning model that can accurately classify chest X-ray images into one of three categories. The notebook includes the following steps:

- **Data Import**: Download and extract datasets from Kaggle.
- **Data Exploration**: Visualize sample images from each category.
- **Data Augmentation and Preprocessing**: Prepare images for model training with normalization and augmentation.
- **Model Building**: Utilize a pre-trained DenseNet-169 model and add custom layers for classification.
- **Model Training**: Train the model and save the best performing version.
- **Evaluation and Prediction**: Evaluate model performance and visualize results.

## Setup

### Prerequisites

Ensure you have the following libraries installed:

- TensorFlow
- OpenCV
- Matplotlib
- NumPy
- Pandas
- Scikit-learn

You can install these libraries using pip:

```bash
pip install tensorflow opencv-python matplotlib numpy pandas scikit-learn
```

### Data

The datasets are available on Kaggle and are used in this notebook:

1. [Chest X-Ray Images (Pneumonia and COVID-19)](https://www.kaggle.com/datasets/andrewmvd/chest-xray-covid19-pneumonia)
2. [DenseNet-169 Weights](https://www.kaggle.com/datasets/keras/densenet-keras)

## Model

The model uses a pre-trained DenseNet-169 architecture with additional fully connected layers for classification. The training involves:

- Freezing the weights of the pre-trained layers
- Adding custom dense layers and dropout to prevent overfitting
- Using the Adam optimizer and binary cross-entropy loss

## Training

Training is performed with the following configuration:

- **Batch Size**: 256 for training and 50 for testing
- **Epochs**: 30
- **Callbacks**: ReduceLROnPlateau and ModelCheckpoint

## Evaluation

The model is evaluated on a test set and includes the following metrics:

- Accuracy
- Loss

The notebook also generates confusion matrices and visualizations of predictions versus actual labels.

