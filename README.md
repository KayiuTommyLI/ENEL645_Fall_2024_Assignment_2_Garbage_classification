
# Dataset Analysis and Model Training Repository For ENEL 645 Assignment 2 Garbage Classification

## Overview

This repository contains analysis visualizations and scripts related to a machine learning classification task. The project includes visualizations for the dataset distribution, confusion matrix analysis, and training metrics, along with code for model building and data processing. The key components of this repository include:

-   **Data Distribution Analysis**: Visualizations showing the class distribution across training, validation, and test datasets.
    
-   **Confusion Matrix**: Analysis of the model's performance on the validation dataset using a confusion matrix.
    
-   **Token Length Distribution**: Visualization of the token length distribution during dataset initialization.
    
-   **Training Metrics**: Training and validation metrics visualization for before and after model unfreezing.
    
-   **Scripts**: Python scripts to build and train the model, as well as a Jupyter notebook for further analysis.
    

## Contents

1.  **Images**
    
    -   `Class Distribution in Training Dataset.png`: Bar chart showing the number of instances for each class in the training dataset.
        
    -   `Class Distribution in Validation Dataset.png`: Bar chart showing the number of instances for each class in the validation dataset.
        
    -   `Class Distribution in Test Dataset.png`: Bar chart showing the number of instances for each class in the test dataset.
        
    -   `confusion_matrix.png`: Confusion matrix of model predictions on the validation dataset.
        
    -   `Sample Token Length Distribution (Dataset Initialization).png`: Histogram showing the token length distribution in the dataset.
        
    -   `Training_and_Validation_Metrics_Before_and_After_Unfreeze.png`: Line plot showing training and validation metrics before and after unfreezing the model layers.
        
2.  **Scripts**
    
    -   `BuildModel.py`: Python script for constructing the machine learning model.
        
    -   `run.slurm`: Slurm script for running the model training on an HPC cluster.
        
3.  **Jupyter Notebook**
    
    -   `Assignment2.ipynb`: Jupyter notebook containing further analysis and experiments.
        

## Usage

-   To build and train the model, use `BuildModel.py`. It is recommended to adjust hyperparameters as needed based on the dataset.
    
-   For running the model on an HPC cluster, use the provided `run.slurm` script.
    
-   To explore and analyze the dataset, refer to the visualizations and the Jupyter notebook for deeper insights into the data and model performance.
    

## Requirements

-   Python 3.x
    
-   Required libraries: `tensorflow`, `numpy`, `matplotlib`, `scikit-learn`
    

## Visualizations

The visualizations included in this repository help provide insights into the dataset characteristics and the model's performance:

-   **Class Distribution**: These charts provide an overview of the class imbalance, if any, in the training, validation, and test sets.
    
-   **Confusion Matrix**: Helps identify which classes are more frequently misclassified, providing insight into model weaknesses.
    
-   **Training Metrics**: The training and validation plots before and after unfreezing demonstrate the impact of fine-tuning the model.
    

## License

This project is licensed under the MIT License - see the LICENSE file for details.
