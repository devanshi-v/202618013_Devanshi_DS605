# DS605 Lab 07: Face Recognition using PCA

## Overview
This project uses the AT&T Database of Faces (ORL) to represent face images as numerical vectors, reduce dimensionality using PCA/eigenfaces, and perform face matching.

## Workflow
- Load grayscale face images and subject labels.
- Split the data into training and testing sets.
- Compute eigenfaces using SVD and a smaller covariance matrix.
- Analyze explained variance and reconstruct images using different numbers of components.
- Match test faces using nearest-neighbor search in raw-pixel and PCA spaces.
- Compare PCA matching with KNN, Logistic Regression, and SVM classifiers.
- Use cross-validation and hyperparameter tuning to select model settings.
- Save PCA features and results for further analysis.

## Technologies
Python, NumPy, Matplotlib, Pillow, and scikit-learn.

## Dataset
Dataset: AT&T Database of Faces (ORL)
