# Feature Extraction and Machine Learning with Image and Text Data

## Overview

This project applies **traditional machine learning and feature extraction** to image and text data.

### Part A — Asphalt Crack Classification

* Dataset: 400 images

  * 200 Crack
  * 200 Non-Crack
* Images converted to grayscale and resized to 128×128.
* Extracted:

  * Mean, standard deviation, variance
  * Minimum, maximum, median intensity
  * Dark/bright pixel ratios
  * Canny edge count and density
* Models:

  * Logistic Regression
  * Decision Tree
  * Random Forest
* Metrics:

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * Training/prediction time
  * Confusion matrix


| Model               |  Accuracy |  Precision |  Recall |  F1-Score |  Training Time (s)	 |  Prediction Time (s) |
| ------------------- | ----------: | ----------: | ----------: | ----------: | ----------: | ----------: |
| Logistic Regression |      0.9000 |      0.921053 |      0.875 |      0.897436 |      0.008023 |      0.002550 |
| Decision Tree       |      0.9375 |      0.948718 |      0.925 |      0.936709 |      0.002021 |      0.000604 |
| Random Forest       |      0.9375 |      0.926829 |      0.950 |      0.938272 |      0.152446 |      0.033186 |


### Part B — Email Spam Classification

* Dataset: 5,172 emails
* 3,000 numerical word-frequency features
* The dataset's existing numerical representation was used directly; CountVectorizer and TF-IDF were not used.
* Models:

  * Logistic Regression
  * Decision Tree
  * Random Forest

| Model               |  Accuracy |  Precision |  Recall |  F1-Score |  Training Time (s)	 |  Prediction Time (s) |
| ------------------- | ----------: | ----------: | ----------: | ----------: | ----------: | ----------: |
| Logistic Regression |      0.9826 |      0.9578 |      0.9833 |      0.9704 |      5.1409 |      0.0355 |
| Decision Tree       |      0.9188 |      0.8699 |      0.8467 |      0.8581 |      0.7206 |      0.0128 |
| Random Forest       |      0.9643 |      0.9340 |      0.9433 |      0.9386 |      0.4147 |      0.0415 |


### Part C — Feature Selection

The email representation was reduced from 3,000 to 500 features using:

SelectKBest(score_func=chi2, k=500)

This reduced dimensionality by **83.33%**.

The experiment shows that feature selection reduced the feature space and computational cost, while its effect on predictive performance varied by model.

## Technologies

Python 3.13.3, NumPy, Pandas, OpenCV, PIL, Matplotlib, Scikit-learn, Jupyter Notebook.

## Conclusion

The project demonstrates how appropriate feature selection can transform image and text data into representations suitable for traditional machine learning classification.

## Observations

* Grayscale conversion and resizing provided a consistent image representation for extracting numerical features.
* Intensity-based features and Canny edge features captured useful information for distinguishing crack and non-crack images.
* For email classification, the original 3,000 numerical word-frequency features produced strong classification performance, particularly for Logistic Regression.
* Feature selection reduced the email feature space from 3,000 to 500 features, an 83.33% reduction.
* After feature selection, training and prediction times decreased in the recorded experiments.
* The effect of feature selection on F1-score varied by model: Decision Tree and Random Forest improved, while Logistic Regression decreased.
* Therefore, reducing the number of features can lower computational cost, but it does not necessarily improve predictive performance for every classifier.
