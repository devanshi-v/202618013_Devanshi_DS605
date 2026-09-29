# Feature Extraction and Machine Learning with Image and Text Data

## Overview

This project applies **traditional machine learning and feature extraction** to image and text data.

### Part A — Asphalt Crack Classification

* Dataset: **400 images**

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

### Part B — Email Spam Classification

* Dataset: 5,172 emails
* 3,000 numerical word-frequency features
* The dataset's existing numerical representation was used directly; CountVectorizer and TF-IDF were not used.
* Models:

  * Logistic Regression
  * Decision Tree
  * Random Forest

### Part C — Feature Selection

The email representation was reduced from 3,000 to 500 features using:

```python
SelectKBest(score_func=chi2, k=500)
```

This reduced dimensionality by **83.33%**.

| Model               | Original F1 | Improved F1 |
| ------------------- | ----------: | ----------: |
| Logistic Regression |      0.9704 |      0.9404 |
| Decision Tree       |      0.8581 |      0.8811 |
| Random Forest       |      0.9386 |      0.9423 |

The experiment shows that feature selection reduced the feature space and computational cost, while its effect on predictive performance varied by model.

## Technologies

Python 3.13.3, NumPy, Pandas, OpenCV, PIL, Matplotlib, Scikit-learn, Jupyter Notebook.

## Conclusion

The project demonstrates how appropriate **feature extraction and feature selection** can transform image and text data into representations suitable for traditional machine learning classification.

