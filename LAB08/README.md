
# Banknote Authentication: Decision Tree vs. Linear SVM

## Overview

This project uses the **Banknote Authentication dataset** to compare a
Decision Tree classifier with a Linear Support Vector Machine (SVM).
Both models are evaluated using the same stratified train-test split and
four features:

-   `variance`
-   `skewness`
-   `curtosis`
-   `entropy`

The project also implements core Decision Tree and Linear SVM
calculations from scratch and compares them with scikit-learn
implementations. Nonlinear SVM kernels are explored using the two
features `variance` and `skewness` to visualize decision boundaries.

## Objectives

-   Implement a Decision Tree classifier from scratch.
-   Implement a Linear SVM from scratch.
-   Compare manual implementations with scikit-learn.
-   Explore Linear, Polynomial, and RBF SVM kernels.
-   Evaluate models using accuracy, precision, recall, F1-score, and
    execution time.
-   Discuss interpretability, feature scaling, overfitting, and model
    complexity.

## Dataset

The dataset file used in the notebook is
`data_banknote_authentication.txt`. It contains four numeric features
and a binary class label.

The notebook loads the file without a header and assigns these column
names:

``` python
["variance", "skewness", "curtosis", "entropy", "class"]
```

Place the dataset file in the same directory as the Jupyter Notebook
before running the notebook.

## Methods

### 1. Decision Tree from Scratch

The manual Decision Tree uses impurity-based splitting (Gini impurity
and entropy are explored) and recursive tree construction. The final
four-feature comparison uses Gini impurity with a maximum depth of 5.

### 2. Linear SVM from Scratch

The manual Linear SVM uses hinge loss and subgradient descent to learn
the weight vector and bias. The labels are represented as `-1` and `+1`
for SVM training.

### 3. scikit-learn Comparisons

The notebook compares the manual implementations with scikit-learn's
`DecisionTreeClassifier` and `SVC(kernel="linear")`.

### 4. Nonlinear SVM Kernels

The notebook compares these kernels on the two-feature experiment:

-   **Linear:** learns a straight decision boundary.
-   **Polynomial (degree 3):** can represent curved boundaries.
-   **RBF:** can represent flexible nonlinear boundaries.

## Data Split and Evaluation

The final comparison uses an 80:20 stratified train-test split with
`random_state=42`. All four models are evaluated on the same test set.

The metrics are:

-   **Accuracy:** fraction of predictions that are correct.
-   **Precision:** fraction of predicted positive cases that are truly
    positive.
-   **Recall:** fraction of actual positive cases correctly identified.
-   **F1-score:** harmonic mean of precision and recall.
-   **Training time:** elapsed time to fit each model.
-   **Prediction time:** measured separately for predictions on the test
    set.

## Results

### Four-feature model comparison

The following results were obtained in the notebook:

  -------------------------------------------------------------------------------
  Model              Accuracy    Precision       Recall     F1-score     Training
                                                                             time
                                                                        (seconds)
  -------------- ------------ ------------ ------------ ------------ ------------
  Manual               0.9927       0.9839       1.0000       0.9919       1.1928
  Decision Tree                                                      

  scikit-learn         0.9927       0.9839       1.0000       0.9919       0.0066
  Decision Tree                                                      

  Manual Linear        0.9709       0.9385       1.0000       0.9683       0.2711
  SVM                                                                

  scikit-learn         0.9855       0.9683       1.0000       0.9839       0.0095
  Linear SVM                                                         
  -------------------------------------------------------------------------------

**Note:** Times depend on hardware, software versions, and runtime
conditions. Prediction times are measured separately in the notebook and
are not included in this table.

### SVM kernel comparison

Using the two features `variance` and `skewness`, the notebook reported:

  Kernel                    Test accuracy   Training time (seconds)
  ----------------------- --------------- -------------------------
  RBF                              0.9127                    0.0072
  Linear                           0.8764                    0.0331
  Polynomial (degree 3)            0.8727                    0.0093

In this two-feature experiment, the RBF kernel achieved the highest test
accuracy. A nonlinear kernel may help when the relationship between
features and classes cannot be represented well by a straight boundary.

## Discussion

-   **Performance:** The Decision Tree achieved the highest accuracy and
    F1-score in the four-feature comparison. Both manual and
    scikit-learn Decision Trees produced the same reported metrics.
-   **Implementation speed:** The scikit-learn implementations trained
    substantially faster than the manual implementations in this
    experiment.
-   **Interpretability:** Decision Trees can be interpreted through
    feature-based splitting rules. A Linear SVM uses a weighted decision
    boundary and is generally less straightforward to explain.
-   **Feature scaling:** Decision Trees generally do not require feature
    scaling. Linear SVMs are sensitive to feature scales, so
    standardization is usually recommended.
-   **Overfitting:** Deep Decision Trees can overfit; limiting tree
    depth can help. SVM regularization, controlled by `C`, affects the
    trade-off between margin size and training errors.
-   **Model complexity:** Decision Trees can represent nonlinear
    decision rules but may become large. A Linear SVM has a linear
    boundary; nonlinear kernels can model more complex boundaries.
-   **Generalization:** Results are based on one train-test split.
    Cross-validation would provide a stronger assessment of how
    consistently the models generalize.


## Conclusion

This project demonstrates the implementation and comparison of Decision
Tree and Linear SVM classifiers, plus an exploration of nonlinear SVM
kernels. On the reported four-feature test split, the Decision Tree
achieved **99.27% accuracy** and a **99.19% F1-score**. The RBF kernel
performed best in the separate two-feature kernel experiment, with
**91.27% accuracy**.

These results apply to the particular split and settings used in the
notebook; further validation is recommended before drawing broader
conclusions.
