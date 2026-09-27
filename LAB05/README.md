# Garment Worker Productivity Prediction

## Overview

This project predicts garment worker productivity using both **Regression** and **Classification**.

Two implementations are compared:

* **Scikit-learn**
* **Manual NumPy/Pandas implementation**

The project evaluates model performance, training time, prediction time, and optimization behavior.

---

## Dataset

**Dataset:** UCI Productivity Prediction of Garment Employees

The dataset contains production-related information such as:

* Department and team
* Targeted productivity
* Overtime
* Incentives
* Idle time
* Number of workers
* Work-in-progress
* Actual productivity

### Tasks

**Regression**

Predict:

`actual_productivity`

Model:

`Linear Regression`

**Classification**

A binary target is created:

```python
MeetsTarget = 1 if actual_productivity >= targeted_productivity
MeetsTarget = 0 otherwise
```

Model:

`Logistic Regression`

`actual_productivity` is excluded from classification features to prevent target leakage.

---

## Workflow

```text
Raw Data
   ↓
Data Preprocessing
   ↓
Stratified Train-Test Split
   ↓
Model Training
   ↓
Prediction
   ↓
Evaluation
   ↓
Manual vs Scikit-learn Comparison
   ↓
Optimization
```

The same fixed train-test samples are used for all model comparisons.

---

## Preprocessing

### Numerical Features

* Median imputation
* Standardization

### Categorical Features

* Most-frequent imputation
* One-hot encoding

Preprocessing parameters are fitted only on the training data to avoid data leakage.

---

# Regression

## Linear Regression

The Scikit-learn model is compared with a manually implemented Linear Regression using NumPy.

The manual implementation uses the normal equation:

$$
\beta=(X^TX)^{-1}X^Ty
$$

The implementation uses `np.linalg.solve()` instead of explicitly calculating the matrix inverse.

### Results

| Model                          |         MAE |        RMSE |          R² | Train Time (s) | Prediction Time (s) |
| ------------------------------ | ----------: | ----------: | ----------: | -------------: | ------------------: |
| Scikit-learn Linear Regression | **0.108968** | **0.145976** | **0.266219** |    **0.013217** |         **0.000355** |
| Manual Linear Regression       | **0.108968** | **0.145976** | **0.266219** |    **0.001220** |         **0.000124** |

---

# Classification

## Logistic Regression

The manual implementation derives Logistic Regression using:

$$
\text{Odds}=\frac{p}{1-p}
$$

$$
\log\left(\frac{p}{1-p}\right)=X\beta
$$

which gives the sigmoid function:

$$
p=\frac{1}{1+e^{-X\beta}}
$$

Gradient descent is used to optimize the coefficients.

The final manual implementation uses **L2 regularization**.

### Results

| Model                            |    Accuracy |   Precision |      Recall |          F1 | Train Time (s) | Prediction Time (s) |
| -------------------------------- | ----------: | ----------: | ----------: | ----------: | -------------: | ------------------: |
| Scikit-learn Logistic Regression | **0.720833** | **0.764706** | **0.891429** | **0.823219** |    **0.014051** |         **0.000974** |
| Manual Logistic Regression (L2)  | **0.741667** | **0.780488** | **0.903955** | **0.837696** |    **6.547149** |         **0.000341** |

---

## Optimization

The manual Logistic Regression was optimized using:

* Learning-rate experiments
* L2 regularization
* Gradient-based convergence checking
* Vectorized NumPy operations

Tested learning rates:

```text
0.001, 0.01, 0.05, 0.1
```

Tested regularization strengths:

```text
0, 0.001, 0.01, 0.1, 1.0
```

The final manual model uses:

```text
Learning rate = 0.001
Lambda = 0.01
Maximum iterations = 50000
Tolerance = 1e-6
```

## Implementation Efficiency

The project compares the execution time of the library and manual implementations.

The manual models demonstrate the mathematical operations explicitly, while Scikit-learn provides optimized implementations designed for efficient numerical computation.

---

## Technologies

* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Jupyter Notebook

---

## Key Learning Outcomes

This project demonstrates:

* Data preprocessing
* Stratified train-test splitting
* Data leakage prevention
* Linear Regression mathematics
* Logistic Regression mathematics
* Sigmoid function
* Gradient descent
* L2 regularization
* Regression and classification metrics
* NumPy vectorization
* Comparison of manual and Scikit-learn implementations
* Model execution-time analysis

