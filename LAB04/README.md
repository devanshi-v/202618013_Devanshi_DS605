# Airbnb Price Prediction (End-to-End ML Project)

## Project Objective
The objective of this project is to build a complete machine learning workflow to predict Airbnb listing prices and deploy the model through a simple web application.

---

## Dataset Overview

- Dataset Size: 48,895 rows × 16 columns
- Contains Airbnb listings with features such as:
  - Neighbourhood group & location  
  - Room type  
  - Reviews and availability  
  - Host listing details  

### Target Variable
- price (Nightly Airbnb listing price)

---

## Task 1: Data Analysis & Preprocessing

### Steps Performed
- Removed irrelevant columns (ID, name, host_name, etc.)
- Handled missing values using SimpleImputer
- Encoded categorical features using OneHotEncoder
- Scaled numerical features using StandardScaler
- Built preprocessing pipeline using ColumnTransformer

### Key Insights
- Room type has a strong impact on price  
- Neighbourhood group significantly influences pricing  
- Listings with higher availability_365 show variation in price  
- Missing values in reviews_per_month were handled properly  

---

## Task 2: Model Training & Evaluation

### Models Used
- Linear Regression 
- Random Forest Regressor  
- Gradient Boosting Regressor  

---

### Model Performance Comparison

| Model               | Train MAE | Test MAE | Train RMSE | Test RMSE | Train R² | Test R² |
|--------------------|----------|----------|------------|------------|----------|----------|
| Random Forest      | 16.05    | 56.29    | 26.60      | 179.72     | 0.9335   | 0.1932   |
| Gradient Boosting  | 44.21    | 58.11    | 72.24      | 182.64     | 0.5094   | 0.1668   |
| Linear Regression  | 47.91    | 61.25    | 76.50      | 184.36     | 0.4499   | 0.1511   |

---

### Final Model
- Selected Model: Random Forest Regressor  

### Observations
- Random Forest achieved the lowest training error and highest training R²  
- However, there is a large gap between train and test performance, indicating overfitting
- Test R² is relatively low (~0.19), showing limited generalization  

---

### Model Saving
- Trained model saved as: `model.pkl`  
- Preprocessing pipeline saved as: `preprocessor.pkl`  

---

## Task 3: Streamlit Web Application

###  Features
- Simple and interactive UI  
- Accepts Airbnb listing details as input  
- Predicts price in real-time using trained model  

### Inputs
- Room type  
- Number of guests  
- Bedrooms / Bathrooms / Beds  
- Minimum nights  
- Number of reviews  
- Availability  

### Output
- Predicted price per night  


## Deployment
Deployed using Streamlit Cloud  

 https://202618013devanshids605lab4.streamlit.app/

---

## Limitations
- Model shows overfitting (high train accuracy, low test accuracy)
- Limited feature set (no amenities, ratings, etc.)  
- Dataset restricted to a specific location (New York)  
- Outliers in price affect predictions  
