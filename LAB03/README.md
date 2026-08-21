Assignment: Scikit-learn: Data Preprocessing and Model Performance Evaluation

Name: Vora Devanshi Suresh

ID: 202618013

Dataset: Kaggle Hotel Booking Demand Dataset

## Preprocessing

* Removed `company` due to very high missing values.
* Removed `reservation_status` and `reservation_status_date` to prevent data leakage.
* Numerical features: `KNNImputer(n_neighbors=5)`.
* Categorical features: `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore")`.
* Pipeline A: KNNImputer + StandardScaler.
* Pipeline B: KNNImputer + MinMaxScaler.
* Used an 80/20 stratified train-test split

## Models

* Logistic Regression (`max_iter=1000`)
* Decision Tree (`random_state=42`)

Both models were tested with both preprocessing pipelines.

## Final Observations

1. Logistic Regression + Pipeline A (StandardScaler) gave the best Logistic Regression result with a testing accuracy of approximately 81.65%.
2. StandardScaler and MinMaxScaler produced very similar Logistic Regression performance, so scaling choice had only a small effect.
3. Scaling had very little effect on the Decision Tree, as expected for a tree-based model.
4. Logistic Regression showed a very small train-test accuracy gap of about 0.003, indicating little evidence of overfitting.
5. Decision Tree showed a much larger gap of about 0.136, indicating noticeable overfitting.
