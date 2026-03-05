# Telco Customer Churn Prediction & Model Optimization

This project focuses on predicting customer attrition (churn) using the popular Telco dataset. By identifying high-risk customers, businesses can proactively offer incentives to improve retention. The project emphasizes handling **imbalanced classes** and optimizing for **Precision** to ensure retention efforts are targeted accurately.

---

## 🚀 Project Overview
The notebook follows a structured data science workflow, moving from raw data cleaning to fine-tuned ensemble models.

### 1. Data Preprocessing & Cleaning
* **Data Cleaning:** Handled missing values and removed duplicates to ensure data integrity.
* **Feature Engineering:** Applied **Scaling** (StandardScaler/RobustScaler) to numerical features and **Encoding** to categorical variables to prepare the data for linear and tree-based algorithms.

### 2. Baseline Model (Logistic Regression)
* **Imbalance Handling:** Implemented techniques (such as `class_weight='balanced'` or resampling) to address the skewed nature of the dataset.
* **Evaluation Focus:** Prioritized **Precision** to minimize "false alarms" in fraud detection.

### 3. Advanced Ensemble Modeling
To improve performance, I trained and optimized high-performance boosters and bagging models:
* **XGBoost (Extreme Gradient Boosting)**
* **Random Forest Classifier**
* **Optimization:** Performed **GridSearchCV** to find the optimal hyperparameters for both models, focusing on depth, learning rate, and estimator counts.

### 4. Model Comparison & Evaluation
Instead of relying on simple accuracy, the models were compared using the **Precision-Recall (PR) Curve**. 
> **Why PR Curve?** In highly imbalanced datasets, the PR Curve provides a more realistic view of a model’s ability to catch the positive class (fraud) compared to a standard ROC curve.
<img width="1622" height="1136" alt="Screenshot 2026-03-05 224432" src="https://github.com/user-attachments/assets/fd321aac-a6cf-4227-ad44-5356a33fbe2b" />


## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib, Seaborn

## 🛠️ What needs to be processed further (Future Scope)
To move this project toward a **production-ready** state, the following advanced methods will be implemented:

### 1. Feature Selection & Engineering
* **Recursive Feature Elimination (RFE):** Automatically removing low-impact features to reduce model noise.
* **Domain-Specific Features:** Creating ratios (e.g., *MonthlyCharges / Tenure*) to capture customer behavior trends over time.

### 2. Advanced Imbalance Handling
* **SMOTE (Synthetic Minority Over-sampling Technique):** Moving beyond class weights by generating synthetic examples of the "Churn" class to improve the decision boundary.
* **Calibration Curves:** Ensuring that the predicted probability (e.g., "80% chance of churn") aligns with the actual observed frequency to make the model reliable for business decision-making.

