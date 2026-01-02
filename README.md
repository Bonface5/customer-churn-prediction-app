# CUSTOMER-CHURN-PREDICTION-USING-ML

## Capstone Project: Customer Churn Prediction Using Machine Learning

### Author: Bonface

---

## Project Overview
Customer churn is a major challenge for subscription-based businesses. Accurately predicting which customers are likely to leave helps improve retention and increase revenue. This project develops a machine learning-based churn prediction system to classify customers as likely to churn or remain active.

---

## Business Objective
- Predict customer churn using machine learning models.  
- Compare model performance using F1-score, ROC-AUC, and accuracy.  
- Identify key drivers of churn through feature importance and model interpretation.  
- Provide actionable insights for customer retention strategies.

---

## Data Overview
The dataset includes multiple customer attributes:  
- **Demographics:** Age, Gender  
- **Account Information:** Tenure, Contract Type, Payment Delay, Total Spend, Last Interaction  
- **Customer Behavior:** Usage Frequency, Support Calls  
- **Target Variable:** Churn (0 = Active, 1 = Churned)  

**Preprocessing & Feature Engineering:**  
- Handling missing values  
- Scaling numeric features  
- One-hot encoding categorical variables  
- Creating engineered features:  
  - `Late_Payer` – Customers with delayed payments  
  - `High_Support_User` – Customers with high support call frequency  
  - `Low_Usage_User` – Customers with low service usage  

- Train-test split with churn rate preserved (Stratified)

---

## Modeling Approach
- Evaluated multiple models using **5-fold cross-validation**:  
  - Logistic Regression  
  - Random Forest Classifier  
  - Gradient Boosting  
  - K-Nearest Neighbors  

- **Evaluation Metrics:** Accuracy, F1 Score, ROC-AUC  
- **Final Model:** Random Forest achieved the best performance  

**Random Forest Test Performance:**  
| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.936  |
| F1 Score  | 0.945  |
| ROC-AUC   | 0.953  |

---

## Feature Importance
The Random Forest model highlights the most influential factors for churn:  
1. **Support Calls** – Frequent support calls indicate higher churn risk  
2. **Total Spend** – Spending patterns affect churn likelihood  
3. **Payment Delay** – Late payments correlate with churn  
4. **Age** – Demographics influence churn behavior  
5. **Contract Length (Monthly)** – Short-term contracts increase churn risk  

Other features like `High_Support_User`, `Last Interaction`, and `Late_Payer` also contribute, though with lower relative importance.  

**Insight:** Focus retention efforts on customers with high support calls, late payments, or short-term contracts to effectively reduce churn.

---

## Deliverables
- Preprocessed dataset ready for modeling  
- Jupyter Notebook with full feature engineering, model training, and evaluation  
- Saved model (`final_model.pkl`) and preprocessor (`preprocessor.pkl`) for reuse  
- Feature importance analysis and actionable business insights

---

## Streamlit Web App Features
-User-friendly input form
-Real-time churn probability prediction
-Risk classification:
  -🟢 Low Risk (< 50%)
  -🟡 Medium Risk (50% – 69%)
  -🔴 High Risk (≥ 70%)
-Churn probability explanation
-Feature importance display (Top drivers of churn)

---

## Conclusion
The project successfully built a robust machine learning system for predicting customer churn. Random Forest emerged as the best-performing model, providing accurate predictions and interpretable insights for business decision-making.


