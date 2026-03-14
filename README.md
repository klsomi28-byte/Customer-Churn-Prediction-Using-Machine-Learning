# Customer Churn Prediction using Machine Learning

## Project Overview

Customer churn refers to customers who stop purchasing from a company. Identifying churn early helps businesses take preventive actions to retain valuable customers.

This project analyzes customer purchase behavior and predicts whether a customer is likely to churn using Machine Learning.

The project combines data processing, machine learning modeling, and business intelligence dashboards.

## Technologies Used
Python  
MySQL  
Scikit-learn  
Power BI  
Pandas  
NumPy  
Matplotlib  
Seaborn  

## Project Workflow
1. Data Collection
Customer, order, and payment datasets were used.

2. Data Preprocessing
Datasets were cleaned and merged to create a unified customer dataset.

3. Feature Engineering
Customer-level features were created:
- Total Orders
- Total Spending
- Average Order Value
- Days Since Last Order

4. Machine Learning Model
A Logistic Regression model was trained to predict customer churn.

5. Model Evaluation
Model performance was evaluated using:
- Accuracy Score
- Classification Report
- Confusion Matrix

6. Data Visualization
An interactive dashboard was created using Power BI to analyze customer churn patterns.

## Machine Learning Model
Algorithm Used:
Logistic Regression

Evaluation Metrics:
Accuracy Score  
Classification Report  
Confusion Matrix

The trained model was saved using a pickle file for future predictions.

## Power BI Dashboard
The dashboard includes:

Customer Churn Distribution  
Revenue Lost from Churned Customers  
Average Orders by Churn Status  
Customer Recency vs Churn  
KPI Cards showing Total Customers, Revenue, and Average Order Value

The dashboard helps businesses understand churn patterns and customer behavior.

## Project Files
customer_churn.py → Main Python ML code  
customer_churn_dataset.csv → Processed dataset  
customer_churn_predictions.csv → Model prediction output  
churn_model.pkl → Saved ML model  
PowerBI_Dashboard.pbix → Power BI dashboard  
Customer_Churn_Project_Report.pdf → Project report  

## Conclusion
This project demonstrates how machine learning can be used to identify customers who are likely to churn. By analyzing historical purchase behavior, businesses can predict churn and take proactive actions to retain valuable customers.

The combination of Python, machine learning, and Power BI provides both predictive insights and interactive business intelligence.

## Author
Sanjana K L
Customer-Churn-Prediction-Using-Machine-Learning
Machine Learning project that predicts customer churn using Python, Scikit-learn, and Power BI. Includes data preprocessing, feature engineering, model training, evaluation, and an interactive business intelligence dashboard.
