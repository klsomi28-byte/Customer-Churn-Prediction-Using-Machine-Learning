import pandas as pd
from sqlalchemy import create_engine
import urllib
import numpy as np
import pickle    #to save ml model.Function	Purpose
#accuracy_score	how correct model is
#classification_report	precision, recall
#confusion_matrix	prediction correctness


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Encode password properly
password = urllib.parse.quote_plus("Somi@2005")

engine = create_engine(f"mysql+mysqlconnector://root:{password}@localhost:3306/project1_db")

print("Connected Successfully ")

# 2. LOAD CSV FILES
# -----------------------------------

customers = pd.read_csv(r"C:\Users\Dell\Desktop\Olist data ana project\olist_customers_dataset.csv")

orders = pd.read_csv(r"C:\Users\Dell\Desktop\Olist data ana project\olist_orders_dataset.csv")

payments = pd.read_csv(r"C:\Users\Dell\Desktop\Olist data ana project\olist_order_payments_dataset.csv")

print("CSV Loaded Successfully")


# -----------------------------------
# 3. SELECT REQUIRED COLUMNS
# -----------------------------------

customers = customers[["customer_id","customer_unique_id"]]

orders = orders[["order_id","customer_id","order_purchase_timestamp"]]

payments = payments[["order_id","payment_value"]]


# -----------------------------------
# 4. UPLOAD TO MYSQL
# -----------------------------------

customers.to_sql("customers", con=engine, if_exists="append", index=False)

orders.to_sql("orders", con=engine, if_exists="append", index=False)

payments.to_sql("payments", con=engine, if_exists="append", index=False)

print("Data Uploaded Successfully")


# -----------------------------------
# 5. MERGE DATA
# -----------------------------------

merged = orders.merge(customers, on="customer_id")

merged = merged.merge(payments, on="order_id")
print(merged);

print("Data Merged Successfully")


# -----------------------------------
# 6. CONVERT DATE COLUMN
# -----------------------------------

merged["order_purchase_timestamp"] = pd.to_datetime(
    merged["order_purchase_timestamp"]
)


# -----------------------------------
# 7. CREATE CUSTOMER FEATURES
# -----------------------------------

customer_data = merged.groupby("customer_unique_id").agg({

    "order_id":"count",
    "payment_value":"sum",
    "order_purchase_timestamp":"max"

}).reset_index()

#changing col names
customer_data.columns = [

    "customer_unique_id",
    "total_orders",
    "total_spent",
    "last_purchase_date"

]

print("Customer Features Created")


# -----------------------------------
# 8. CREATE RECENCY FEATURE
# -----------------------------------

today = merged["order_purchase_timestamp"].max()

customer_data["days_since_last_order"] = (

    today - customer_data["last_purchase_date"]

).dt.days


# -----------------------------------
# 9. CREATE AVERAGE ORDER VALUE
# -----------------------------------

customer_data["avg_order_value"] = (

    customer_data["total_spent"] /

    customer_data["total_orders"]

)


# -----------------------------------
# 10. CREATE CHURN LABEL
# -----------------------------------

customer_data["churn"] = np.where(

    customer_data["days_since_last_order"] > 90,

    1,

    0

)

print("Churn Column Created")


# -----------------------------------
# 11. PREPARE ML DATASET
# -----------------------------------

X = customer_data[

    [

    "total_orders",
    "total_spent",
    "avg_order_value",
    "days_since_last_order"

    ]

]

y = customer_data["churn"]


# -----------------------------------
# 12. FEATURE SCALING
# -----------------------------------

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# -----------------------------------
# 13. TRAIN TEST SPLIT
# -----------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    X_scaled,
    y,
    test_size=0.2,
    random_state=42

)


# -----------------------------------
# 14. TRAIN MODEL
# -----------------------------------

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)

print("Model Training Completed")


# -----------------------------------
# 15. PREDICTIONS
# -----------------------------------

predictions = model.predict(X_test)


# -----------------------------------
# 16. MODEL EVALUATION
# -----------------------------------

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

print("\nClassification Report\n")

print(classification_report(y_test, predictions))

print("\nConfusion Matrix\n")

print(confusion_matrix(y_test, predictions))



# 17. SAVE MODEL
# -----------------------------------

with open("churn_model.pkl","wb") as file:

    pickle.dump(model,file)

print("Model Saved Successfully")

# 18. SAVE PREDICTION OUTPUT
# -----------------------------------

results = pd.DataFrame(
    X_test,
    columns=[
        "total_orders",
        "total_spent",
        "avg_order_value",
        "days_since_last_order"
    ]
)

results["Actual"] = y_test.values

results["Predicted"] = predictions

results.to_csv("customer_churn_predictions.csv",index=False)

print("Prediction File Saved")

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,4))

sns.countplot(x=customer_data["churn"])

plt.title("Customer Churn Distribution")
plt.xlabel("Churn (0 = Active, 1 = Churned)")
plt.ylabel("Number of Customers")

plt.show()

plt.figure(figsize=(8,5))

sns.boxplot(x="churn", y="total_orders", data=customer_data)

plt.title("Total Orders vs Customer Churn")

plt.xlabel("Churn")

plt.ylabel("Total Orders")

plt.show()

plt.figure(figsize=(8,6))

corr = customer_data[[
    "total_orders",
    "total_spent",
    "avg_order_value",
    "days_since_last_order"
]].corr()

sns.heatmap(corr, annot=True)

plt.title("Feature Correlation Heatmap")

plt.show()

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6,4))

sns.heatmap(cm, annot=True, fmt="d")

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

customer_data.to_csv("customer_churn_dataset.csv", index=False)
