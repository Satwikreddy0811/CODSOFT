# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

# Load the datasets
train_df = pd.read_csv(r"/Users/satwikreddy/Downloads/archive/fraudTrain.csv")
test_df = pd.read_csv(r"/Users/satwikreddy/Downloads/archive/fraudTest.csv")

# Combine train and test data
df = pd.concat([train_df, test_df], axis=0)

# Drop columns that aren't useful for the model
columns_to_drop = ['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 'first', 
                   'last', 'street', 'city', 'state', 'zip', 'lat', 'long', 'job', 'dob', 
                   'trans_num', 'unix_time', 'merch_lat', 'merch_long']
df = df.drop(columns=columns_to_drop)

# Perform one-hot encoding for categorical columns
df_encoded = pd.get_dummies(df, drop_first=True)

# Define features (X) and target (y)
X = df_encoded.drop('is_fraud', axis=1)
y = df_encoded['is_fraud']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features for better performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
rf_predictions = rf_model.predict(X_test)

# Evaluate the model
print("\nRandom Forest Evaluation:")
print("Accuracy:", accuracy_score(y_test, rf_predictions))
print("Classification Report:\n", classification_report(y_test, rf_predictions))

# Plot the confusion matrix
conf_matrix = confusion_matrix(y_test, rf_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Plot the ROC Curve
rf_probabilities = rf_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, rf_probabilities)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Random Forest (AUC = {roc_auc_score(y_test, rf_probabilities):.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("ROC Curve - Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# Function to predict fraud for a new transaction
def predict_fraud(transaction_data):
    input_data = pd.DataFrame([transaction_data], columns=X.columns)
    input_data_scaled = scaler.transform(input_data)
    prediction = rf_model.predict(input_data_scaled)
    return "Fraudulent" if prediction[0] == 1 else "Legitimate"

# Collect user input
def get_user_input():
    print("Enter transaction details for fraud prediction:")
    age = float(input("Age: "))
    income = float(input("Income: "))
    transaction_amount = float(input("Transaction Amount: "))
    # Add other features as needed, keeping them in the same order as in your model
    transaction_data = {
        'age': age,
        'income': income,
        'transaction_amount': transaction_amount,
        # Add other necessary features
    }
    return transaction_data

# Get user input and predict
user_transaction = get_user_input()
result = predict_fraud(user_transaction)
print("\nThe transaction is:", result)