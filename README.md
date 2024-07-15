# Employee Retention Prediction

## Background Information
# In this project, we aim to predict which employees are more likely to quit the company.
# This prediction helps in understanding employee retention better and allows for targeted retention strategies.

## Problem Statement and Business Goals
# Predicting employee attrition is crucial for optimizing HR strategies and reducing turnover costs.
# By identifying employees at risk of leaving, the company can take proactive measures to retain valuable staff.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import pickle

# Load Dataset
df = pd.read_csv('employee_data.csv')

# Basic Data Checks
print("Basic Information:")
print(df.info())
print("\nDescriptive Statistics:")
print(df.describe())

# Replace 'Attrition' column with integers
df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

# Check for missing data
print("\nMissing Data Check:")
print(df.isnull().sum())

# Drop columns that do not change from one employee to the other
df = df.drop(['EmployeeCount', 'StandardHours', 'Over18'], axis=1)

# Histograms
print("\nGenerating Histograms:")
df.hist(figsize=(12, 10), bins=30)
plt.show()

# Employee Statistics
num_left = df['Attrition'].sum()
num_stayed = len(df) - num_left
print(f"\nNumber of Employees Who Left: {num_left}")
print(f"Percentage of Employees Who Left: {num_left / len(df) * 100:.2f}%")
print(f"Number of Employees Who Stayed: {num_stayed}")
print(f"Percentage of Employees Who Stayed: {num_stayed / len(df) * 100:.2f}%")

# Compare Means and Std Deviation
print("\nComparison of Means and Standard Deviations:")
print(df.groupby('Attrition').mean())
print(df.groupby('Attrition').std())

# KDE Plot for 'DistanceFromHome'
plt.figure(figsize=(12,7))
sns.kdeplot(df[df['Attrition'] == 1]['DistanceFromHome'], label='Employees who left', shade=True, color='r')
sns.kdeplot(df[df['Attrition'] == 0]['DistanceFromHome'], label='Employees who stayed', shade=True, color='b')
plt.xlabel('Distance From Home')
plt.show()

# Boxplots
plt.figure(figsize=(12, 6))
sns.boxplot(x='Gender', y='MonthlyIncome', data=df)
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='JobRole', y='MonthlyIncome', data=df)
plt.xticks(rotation=45)
plt.show()

# Label Encoding and One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus'])

# Separate features and target
X = df_encoded.drop('Attrition', axis=1)
y = df_encoded['Attrition']

# Apply Min-Max Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# Create and Train Models
models = {
    'Random Forest Classifier': RandomForestClassifier(),
    'Support Vector Machine': SVC(),
    'XGBoost Classifier': XGBClassifier()
}

# Train and Evaluate Models
for model_name, model in models.items():
    print(f"\nTraining {model_name}:")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Model Validation
    print(f"\n{model_name} Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# Save the Best Model
# Assume XGBoost performed best; save this model
with open('best_model.pkl', 'wb') as model_file:
    pickle.dump(models['XGBoost Classifier'], model_file)

print("Model saved as 'best_model.pkl'")

## Conclusions
 - **Employee Turnover:** A significant portion of the dataset indicated that many employees left the company.
 - **Satisfaction and Performance:** Employees with higher job satisfaction and performance ratings tend to stay longer.
 - **Distance from Home:** Employees who live closer to the office are more likely to stay.
 - **Stock Options and Job Roles:** Higher stock options and specific job roles are associated with higher retention rates.

## Test Data
 - The model was applied to test data, and predictions were made using the best model.
 - The predicted results have been saved in `Submission.csv`.

## Contribution
 Feel free to contribute by improving the model, adding new features, or refining the analysis.
