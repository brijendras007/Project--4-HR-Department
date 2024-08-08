# Employee Retention Prediction

## Overview

The **Employee Retention Prediction** project aims to identify employees who are more likely to leave the company, using various features from employee data. By predicting which employees are at risk of quitting, this project helps organizations to implement targeted retention strategies and reduce turnover costs.

## Problem Statement

Employee attrition is a significant issue for many companies, affecting their operations and incurring additional costs related to recruitment and training. This project focuses on predicting which employees are likely to leave the company based on historical data, which allows HR departments to proactively address potential issues and retain valuable staff.

## Data Description

The dataset used in this project includes employee-related features such as:

- **Attrition**: Target variable indicating whether an employee left the company (1 for 'Yes', 0 for 'No').
- **DistanceFromHome**: Distance from the employee's home to the office.
- **JobRole**: Employee's job role.
- **MonthlyIncome**: Monthly income of the employee.
- **BusinessTravel**: Frequency of business travel.
- **Department**: Department where the employee works.
- **EducationField**: Field of education of the employee.
- **Gender**: Gender of the employee.
- **MaritalStatus**: Marital status of the employee.
- **StockOptionLevel**: Stock option level of the employee.

## Approach

1. **Data Preprocessing**:
   - Load and explore the dataset.
   - Handle missing values and drop irrelevant columns.
   - Convert categorical variables to numerical formats using label encoding and one-hot encoding.

2. **Exploratory Data Analysis (EDA)**:
   - Generate histograms, KDE plots, and boxplots to visualize the distribution and relationships within the data.

3. **Feature Scaling**:
   - Standardize features using Min-Max Scaling to prepare them for model training.

4. **Model Training**:
   - Train multiple classification models, including Random Forest, Support Vector Machine (SVM), and XGBoost.
   - Evaluate model performance using accuracy, precision, recall, and other metrics.

5. **Model Evaluation and Saving**:
   - Assess the performance of each model.
   - Save the best-performing model (XGBoost) for future use.

## Results

- **Key Findings**: Employees who live closer to the office, have higher job satisfaction, and possess specific job roles are more likely to stay with the company.
- **Model Performance**: The XGBoost Classifier achieved the best performance in terms of accuracy and other metrics.

## Future Work

- **Improvement**: Further enhancements can be made by exploring additional features, trying advanced algorithms, or incorporating external data sources.
- **Contribution**: Contributions to improve the model or add new features are welcome.

## Files

- `employee_data.csv`: Dataset used for analysis.
- `best_model.pkl`: The saved model that achieved the best performance.
- `Submission.csv`: File containing predictions made by the model on test data.

## Contact

For any questions or contributions, please feel free to reach out.
