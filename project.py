import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
file_path = 'startup_data.csv' 
data = pd.read_csv(file_path)

# Define target and features
target_column = 'funding_total_usd'  # Target variable to predict
categorical_columns = ['status', 'state_code']  # Categorical features
numerical_columns = ['avg_participants', 'funding_rounds']  # Numerical features

# Prepare dataset: filter necessary columns and handle missing values
data_cleaned = data[categorical_columns + numerical_columns + [target_column]].dropna()

# One-Hot Encoding for categorical variables
data_encoded = pd.get_dummies(data_cleaned, columns=categorical_columns, drop_first=True)

# Define Features (X) and Target (y)
X = data_encoded.drop(columns=[target_column])
y = data_encoded[target_column]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Results Output
print("Mean Squared Error:", mse)
print("R-Squared Value:", r2)

# Plot Actual vs Predicted
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Funding")
plt.ylabel("Predicted Funding")
plt.title("Actual vs Predicted Funding")
plt.show()

# Model Coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
