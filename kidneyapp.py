# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import pickle

# Step 2: Load dataset
df = pd.read_csv('kidney_disease.csv')

# Step 3: Strip column names of spaces
df.columns = df.columns.str.strip()

# Step 4: Define target column
target_col = 'classification'  # Replace with your exact column name if different

# Step 5: Create target variable
y = df[target_col].apply(lambda x: 1 if str(x).lower()=='ckd' else 0)

# Step 6: Drop target from features
X = df.drop(target_col, axis=1)

# Step 7: Drop 'id' column if present
if 'id' in X.columns:
    X = X.drop('id', axis=1)

# Step 8: Replace placeholders for missing values
X.replace('?', np.nan, inplace=True)

# Step 9: Separate numeric and categorical columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Step 10: Fill missing values
X[numeric_cols] = X[numeric_cols].astype(float).fillna(X[numeric_cols].mean())
X[categorical_cols] = X[categorical_cols].ffill()

# Step 11: Encode categorical feature columns only
X = pd.get_dummies(X, drop_first=True)

feature_columns = X.columns.tolist()


with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)

# Step 12: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 13: Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 14: Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


with open('kidney_disease_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('kidney_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)


