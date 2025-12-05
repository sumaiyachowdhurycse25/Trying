# -------------------------------
# 1. Import Required Libraries
# -------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import pickle

# -------------------------------
# 2. Load Your Dataset
# -------------------------------
df = pd.read_csv("thyroid_data.csv", encoding='latin1')   # <-- replace with your dataset filename

# -------------------------------
# 3. Drop Unwanted Features
# -------------------------------
features_to_drop = [
    'S.no',
    'On Thyroxine',
    'Query on Thyroxine',
    'On Antithyroid Medication',
    'I131 Treatment',
    'Query Hypothyroid',
    'Query Hyperthyroid',
    'Lithium',
    'TSH Measured',
    'Hypopituitary',
    'Psych',
    'T3 Measured',
    'TT4 Measured',
    'T4U Measured',
    'FTI Measured',
    'TSH'
]

df = df.drop(columns=features_to_drop, errors='ignore')

# -------------------------------
# 4. Separate Features and Target
# -------------------------------
X = df.drop("Category", axis=1)    # <-- replace "Class" with your label column
y = df["Category"]

# Optional: Convert categorical columns if needed
X = pd.get_dummies(X, drop_first=True)

# -------------------------------
# 5. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 6. Feature Scaling
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 7. Train Logistic Regression
# -------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# -------------------------------
# 8. Evaluate the Model
# -------------------------------
y_pred = model.predict(X_test_scaled)


# -------------------------------
# 9. Save Model & Scaler Using Pickle
# -------------------------------
with open("thyroid_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("thyroid_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)



