# ---------------------------------------------
# 1. Import Required Libraries
# ---------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

import pickle

# ---------------------------------------------
# 2. Load Dataset
# ---------------------------------------------
# Replace with your dataset path
df = pd.read_csv("heart.csv")

# ---------------------------------------------
# 3. Drop the restecg feature
# ---------------------------------------------
if "restecg" in df.columns:
    df = df.drop("restecg", axis=1)

# ---------------------------------------------
# 4. Split Features (X) and Target (y)
# ---------------------------------------------
# Assuming the target column is named "target"
X = df.drop("target", axis=1)
y = df["target"]

# ---------------------------------------------
# 5. Train/Test Split
# ---------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------
# 6. Feature Scaling (important for KNN)
# ---------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------
# 7. Build and Train KNN Model
# ---------------------------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# ---------------------------------------------
# 8. Evaluate Model
# ---------------------------------------------
y_pred = knn.predict(X_test_scaled)

# ---------------------------------------------
# 9. Save Model and Scaler with Pickle
# ---------------------------------------------
with open("heart_model.pkl", "wb") as f:
    pickle.dump(knn, f)

with open("heart_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)



