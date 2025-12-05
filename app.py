import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
df = pd.read_csv("diabetes.csv")
df.head()
X = df.drop(["SkinThickness", "Outcome"], axis=1)

y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
preds = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))
with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("diabetes_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
