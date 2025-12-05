import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import pickle

# --------------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------------
df = pd.read_csv("Parkinsson.csv")  # Change path to your dataset

# --------------------------------------------------------
# 2. Drop unwanted features
# --------------------------------------------------------
drop_cols = {
    'name', 'Shimmer:APQ3', 'MDVP:PPQ', 'MDVP:RAP',
    'Shimmer:APQ5', 'Shimmer:DDA', 'MDVP:Jitter(Abs)',
    'PPE', 'Jitter:DDP', 'MDVP:APQ', 'MDVP:Shimmer(dB)'
}

df = df.drop(columns=drop_cols)

# --------------------------------------------------------
# 3. Split into features (X) and target (y)
# --------------------------------------------------------
X = df.drop("status", axis=1)   # status = 1 (Parkinson’s), 0 (Healthy)
y = df["status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------------
# 4. Train Random Forest model
# --------------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42
)

rf.fit(X_train, y_train)

# --------------------------------------------------------
# 5. Evaluate model
# --------------------------------------------------------
y_pred = rf.predict(X_test)


# --------------------------------------------------------
# 6. Save model with pickle
# --------------------------------------------------------
with open("parkinsons_model.pkl", "wb") as f:
    pickle.dump(rf, f)


