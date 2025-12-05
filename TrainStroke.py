import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

import pickle

# -------------------------------
# 1. LOAD DATA
# -------------------------------
df = pd.read_csv("stroke-data.csv", encoding='latin1')  # replace with your filename

# -------------------------------
# 2. HANDLE CATEGORICAL FEATURES
# -------------------------------
label_encoder = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = label_encoder.fit_transform(df[col].astype(str))

# -------------------------------
# 3. SPLIT INTO FEATURES + LABEL
# -------------------------------
X = df.drop(["id", "stroke"], axis=1)

y = df["stroke"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 4. TRAIN DECISION TREE
# -------------------------------
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# -------------------------------
# 5. EVALUATE MODEL
# -------------------------------
predictions = model.predict(X_test)

# -------------------------------
# 6. SAVE MODEL WITH PICKLE
# -------------------------------
with open("stroke_model.pkl", "wb") as f:
    pickle.dump(model, f)



