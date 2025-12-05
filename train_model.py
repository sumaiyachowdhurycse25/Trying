import pandas as pd







df = pd.read_csv("clinical_notes_diagnosis_prediction_5000.csv")

X = df["Clinical Notes"]
y = df["Diagnosis"]

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=200))
])

model.fit(X_train, y_train)

import pickle

with open("clinical_diagnosis_model.pkl", "wb") as f:
    pickle.dump(model, f)


