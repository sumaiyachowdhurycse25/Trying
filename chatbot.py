import pandas as pd

# Replace with your file name
df = pd.read_csv("chatbotfile.csv")

df.head()
# Keep only the important columns
df = df[['Patient', 'Doctor']].dropna()

X = df['Patient']      # input text
y = df['Doctor']       # output text


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Patient'])

from sklearn.neighbors import NearestNeighbors

model = NearestNeighbors(n_neighbors=1, metric='cosine')
model.fit(X)
import pickle

with open("health_chatbot.pkl", "wb") as f:
    pickle.dump((model, vectorizer, df), f)


