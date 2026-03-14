import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

from preprocess import load_data

# load dataset
df = load_data()

# create target variable (who wins)
df["winner"] = (df["home_score"] > df["away_score"]).astype(int)

# features
X = df[["home_score", "away_score"]]

# target
y = df["winner"]

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# create model
model = LogisticRegression()

# train model
model.fit(X_train, y_train)

# save trained model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")