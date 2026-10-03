import pandas as pd
import pickle
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from utils import clean_text

print("Loading datasets...")

fake_df = pd.read_csv("data/Fake.csv")
true_df = pd.read_csv("data/True.csv")

# Add labels
fake_df["label"] = 0
true_df["label"] = 1

# Combine title + text
fake_df["content"] = fake_df["title"] + " " + fake_df["text"]
true_df["content"] = true_df["title"] + " " + true_df["text"]

df = pd.concat([fake_df, true_df])
df = df.sample(frac=1).reset_index(drop=True)

print("Dataset size:", df.shape)

df["content"] = df["content"].apply(clean_text)

X = df["content"]
y = df["label"]

# Vectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_vectorized = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# Train and compare three models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(),
}

comparison = []

for name, clf in models.items():
    print("Training", name, "...")
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    comparison.append({
        "name": name,
        "accuracy": round(accuracy_score(y_test, pred) * 100, 2),
        "f1": round(f1_score(y_test, pred) * 100, 2),
    })

# The web app uses Logistic Regression (it gives confidence scores and word weights)
model = models["Logistic Regression"]
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save metrics for the Accuracy page
report = classification_report(y_test, y_pred, output_dict=True)
metrics = {
    "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
    "precision": round(report["1"]["precision"] * 100, 2),
    "recall": round(report["1"]["recall"] * 100, 2),
    "f1": round(report["1"]["f1-score"] * 100, 2),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    "models": comparison,
}

# Save model
os.makedirs("model", exist_ok=True)

with open("model/metrics.json", "w") as f:
    json.dump(metrics, f)

pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("Model saved successfully")