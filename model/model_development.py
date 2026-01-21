import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

print("Loading CSV...")

df = pd.read_csv("train.csv")
print("CSV loaded successfully.")

# Features
X = df[["Pclass", "Sex", "Age", "SibSp", "Fare"]]
y = df["Survived"]

# Encode sex
print("Encoding and cleaning...")
X["Sex"] = X["Sex"].map({"male": 0, "female": 1})

# Fill missing age and fare
X["Age"].fillna(X["Age"].median(), inplace=True)
X["Fare"].fillna(X["Fare"].median(), inplace=True)

print("Preparing pipeline...")

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

print("Splitting data and training model...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

print("\nModel trained successfully.")

print("\nEvaluation:")
print(classification_report(y_test, pipeline.predict(X_test)))

# Save the model
os.makedirs("model", exist_ok=True)
model_path = "model/titanic_logreg_model.pkl"
joblib.dump(pipeline, model_path)

print(f"\nModel saved to: {model_path}")
