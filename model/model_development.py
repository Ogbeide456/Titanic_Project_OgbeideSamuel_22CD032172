import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import os

df = pd.read_csv("train.csv")

X = df[["Pclass", "Sex", "Age", "SibSp", "Fare"]]
y = df["Survived"]

X["Sex"] = X["Sex"].map({"male": 0, "female": 1})
X["Age"].fillna(X["Age"].median(), inplace=True)
X["Fare"].fillna(X["Fare"].median(), inplace=True)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

print(classification_report(y_test, pipeline.predict(X_test)))

os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, "model/titanic_logreg_model.pkl")
