import os
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

MODEL_PATH = os.path.join("model", "titanic_logreg_model.pkl")
model = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probability = None

    if request.method == "POST":
        pclass = int(request.form["pclass"])
        sex = 0 if request.form["sex"] == "male" else 1
        age = float(request.form["age"])
        sibsp = int(request.form["sibsp"])
        fare = float(request.form["fare"])

        sample = pd.DataFrame([{
            "Pclass": pclass,
            "Sex": sex,
            "Age": age,
            "SibSp": sibsp,
            "Fare": fare
        }])

        pred = model.predict(sample)[0]
        prob = model.predict_proba(sample)[0][1]

        result = "Survived" if pred == 1 else "Did Not Survive"
        probability = f"{prob:.2f}"

    return render_template("index.html", result=result, probability=probability)

if __name__ == "__main__":
    app.run(debug=True)
