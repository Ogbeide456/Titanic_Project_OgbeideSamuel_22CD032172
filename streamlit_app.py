import streamlit as st
import pandas as pd
import joblib
import os

st.title("Titanic Survival Prediction")

# Load the model
model_path = os.path.join("model", "titanic_logreg_model.pkl")
if not os.path.exists(model_path):
    st.error("Model file not found: make sure titanic_logreg_model.pkl is in /model")
else:
    model = joblib.load(model_path)

# Form inputs
pclass = st.selectbox("Pclass", [1,2,3])
sex = st.selectbox("Sex", ["male","female"])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
sibsp = st.number_input("Siblings/Spouses aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0)

if st.button("Predict"):
    sex_val = 0 if sex == "male" else 1
    sample = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex_val,
        "Age": age,
        "SibSp": sibsp,
        "Fare": fare
    }])
    pred = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0][1] if hasattr(model, "predict_proba") else None

    if pred == 1:
        st.success("Prediction: Survived")
    else:
        st.error("Prediction: Did Not Survive")

    if prob is not None:
        st.write(f"Survival probability: {prob:.2f}")
