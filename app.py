# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model, scaler, and feature names
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

@app.route("/")
def home():
    return render_template("index.html", features=features)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = [float(request.form[feature]) for feature in features]
        values_scaled = scaler.transform([values])
        prediction = model.predict(values_scaled)[0]
        result = "Malignant" if prediction == 0 else "Benign"
        return render_template("index.html", features=features, result=result)
    except Exception as e:
        return render_template("index.html", features=features, result="Invalid input!")

if __name__ == "__main__":
    app.run(debug=True)
