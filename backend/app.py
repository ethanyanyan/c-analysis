# backend/app.py
from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)
MODEL_PATH = os.path.join("models","elasticnet_best.joblib")
model = joblib.load(MODEL_PATH)

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Expects JSON of form:
      { "Hours_Studied": 5, "Attendance": 85, ..., "Gender": "Male", ... }
    """
    data = request.json
    # model expects the same pipeline inputs as in training
    # e.g. numeric + one-hot categorical columns
    # so you must reconstruct a DataFrame row
    import pandas as pd
    df = pd.DataFrame([data])
    # run pipeline (preprocessor + model)
    y_pred = model.predict(df)[0]
    return jsonify({ "predicted_score": float(y_pred) })

if __name__=="__main__":
    app.run(port=5000)
