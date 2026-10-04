# Flask REST API for the SuperKart sales forecasting model
import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Initialise the Flask application
superkart_api = Flask("SuperKart Sales Forecaster")

# Load the serialized model pipeline (preprocessing + regressor) once at start-up
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "superkart_model.joblib")
model = joblib.load(MODEL_PATH)

# Feature columns expected by the model 
FEATURES = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]


def _clean(df):
    """Light input hygiene: fix the known 'reg' typo and keep only model features."""
    df = df.copy()
    df["Product_Sugar_Content"] = df["Product_Sugar_Content"].replace({"reg": "Regular"})
    return df[FEATURES]


@superkart_api.get("/")
def home():
    """Health-check endpoint."""
    return jsonify({"message": "Welcome to the SuperKart Sales Forecasting API!", "status": "ok"})


@superkart_api.post("/v1/predict")
def predict_sales():
    """Online inference: one JSON record -> one sales prediction."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object with the model features."}), 400
    missing = [c for c in FEATURES if c not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400
    try:
        df = _clean(pd.DataFrame([data]))
        pred = float(model.predict(df)[0])
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 400
    return jsonify({"Predicted_Product_Store_Sales_Total": round(pred, 2)})


@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    """Batch inference: CSV uploaded under the key 'file' -> {row_index: prediction}."""
    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "Upload a CSV file under the key 'file'."}), 400
    try:
        df = pd.read_csv(file)
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            return jsonify({"error": f"CSV is missing columns: {missing}"}), 400
        preds = model.predict(_clean(df))
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 400
    return jsonify({str(i): round(float(p), 2) for i, p in enumerate(preds)})


if __name__ == "__main__":
    # Local run; inside Docker the app is served by gunicorn instead
    superkart_api.run(host="0.0.0.0", port=7860, debug=False)
