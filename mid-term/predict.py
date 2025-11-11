import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# --- Configuration ---
# Construct an absolute path to the model file
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = os.path.join(MODEL_DIR, "xgboost_insurance_model.bin")


# --- Pydantic Model for Input Data ---
# This defines the structure of the request body.
# It ensures type checking and provides automatic documentation.
class CustomerData(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str


# --- Initialize FastAPI App ---
app = FastAPI(
    title="Insurance Charge Prediction API",
    description="An API to predict insurance charges using a trained XGBoost model.",
)

# --- Load Model and Preprocessors ---
# Load the model, DictVectorizer, and StandardScaler when the application starts
try:
    with open(MODEL_FILENAME, "rb") as f_in:
        model, dv, scaler = pickle.load(f_in)
    print(f"Model, vectorizer, and scaler loaded successfully from {MODEL_FILENAME}")
except FileNotFoundError:
    print(
        f"Error: Model file not found at {MODEL_FILENAME}. Please run train_xgb.py first."
    )
    model, dv, scaler = None, None, None


# --- Prediction Endpoint ---
@app.post("/predict")
def predict(customer_data: CustomerData):
    """
    Predicts the insurance charge for a given customer.
    - **customer_data**: A JSON object with customer attributes.
    """
    if model is None:
        return {"error": "Model not loaded. Check server logs."}

    # Convert the input data into a dictionary, then into a list of dictionaries
    # as required by the DictVectorizer
    customer_dict = customer_data.dict()
    data_for_dv = [customer_dict]

    # 1. Vectorize the input data
    X_encoded = dv.transform(data_for_dv)

    # 2. Scale the input data
    X_scaled = scaler.transform(X_encoded)

    # 3. Make a prediction (output is on the log scale)
    log_prediction = model.predict(X_scaled)

    # 4. Convert the prediction back to the original dollar scale
    original_prediction = np.expm1(log_prediction)

    # 5. Convert the NumPy float to a standard Python float for JSON serialization
    final_prediction = float(original_prediction[0])

    # Return the prediction as a JSON response
    return {"prediction": np.round(final_prediction, 2)}


# --- Root Endpoint ---
@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Insurance Charge Prediction API. Go to /docs for more information."
    }


# To run this app:
# 1. Make sure you have uvicorn installed: pip install uvicorn
# 2. Run the following command in your terminal:
# uvicorn predict:app --reload
