import pandas as pd
import numpy as np
import pickle
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Load model and scaler from saved files
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Define the FastAPI app
app = FastAPI()

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define feature columns (for input validation)
FEATURE_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age"
]


@app.post("/predict")
def predict_diabetes(data: dict):
    try:
        # Convert input data into DataFrame with correct feature names
        input_df = pd.DataFrame([data], columns=FEATURE_COLUMNS)

        # Standardize input features
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)

        return {"prediction": int(prediction[0])}

    except Exception as e:
        return {"error": str(e)}


# Run the API locally
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
