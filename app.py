import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Define the FastAPI app
app = FastAPI()

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ "https://ajaygb.dev"],  # Allow only frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

MODEL_FILE = "medical_prediction_model.pkl"

# Define schema for prediction input
class PredictionInput(BaseModel):
    features: Dict[str, str | float | int]


# Training endpoint
@app.post("/train")
def train_model():
    try:
        # Load dataset
        file_path = "data/medical_history_sample.csv"
        df = pd.read_csv(file_path)

        # Define target columns (diseases to predict)
        target_columns = ["High Blood Pressure", "Kidney Stones"]

        # Define features (all columns except target columns)
        X = df.drop(columns=target_columns)
        y = df[target_columns]

        # Convert categorical variables (Gender and Family History) to numeric
        categorical_columns = ["Gender", "Family History Hypertension", "Family History Kidney Stones",
                               "Previous Kidney Stone", "Smoking", "Alcohol Consumption"]
        for col in categorical_columns:
            if col in X.columns:
                X[col] = X[col].map({"No": 0, "Yes": 1, "Male": 0, "Female": 1})

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale numerical data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Save model, scaler, and feature order
        with open(MODEL_FILE, "wb") as f:
            pickle.dump((model, scaler, X.columns.tolist()), f)

        return {"message": "Model trained successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Prediction endpoint
@app.post("/predict")
def predict(data: PredictionInput):
    try:
        # Load model
        with open(MODEL_FILE, "rb") as f:
            model, scaler, feature_columns = pickle.load(f)

        # Convert input dictionary to DataFrame
        input_df = pd.DataFrame([data.features])

        # Convert categorical values to numeric
        categorical_columns = ["Gender", "Family History Hypertension", "Family History Kidney Stones",
                               "Previous Kidney Stone", "Smoking", "Alcohol Consumption"]
        for col in categorical_columns:
            if col in input_df.columns:
                input_df[col] = input_df[col].map({"No": 0, "Yes": 1, "Male": 0, "Female": 1})

        # Ensure input has the same feature columns as the training data
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0  # Fill missing features with 0

        # Ensure correct column order
        input_df = input_df[feature_columns]

        # Convert all values to float before scaling
        input_df = input_df.astype(float)

        # Scale input data
        input_scaled = scaler.transform(input_df)

        # Get prediction probabilities
        prediction_probs = model.predict_proba(input_scaled)

        # Extract probabilities for "Yes" class (disease presence)
        high_bp_prob = round(float(prediction_probs[0][:, 1][0]) * 100, 2)  # ✅ Fixed indexing
        kidney_stone_prob = round(float(prediction_probs[1][:, 1][0]) * 100, 2)  # ✅ Fixed indexing

        # Generate recommendations based on risk levels
        recommendations = []
        if high_bp_prob > 50:
            recommendations.append("Maintain a low-sodium diet and monitor blood pressure regularly.")
        if kidney_stone_prob > 50:
            recommendations.append("Increase water intake and reduce oxalate-rich foods.")
        if not recommendations:
            recommendations.append("No immediate health risks detected. Maintain a healthy lifestyle!")

        # Create a structured response for frontend
        response = {
            "highBloodPressure": high_bp_prob > 50,  # Boolean risk indicator
            "kidneyStones": kidney_stone_prob > 50,  # Boolean risk indicator
            "diabetes": False,  # Placeholder if diabetes prediction is added later
            "highBloodPressureProbability": str(high_bp_prob),
            "kidneyStonesProbability": str(kidney_stone_prob),
            "predictionConfidence": str(round((high_bp_prob + kidney_stone_prob) / 2, 2)),
            "recommendations": recommendations,
            "modelInfo": {
                "type": "RandomForestClassifier",
                "featuresUsed": feature_columns
            }
        }

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the API locally
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
