import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Define the FastAPI app
app = FastAPI()

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ajaygb.dev"],  # Allow only frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


MODEL_FILE = "diabetes_model.pkl"


# Training endpoint
@app.post("/train")
def train_model():
    try:
        # Load dataset
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
                   "DiabetesPedigreeFunction", "Age", "Outcome"]
        df = pd.read_csv(url, names=columns)

        # Split data
        X = df.drop(columns=["Outcome"])
        y = df["Outcome"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        # Train model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Save model
        with open(MODEL_FILE, "wb") as f:
            pickle.dump((model, scaler), f)

        return {"message": "Model trained successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Prediction endpoint
@app.post("/predict")
def predict(data: dict):
    try:
        # Load model
        with open(MODEL_FILE, "rb") as f:
            model, scaler = pickle.load(f)

        # Convert input to array
        input_data = np.array([data["features"]]).reshape(1, -1)
        input_data = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        return {"prediction": int(prediction), "probability": float(probability)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the API locally
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)