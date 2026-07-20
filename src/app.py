from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
pipeline = joblib.load("models/churn_pipeline.joblib")

class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def predict(customer: Customer):
    df = pd.DataFrame([customer.dict()])
    prediction = pipeline.predict(df)[0]
    probability = pipeline.predict_proba(df)[0][1]
    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 4)
    }