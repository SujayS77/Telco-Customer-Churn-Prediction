# Customer Churn Prediction

## Problem
Telecom companies lose revenue when customers churn (cancel service). This project predicts whether a customer is likely to churn based on their account details, services subscribed, and billing information — enabling proactive retention efforts.

## Dataset
Telco Customer Churn dataset (Kaggle) — ~7,043 customers, 19 features (demographics, services, contract/billing info), binary target (`Churn`: Yes/No). Class distribution is imbalanced (~27% churn).

## Approach
1. **EDA** — explored churn rate by contract type, tenure, payment method, and internet service to identify key drivers.
2. **Data cleaning** — fixed `TotalCharges` (stored as string with blank values for zero-tenure customers), converted to numeric, imputed with 0.
3. **Preprocessing** — built a `ColumnTransformer` combining `StandardScaler` for numeric features and `OneHotEncoder` for categoricals, wrapped in a single `Pipeline` with the model to keep training and inference preprocessing consistent.
4. **Modeling** — trained a `RandomForestClassifier`. Evaluated using precision, recall, and F1 (not just accuracy, given class imbalance).
5. **Serving** — wrapped the trained pipeline in a FastAPI `/predict` endpoint returning churn prediction and probability.
6. **Deployment** — wrote a `Dockerfile` and `requirements.txt` to containerize the service. *(Note: local Docker build pending — Docker Desktop requires virtualization enabled in BIOS, not yet enabled on my pc. The Dockerfile is written and ready; the service runs and serves predictions locally via `uvicorn`.)*


## Project structure
├── data/raw/ # original dataset

├── notebooks/ # EDA and preprocessing exploration

├── src/
│ ├── train.py # training script
│ └── app.py # FastAPI serving app

├── models/ # saved trained pipeline

├── Dockerfile
├── requirements.txt
└── README.md
