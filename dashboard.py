
import requests
import streamlit as st
import pandas as pd

from src.fuzzy_logic import assess_fuzzy_risk

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="📊",
    layout="wide",
)

st.title("📊 ChurnGuard AI")
st.caption("Customer Churn Prediction & Fuzzy Risk Assessment")
st.divider()

with st.form("customer_form"):
    st.subheader("Customer Information")

    c1, c2, c3 = st.columns(3)

    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"],
        )

    with c2:
        internet = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"],
        )
        backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"],
        )
        device = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"],
        )
        support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"],
        )
        tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"],
        )
        movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"],
        )

    with c3:
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
        )
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly = st.number_input(
            "Monthly Charges (₹)", min_value=0.0,
            max_value=10000.0, value=70.0,
        )
        total = st.number_input(
            "Total Charges (₹)", min_value=0.0,
            max_value=1000000.0, value=840.0,
        )

    submitted = st.form_submit_button(
        "🔍 Predict Churn Risk", type="primary", use_container_width=True
    )

if submitted:
    customer = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": device,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }

    try:
        with st.spinner("Analyzing customer..."):
            response = requests.post(
                API_URL, json=customer, timeout=15
            )
            response.raise_for_status()
            result = response.json()

        probability = float(result["churn_probability"])
        prediction = int(result["churn_prediction"])

        fuzzy = assess_fuzzy_risk(probability, tenure)

        st.divider()
        st.header("Prediction Results")

        a, b, c = st.columns(3)

        with a:
            st.metric("Churn Probability", f"{probability:.1%}")

        with b:
            st.metric(
                "ML Prediction",
                "Likely to Churn" if prediction == 1 else "Likely to Stay",
            )

        with c:
            st.metric("Fuzzy Retention Priority", fuzzy["retention_priority"])

        st.progress(probability)
        st.caption(
            "The probability comes from the trained ML model. "
            "The fuzzy priority is a separate rule-based assessment."
        )

        st.subheader("Fuzzy Risk Memberships")

        risk_df = pd.DataFrame.from_dict(
            fuzzy["risk_membership"], orient="index",
            columns=["Membership"],
        )
        st.bar_chart(risk_df, y="Membership", height=250)

        st.subheader("Fuzzy Membership Values")

        cols = st.columns(3)
        for col, (name, value) in zip(
            cols, fuzzy["risk_membership"].items()
        ):
            col.metric(name, f"{value:.2f}")
            col.progress(float(value))

        st.subheader("Retention Priority Rule Strengths")

        priority_df = pd.DataFrame.from_dict(
            fuzzy["priority_strength"], orient="index",
            columns=["Strength"],
        )
        st.bar_chart(priority_df, y="Strength", height=200)

        with st.expander("How does fuzzy logic work?"):
            st.write(
                "Fuzzy sets allow a customer to partially belong to "
                "multiple categories, such as low, medium, and high risk. "
                "Membership values range from 0 to 1."
            )
            st.write(
                "The system combines churn-risk memberships with customer "
                "tenure memberships using IF-THEN rules. The strongest "
                "result determines the displayed retention priority."
            )
            st.write(
                "Membership values are not churn probabilities. The "
                "fuzzy boundaries are illustrative and should be "
                "validated before real-world use."
            )

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to FastAPI. Start the backend first "
            "using the command shown below."
        )
    except requests.exceptions.RequestException as exc:
        st.error(f"Prediction API error: {exc}")
    except (KeyError, ValueError, TypeError) as exc:
        st.error(f"Unexpected prediction response: {exc}")
