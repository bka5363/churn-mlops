"""Streamlit UI: fill in a customer, get a churn score from the API."""

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Churn Predictor", page_icon="📉")
st.title("Telecom Churn Predictor")
st.caption("Enter a customer's details to estimate the chance they will leave.")

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 5)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    payment = st.selectbox(
        "Payment method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
with col2:
    phone = st.selectbox("Phone service", ["Yes", "No"])
    lines = st.selectbox("Multiple lines", ["Yes", "No", "No phone service"])
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    monthly = st.number_input("Monthly charges", 0.0, 200.0, 89.9)
    total = st.number_input("Total charges", 0.0, 10000.0, 450.5)
    security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
    support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])

payload = {
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": lines,
    "InternetService": internet,
    "OnlineSecurity": security,
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": support,
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
}

if st.button("Predict", type="primary"):
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        r.raise_for_status()
        out = r.json()
        p = out["churn_probability"]
        st.metric("Churn probability", f"{p*100:.1f}%")
        st.progress(min(p, 1.0))
        (st.error if out["will_churn"] else st.success)(f"Risk: {out['risk'].upper()}")
    except Exception as e:
        st.warning(f"Could not reach the API at {API_URL}. Is it running? ({e})")

st.divider()
st.caption(f"Scoring API: {API_URL}")
