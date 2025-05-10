import streamlit as st
import pandas as pd
import joblib

# Load the trained fraud detection model
model = joblib.load("fraud_model.pkl")

# App title
st.title("🛡️ Securelytics – Credit Card Fraud Predictor")

# Sidebar description
st.markdown("""
This app uses a machine learning model to predict whether a credit card transaction is fraudulent.
Fill in the transaction details below and click **Predict**.
""")

# Input features
amount = st.slider("Transaction Amount (€)", 0.0, 5000.0, 100.0)
v_features = [st.slider(f"V{i}", -5.0, 5.0, 0.0) for i in range(1, 29)]

# Construct DataFrame with correct feature names
columns = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
values = [0.0, amount] + v_features
input_df = pd.DataFrame([values], columns=columns)

# Predict and display result
if st.button("Predict"):
    prediction = model.predict(input_df)[0]  # ✅ Pass DataFrame directly
    st.markdown("---")
    if prediction == 1:
        st.error("⚠️ Fraud Detected! Take immediate action.")
    else:
        st.success("✅ Legit Transaction")
