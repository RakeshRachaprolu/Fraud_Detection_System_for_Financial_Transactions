import streamlit as st
import pandas as pd
import joblib

# Streamlit app title
st.title("Financial Fraud Detection")

# Description
st.markdown("""
Enter the details of a financial transaction to predict whether it is fraudulent.
The model uses a pre-trained XGBoost classifier to provide real-time predictions.
""")

# Input fields for transaction details
st.header("Transaction Details")
step = st.number_input("Step (time in hours, 1-744)", min_value=1, max_value=744, value=100)
type_txn = st.selectbox("Transaction Type", ["CASH-IN", "CASH-OUT", "DEBIT", "PAYMENT", "TRANSFER"])
amount = st.number_input("Amount", min_value=0.0, value=250000.0)
nameOrig = st.text_input("Originator ID", value="C123")
oldbalanceOrg = st.number_input("Originator Balance Before", min_value=0.0, value=300000.0)
newbalanceOrig = st.number_input("Originator Balance After", min_value=0.0, value=50000.0)
nameDest = st.text_input("Destination ID", value="C456")
oldbalanceDest = st.number_input("Destination Balance Before", min_value=0.0, value=100000.0)
newbalanceDest = st.number_input("Destination Balance After", min_value=0.0, value=350000.0)

# Features list (must match training)
features = [
    'amount',
    'oldbalanceOrg',
    'newbalanceOrig',
    'oldbalanceDest',
    'newbalanceDest',
    'balance_diff_orig',
    'balance_diff_dest',
    'amount_to_balance_ratio',
    'hour',
    'day',
    'is_night',
    'type_encoded',
    'is_merchant',
    'orig_txn_count'
]

# Load pre-trained pipeline and encoder
try:
    pipeline = joblib.load('fraud_detection_pipeline.pkl')
    le_type = joblib.load('type_encoder.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'fraud_detection_pipeline.pkl' and 'type_encoder.pkl' are in the same directory.")
    st.stop()

# Validate transaction type
valid_types = ['CASH-IN', 'CASH-OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
if type_txn not in valid_types:
    st.error(f"Invalid transaction type. Please select one of: {valid_types}")
    st.stop()

# Predict button
if st.button("Predict"):
    # Create single transaction DataFrame
    sample_transaction = {
        'step': step,
        'type': type_txn,
        'amount': amount,
        'nameOrig': nameOrig,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'nameDest': nameDest,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest
    }
    df_single = pd.DataFrame([sample_transaction])

    # Feature engineering
    df_single['hour'] = df_single['step'] % 24
    df_single['day'] = df_single['step'] // 24
    df_single['is_night'] = ((df_single['hour'] >= 22) | (df_single['hour'] <= 5)).astype(int)
    df_single['balance_diff_orig'] = df_single['oldbalanceOrg'] - df_single['newbalanceOrig']
    df_single['balance_diff_dest'] = df_single['newbalanceDest'] - df_single['oldbalanceDest']
    df_single['amount_to_balance_ratio'] = df_single['amount'] / (df_single['oldbalanceOrg'] + 1e-6)
    df_single['is_merchant'] = df_single['nameDest'].str.startswith('M').astype(int)
    df_single['type_encoded'] = le_type.transform([sample_transaction['type']])[0]
    df_single['orig_txn_count'] = 1  # Single transaction

    # Prepare features for prediction
    X_single = df_single[features]
    
    # Make prediction
    prediction = pipeline.predict(X_single)[0]
    probability = pipeline.predict_proba(X_single)[0][1]

    # Display results
    st.header("Prediction Results")
    st.write(f"**Prediction**: {'Fraud' if prediction == 1 else 'Non-Fraud'}")
    st.write(f"**Fraud Probability**: {probability:.4f}")

    # Provide interpretation
    if prediction == 1:
        st.warning("This transaction is predicted to be fraudulent. Consider flagging for further review.")
    else:
        st.success("This transaction is predicted to be non-fraudulent.")