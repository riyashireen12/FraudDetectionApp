import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import RobustScaler

# Load the model and scaler
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit app
st.title('Credit Card Fraud Detection')

st.write("""
This app predicts whether a credit card transaction is fraudulent using machine learning.
""")

# Sidebar for user input
st.sidebar.header('Transaction Details')

# Create input fields for all V features and Time/Amount
def user_input_features():
    v_features = {}
    for i in range(1, 29):
        v_features[f'V{i}'] = st.sidebar.slider(f'V{i}', -20.0, 20.0, 0.0)
    
    time = st.sidebar.number_input('Time (seconds since first transaction)', min_value=0)
    amount = st.sidebar.number_input('Amount', min_value=0.0, format="%.2f")
    
    data = {
        'Time': time,
        'Amount': amount,
        **v_features
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Preprocess the input
def preprocess_input(df):
    # Scale Time and Amount
    df['scaled_amount'] = scaler.transform(df['Amount'].values.reshape(-1,1))
    df['scaled_time'] = scaler.transform(df['Time'].values.reshape(-1,1))
    
    # Drop original Time and Amount
    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    
    # Reorder columns to match training data
    cols = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
            'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
            'scaled_amount', 'scaled_time']
    
    df = df[cols]
    return df

processed_df = preprocess_input(input_df.copy())

# Display user input
st.subheader('Transaction Features')
st.write(processed_df)

# Make prediction
prediction = model.predict(processed_df)
prediction_proba = model.predict_proba(processed_df)

st.subheader('Prediction')
fraud_prob = prediction_proba[0][1] * 100
st.write(f'Probability of being fraudulent: {fraud_prob:.2f}%')

if prediction[0] == 1:
    st.error('Warning: This transaction is predicted to be FRAUDULENT!')
else:
    st.success('This transaction is predicted to be LEGITIMATE.')

# Add some explanations
st.subheader('Model Information')
st.write("""
This model uses XGBoost trained on a highly imbalanced dataset of credit card transactions.
The model was trained on resampled data to better detect fraudulent cases.
""")

# Add ROC curve image
st.subheader('Model Performance')
st.image('roc_curve.png', caption='ROC Curve')