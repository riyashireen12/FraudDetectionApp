import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import RobustScaler

# App Configuration
st.set_page_config(page_title="Fraud Detection", layout="wide")

# Load Model and Scaler
@st.cache_resource
def load_artifacts():
    try:
        with open('xgb_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        st.stop()

model, scaler = load_artifacts()

# Input Form
st.sidebar.header("Transaction Details")

def get_user_input():
    input_data = {'Time': st.sidebar.number_input('Time (seconds)', min_value=0),
                 'Amount': st.sidebar.number_input('Amount', min_value=0.0, format="%.2f")}
    
    # Add V1-V28 features
    for i in range(1, 29):
        input_data[f'V{i}'] = st.sidebar.slider(f'V{i}', -20.0, 20.0, 0.0)
    
    return pd.DataFrame(input_data, index=[0])

# Preprocessing
def preprocess_input(df):
    try:
        # Scale Time and Amount
        df['scaled_amount'] = scaler.transform(df[['Amount']])
        df['scaled_time'] = scaler.transform(df[['Time']])
        
        # Reorder columns to match training data
        features = [f'V{i}' for i in range(1,29)] + ['scaled_amount', 'scaled_time']
        return df[features]
    except Exception as e:
        st.error(f"Preprocessing error: {str(e)}")
        st.stop()

# Prediction
def make_prediction(features):
    try:
        proba = model.predict_proba(features)[0][1] * 100
        prediction = model.predict(features)[0]
        return prediction, proba
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.stop()

# Main App
def main():
    st.title("Credit Card Fraud Detection")
    
    # User Input
    input_df = get_user_input()
    
    # Display Input
    st.subheader("Transaction Features")
    st.write(input_df)
    
    # Preprocess and Predict
    if st.sidebar.button("Check Fraud Risk"):
        features = preprocess_input(input_df.copy())
        prediction, fraud_prob = make_prediction(features)
        
        # Show Results
        st.subheader("Result")
        st.metric("Fraud Probability", f"{fraud_prob:.2f}%")
        
        if prediction == 1:
            st.error("ALERT: High fraud risk detected!")
        else:
            st.success("Transaction appears legitimate")
    
    # Model Info
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Model Details**:
    - Algorithm: XGBoost
    - Trained on: Kaggle Credit Card Dataset
    - Last updated: May 2025
    """)

if __name__ == "__main__":
    main()