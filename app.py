import streamlit as st
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
import joblib


@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("classification_model.keras")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading model assets: {e}")

# 2. User Interface Design
st.title("Health Classification System (Healthy vs Sick)")
st.write("Upload a `.mat` file containing the data to extract predictions.")

# File uploader
uploaded_file = st.file_uploader("Choose a .mat file", type=["mat"])

if uploaded_file is not None:
    try:
        mat_data = loadmat(uploaded_file)
        
        keys = [k for k in mat_data.keys() if not k.startswith('__')]
        
        if len(keys) == 0:
            st.error("The file is empty or does not contain valid matrices.")
        else:
            feature_name = keys[0]
            data_matrix = mat_data[feature_name]
            
            st.info(f"Matrix detected: `{feature_name}` with shape {data_matrix.shape}")
            
            # Ensure features match (52 features)
            if data_matrix.shape[1] != 52 and data_matrix.shape[0] == 52:
                data_matrix = data_matrix.T
                
            if data_matrix.shape[1] == 52:
                # 3. Data processing and prediction
                scaled_data = scaler.transform(data_matrix)
                predictions = model.predict(scaled_data)
                
                st.subheader("Prediction Results:")
                
                for i, pred in enumerate(predictions):
                    prob = float(pred[0])
                    # 1 represents Healthy, 0 represents Sick
                    status = "Healthy" if prob >= 0.5 else "Sick"
                    confidence = prob if prob >= 0.5 else (1 - prob)
                    
                    st.write(f"Sample {i+1}: **{status}** (Confidence: {confidence*100:.2f}%)")
            else:
                st.error(f"Error: The file must contain exactly 52 features. Current shape: {data_matrix.shape}")
                
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
