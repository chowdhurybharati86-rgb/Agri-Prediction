import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load model and preprocessing files
pest_model = load_model("MLP_Pest_Model.keras")
pesticide_model = load_model("MLP_Pesticide_Model.keras")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
pest_binarizer = joblib.load("pest_binarizer.pkl")
pesticide_binarizer = joblib.load("pesticide_binarizer.pkl")

st.set_page_config(
    page_title="Pest & Pesticide Prediction",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 Pest & Pesticide Prediction System")
st.write("AI-Based Agricultural Advisory System")

crop = st.text_input("Crop Name")
rainfall = st.number_input("Rainfall (mm)", min_value=0.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0)

month = st.selectbox(
    "Month",
    ["January","February","March","April","May","June",
     "July","August","September","October","November","December"]
)

if st.button("Predict"):

    st.info("Model loaded successfully.")
    st.write("Prediction code will go here.")

    st.success("Predicted Pest: ...")
    st.success("Recommended Pesticide: ...")