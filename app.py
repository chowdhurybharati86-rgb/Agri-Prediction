import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# -----------------------------
# Load trained models and files
# -----------------------------

@st.cache_resource
def load_files():
    pest_model = load_model("MLP_Pest_Model.keras")
    pesticide_model = load_model("MLP_Pesticide_Model.keras")

    scaler = joblib.load("scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    pest_binarizer = joblib.load("pest_binarizer.pkl")
    pesticide_binarizer = joblib.load("pesticide_binarizer.pkl")

    return (
        pest_model,
        pesticide_model,
        scaler,
        label_encoders,
        pest_binarizer,
        pesticide_binarizer,
    )

(
    pest_model,
    pesticide_model,
    scaler,
    label_encoders,
    pest_binarizer,
    pesticide_binarizer,
) = load_files()

# -----------------------------
# Streamlit UI
# -----------------------------

st.title("🌱 Agri Prediction System")
st.write("Predict possible pests and pesticides based on agricultural conditions")

st.subheader("Enter Crop Details")

crop = st.text_input("Crop")
state = st.text_input("State")
month = st.number_input("Month", min_value=1, max_value=12, value=1)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0)

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict"):

    try:

        input_data = pd.DataFrame({
            "crop": [crop],
            "state": [state],
            "month": [month],
            "rainfall": [rainfall],
            "temperature": [temperature],
            "humidity": [humidity]
        })

        # Encode categorical columns
        for col in ["crop", "state"]:
            if col in label_encoders:

                le = label_encoders[col]

                if input_data[col][0] in le.classes_:
                    input_data[col] = le.transform(input_data[col])
                else:
                    input_data[col] = 0

        # Scale data
        input_scaled = scaler.transform(input_data)

        # Predict Pest
        pest_prediction = pest_model.predict(input_scaled)

        pest_result = pest_binarizer.inverse_transform(
            (pest_prediction > 0.5).astype(int)
        )

        # Predict Pesticide
        pesticide_prediction = pesticide_model.predict(input_scaled)

        pesticide_result = pesticide_binarizer.inverse_transform(
            (pesticide_prediction > 0.5).astype(int)
        )

        st.success("Prediction Completed Successfully!")

        st.subheader("🐛 Predicted Pest")
        st.write(", ".join(pest_result[0]))

        st.subheader("🧪 Recommended Pesticide")
        st.write(", ".join(pesticide_result[0]))

    except Exception as e:
        st.error(f"Error: {e}")