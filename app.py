import streamlit as st
import joblib
import numpy as np

# Load model pipeline & label encoder
pipeline = joblib.load("crop_pipeline.pkl")
le = joblib.load("label_encoder.pkl")

# App Title
st.title("🌾 Crop Recommendation System")
st.write("Enter soil and climate values to get the best crop recommendation.")

# Create input fields
N = st.number_input("Nitrogen (N)", 0, 200, 50)
P = st.number_input("Phosphorus (P)", 0, 200, 50)
K = st.number_input("Potassium (K)", 0, 200, 50)
temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
ph = st.number_input("Soil pH Value", 0.0, 14.0, 6.5)
rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 200.0)

if st.button("🌱 Predict Crop"):
    # Convert to array
    sample = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    # Make prediction
    pred = pipeline.predict(sample)
    crop = le.inverse_transform(pred)[0]

    st.success(f"✔ Recommended Crop: **{crop.upper()}**")
