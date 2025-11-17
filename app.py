import streamlit as st
import joblib
import numpy as np

# Page config
st.set_page_config(
    page_title="Crop Recommendation System",
    layout="centered",
    page_icon="🌾"
)

# Custom Dark Theme CSS
dark_theme_css = """
<style>
body {
    background-color: #0d0d0d;
    color: #ffffff;
}
.sidebar .sidebar-content {
    background-color: #111111;
}
.css-18e3th9 {
    background-color: #0d0d0d !important;
}
h1 {
    color: #39ff14 !important;
    text-shadow: 0px 0px 10px #39ff14;
}
label {
    font-size: 18px !important;
    color: #00ffcc !important;
}
.stButton>button {
    background-color: #39ff14;
    color: black;
    border-radius: 10px;
    padding: 10px 25px;
    font-size: 18px;
    border: none;
    transition: 0.3s;
    box-shadow: 0 0 10px #39ff14;
}
.stButton>button:hover {
    background-color: #00cc00;
    box-shadow: 0 0 15px #00ff00;
    transform: scale(1.05);
}
.result-box {
    background-color: #121212;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    border: 1px solid #39ff14;
    box-shadow: 0 0 20px #39ff14;
}
</style>
"""

st.markdown(dark_theme_css, unsafe_allow_html=True)

# Load model + label encoder
pipeline = joblib.load("crop_pipeline.pkl")
le = joblib.load("label_encoder.pkl")

# Title
st.markdown("<h1>🌾 Crop Recommendation System</h1>", unsafe_allow_html=True)
st.write("Enter the environmental values to predict the best crop.")

# Inputs
N = st.number_input("Nitrogen (N)", 0, 200, 50)
P = st.number_input("Phosphorus (P)", 0, 200, 50)
K = st.number_input("Potassium (K)", 0, 200, 50)
temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
ph = st.number_input("Soil pH Value", 0.0, 14.0, 6.5)
rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 200.0)

# Prediction button
if st.button("Predict Crop 🌱"):
    sample = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    pred = pipeline.predict(sample)
    crop = le.inverse_transform(pred)[0]

    st.markdown(
        f"<div class='result-box'><h2>Recommended Crop: {crop.upper()}</h2></div>",
        unsafe_allow_html=True
    )
