import streamlit as st
import joblib
import numpy as np

# Page settings
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="centered"
)

# Modern Clean UI CSS
clean_css = """
<style>
body {
    background-color: #f7f9fc;
}

h1 {
    text-align: center;
    font-size: 40px;
    color: #0057ff;
    font-weight: 800;
    margin-bottom: 10px;
}

.subtext {
    text-align: center;
    font-size: 18px;
    color: #333333;
    margin-top: -10px;
    margin-bottom: 20px;
}

.input-card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

.stButton>button {
    width: 100%;
    background-color: #0057ff;
    color: white;
    border-radius: 10px;
    padding: 12px;
    font-size: 18px;
    border: none;
    transition: 0.3s;
    margin-top: 10px;
}
.stButton>button:hover {
    background-color: #003bb5;
}

.result-box {
    background: #e8f0ff;
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
    border-left: 6px solid #0057ff;
}
</style>
"""

st.markdown(clean_css, unsafe_allow_html=True)

# Load model + encoder
pipeline = joblib.load("crop_pipeline.pkl")
le = joblib.load("label_encoder.pkl")

# Title
st.markdown("<h1>🌾 Crop Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtext'>Enter the environmental conditions to find the best crop.</p>", unsafe_allow_html=True)

# Input Card
with st.container():
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)

    N = st.number_input("Nitrogen (N)", 0, 200, 50)
    P = st.number_input("Phosphorus (P)", 0, 200, 40)
    K = st.number_input("Potassium (K)", 0, 200, 50)
    temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
    ph = st.number_input("Soil pH Value", 0.0, 14.0, 6.5)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 200.0)

    predict_btn = st.button("Predict Crop 🌱")

    st.markdown("</div>", unsafe_allow_html=True)

# Predict
if predict_btn:
    sample = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    pred = pipeline.predict(sample)
    crop = le.inverse_transform(pred)[0]

    st.markdown(
        f"""
        <div class='result-box'>
            <h3>Recommended Crop: <b>{crop.upper()}</b></h3>
        </div>
        """,
        unsafe_allow_html=True
    )
