import streamlit as st
import joblib
import numpy as np
import pandas as pd
import base64

# PAGE SETTINGS
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide"
)

# BACKGROUND IMAGE
def add_bg(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg("background.jpg")


# ---------------------- ADVANCED CSS UI ----------------------
css = """
<style>

/* White Text Everywhere */
html, body, [class*="css"]  {
    color: white !important;
}

/* Glass Sidebar */
.sidebar .sidebar-content {
    background: rgba(255, 255, 255, 0.10) !important;
    backdrop-filter: blur(10px) !important;
    border-right: 2px solid rgba(255,255,255,0.3);
    height: 100%;
}

.sidebar .sidebar-content h1, h2, h3, p, label {
    color: white !important;
}

/* Sidebar Navigation Buttons */
.stRadio > div {
    background: rgba(255,255,255,0.07);
    padding: 10px;
    border-radius: 10px;
}

.stRadio label {
    font-size: 18px;
    font-weight: 600;
    color: white !important;
}

/* Hover Effect */
.stRadio div:hover {
    background: rgba(255,255,255,0.18);
    transition: 0.3s;
}

/* Title */
.main-title {
    font-size: 42px;
    font-weight: 900;
    text-align: center;
    color: white;
    margin-top: 0px;
    text-shadow: 2px 2px 8px black;
}

/* Input Card */
.input-card {
    background: rgba(0, 0, 0, 0.45);
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.6);
    backdrop-filter: blur(15px);
}

/* Predict Button */
.stButton > button {
    width: 100%;
    background: #00c6ff;
    color: black;
    font-weight: 700;
    border-radius: 8px;
    padding: 12px;
    font-size: 18px;
    border: none;
    transition: 0.3s ease;
}
.stButton > button:hover {
    background: #00a3d5;
}

/* Result Box */
.result-box {
    background: rgba(0, 0, 0, 0.5);
    padding: 18px;
    border-radius: 12px;
    border-left: 5px solid #00c6ff;
    text-shadow: 1px 1px 6px black;
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)


# LOAD MODEL + ENCODER
pipeline = joblib.load("crop_pipeline.pkl")
le = joblib.load("label_encoder.pkl")

# SIDEBAR
st.sidebar.title("📌  Navigation Menu")
page = st.sidebar.radio("Select Page", ["Home", "About Project", "Dataset Info", "Model Used"])

# ABOUT PAGES
if page == "About Project":
    st.markdown("<h1 class='main-title'>📘 About The Project</h1>", unsafe_allow_html=True)
    st.write("""
    This ML system predicts the best crop using environmental conditions:
    - Nitrogen  
    - Phosphorus  
    - Potassium  
    - Temperature  
    - Humidity  
    - pH Value  
    - Rainfall  
    """)
    st.stop()

if page == "Dataset Info":
    st.markdown("<h1 class='main-title'>📊 Dataset Information</h1>", unsafe_allow_html=True)
    st.write("""
    - 22 crop classes  
    - 7 numerical features  
    - Clean and balanced  
    - Used for ML training  
    """)
    st.stop()

if page == "Model Used":
    st.markdown("<h1 class='main-title'>🤖 Machine Learning Model</h1>", unsafe_allow_html=True)
    st.write("""
    - Multiple ML models tested  
    - Final model selected based on best performance  
    - Pipeline: StandardScaler + Best Model  
    """)
    st.stop()


# ---------------------- HOME PAGE ----------------------
st.markdown("<h1 class='main-title'>🌾 Crop Recommendation System</h1>", unsafe_allow_html=True)

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

# ---------------------- PREDICTION ----------------------
if predict_btn:
    sample = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = pipeline.predict(sample)
    crop = le.inverse_transform(prediction)[0]

    st.markdown(
        f"<div class='result-box'><h2>🌟 Recommended Crop: {crop.upper()}</h2></div>",
        unsafe_allow_html=True
    )

    df_compare = pd.DataFrame({
        "Parameter": ["N", "P", "K", "Temperature", "Humidity", "pH", "Rainfall"],
        "Your Input": [N, P, K, temperature, humidity, ph, rainfall]
    })

    st.bar_chart(df_compare.set_index("Parameter"))
