import streamlit as st
import joblib
import numpy as np
import pandas as pd
import base64

# ---------------------------
# PAGE SETTINGS
# ---------------------------
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="centered"
)

# ---------------------------
# BACKGROUND IMAGE
# ---------------------------
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

# Use your own background image (upload file to GitHub)
add_bg("background.jpg")   # <-- add your image here


# ---------------------------
# LOAD MODEL & ENCODER
# ---------------------------
pipeline = joblib.load("crop_pipeline.pkl")
le = joblib.load("label_encoder.pkl")

# ---------------------------
# CROP IMAGES (ADD YOUR OWN IMAGES)
# ---------------------------
crop_images = {
    "rice": "images/rice.jpg",
    "maize": "images/maize.jpg",
    "mango": "images/mango.jpg",
    "banana": "images/banana.jpg",
    "coconut": "images/coconut.jpg",
    "apple": "images/apple.jpg"
}

# ---------------------------
# CROP INFORMATION
# ---------------------------
crop_info = {
    "rice": "Ideal Temperature: 20–35°C\nRainfall: 150–300mm\nSoil: Clay Loam\nNeeds standing water.",
    "maize": "Ideal Temperature: 18–27°C\nRainfall: 50–100mm\nSoil: Loamy Soil\nHighly adaptable crop.",
    "apple": "Temperature: 8–15°C\nRainfall: 100–200mm\nSoil: Well-Drained Loam\nNeeds cool climate.",
    "mango": "Temperature: 24–30°C\nRainfall: 80–120mm\nSoil: Red/Loamy Soil\nRequires warm climate.",
    "banana": "Temperature: 20–30°C\nRainfall: 100–200mm\nSoil: Rich Loam\nNeeds high humidity.",
    "coconut": "Temperature: 22–30°C\nRainfall: 100–300mm\nSoil: Sandy/Loam\nBest near coastal regions."
}

# ---------------------------
# CUSTOM CSS
# ---------------------------
clean_css = """
<style>
h1 {
    text-align: center;
    color: #0057ff;
    font-weight: 900;
    margin-bottom: -10px;
}
.input-card {
    background: rgba(255,255,255,0.80);
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.25);
    margin-bottom: 20px;
    backdrop-filter: blur(5px);
}
.result-box {
    background: rgba(0, 175, 255, 0.2);
    padding: 15px;
    border-left: 6px solid #0057ff;
    border-radius: 10px;
    animation: glow 1.5s infinite alternate;
}
@keyframes glow {
    0% { box-shadow: 0 0 10px #80d4ff; }
    100% { box-shadow: 0 0 25px #1aa3ff; }
}
</style>
"""
st.markdown(clean_css, unsafe_allow_html=True)

# ---------------------------
# SIDEBAR MENU
# ---------------------------
st.sidebar.title("📌 Navigation Menu")
menu = st.sidebar.radio("Select Page", ["Home", "About Project", "Dataset Info", "Model Used"])

if menu == "About Project":
    st.title("📘 About Project")
    st.write("""
    This ML-based Crop Recommendation System uses soil and environmental input data 
    to determine the best crop to cultivate.
    """)
    st.stop()

if menu == "Dataset Info":
    st.title("📊 Dataset Information")
    st.write("""
    - 22 crops
    - 7 environmental parameters
    - Cleaned and processed dataset
    """)
    st.stop()

if menu == "Model Used":
    st.title("🤖 Machine Learning Model")
    st.write("""
    - 16+ ML models tested  
    - Best performing model selected  
    - Pipeline with StandardScaler + Best Model  
    """)
    st.stop()

# ---------------------------
# MAIN TITLE
# ---------------------------
st.markdown("<h1>🌾 Crop Recommendation System</h1>", unsafe_allow_html=True)

# ---------------------------
# INPUT CARD
# ---------------------------
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

# ---------------------------
# PREDICTION SECTION
# ---------------------------
if predict_btn:
    sample = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    pred = pipeline.predict(sample)
    crop = le.inverse_transform(pred)[0]

    # ANIMATED RESULT BOX
    st.markdown(
        f"<div class='result-box'><h2>Recommended Crop: {crop.upper()}</h2></div>",
        unsafe_allow_html=True
    )

    # SHOW CROP IMAGE (if available)
    if crop.lower() in crop_images:
        st.image(crop_images[crop.lower()], width=300)

    # CROP INFORMATION
    st.subheader("📘 Crop Requirements")
    st.write(crop_info.get(crop.lower(), "Information not available."))

    # COMPARISON CHART
    st.subheader("📊 Comparison Chart")
    df_compare = pd.DataFrame({
        "Parameter": ["N", "P", "K", "Temp", "Humid", "pH", "Rainfall"],
        "Your Input": [N, P, K, temperature, humidity, ph, rainfall]
    })
    st.bar_chart(df_compare.set_index("Parameter"))

