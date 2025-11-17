import streamlit as st
import joblib
import numpy as np
import pandas as pd

# PAGE SETTINGS
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide"
)

# ---------------------- CLEAN WHITE-BLUE UI ----------------------
clean_css = """
<style>

html, body, [class*="css"] {
    color: #000000 !important;
    font-family: 'Segoe UI', sans-serif;
}

/* Top Navigation Bar */
.nav-bar {
    width: 100%;
    background: #ffffff;
    border-bottom: 2px solid #e6e6e6;
    padding: 15px;
    display: flex;
    justify-content: center;
    gap: 40px;
    position: sticky;
    top: 0;
    z-index: 100;
}

.nav-item {
    color: #0057ff;
    font-size: 20px;
    font-weight: 700;
    cursor: pointer;
    padding: 6px 15px;
    border-radius: 6px;
    transition: 0.3s;
}

.nav-item:hover {
    background: #e8f0ff;
}

.active {
    background: #d9e7ff;
    border-bottom: 3px solid #0057ff;
}

/* Page Title */
.title-box {
    width: 100%;
    background: #f1f6ff;
    padding: 25px;
    text-align: center;
    border-radius: 12px;
    margin-top: 12px;
}

.title-text {
    font-size: 45px;
    font-weight: 900;
    color: #003a96;
}

/* Input Card */
.input-card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #e6e6e6;
    box-shadow: 0px 4px 8px rgba(0,0,0,0.06);
    margin-top: 20px;
}

/* Predict Button */
.stButton > button {
    width: 100%;
    background: #0057ff;
    color: white;
    font-weight: 700;
    padding: 12px;
    border-radius: 8px;
    border: none;
    font-size: 18px;
}
.stButton > button:hover {
    background: #003bb5;
}

/* Result Box */
.result-box {
    background: #e8f0ff;
    padding: 18px;
    border-left: 6px solid #0057ff;
    border-radius: 10px;
    margin-top: 15px;
}

</style>
"""
st.markdown(clean_css, unsafe_allow_html=True)


# ---------------------- TOP NAVIGATION ----------------------
selected_page = st.session_state.get("selected_page", "Home")

# Build nav bar
nav_cols = st.columns([1,1,1,1,1])
with nav_cols[0]:
    if st.button("🏠 Home"):
        selected_page = "Home"
with nav_cols[1]:
    if st.button("📘 About Project"):
        selected_page = "About"
with nav_cols[2]:
    if st.button("📊 Dataset Info"):
        selected_page = "Dataset"
with nav_cols[3]:
    if st.button("🤖 Model Used"):
        selected_page = "Model"

st.session_state["selected_page"] = selected_page


# ---------------------- TITLE ----------------------
st.markdown("""
    <div class="title-box">
        <h1 class="title-text">🌾 Crop Recommendation System</h1>
    </div>
""", unsafe_allow_html=True)


# ---------------------- LOAD ML MODEL ----------------------
pipeline = joblib.load("crop_pipeline.pkl")
le = joblib.load("label_encoder.pkl")


# ---------------------- PAGE CONTENT ----------------------

# ABOUT PROJECT
if selected_page == "About":
    st.subheader("📘 About The Project")
    st.write("""
    This ML system predicts the best crop using:
    - Nitrogen  
    - Phosphorus  
    - Potassium  
    - Temperature  
    - Humidity  
    - pH  
    - Rainfall  
    """)
    st.stop()

# DATASET PAGE
if selected_page == "Dataset":
    st.subheader("📊 Dataset Information")
    st.write("""
    - 22 crops  
    - 7 numerical features  
    - Clean and balanced dataset  
    """)
    st.stop()

# MODEL PAGE
if selected_page == "Model":
    st.subheader("🤖 Machine Learning Model")
    st.write("""
    - Multiple ML models tested  
    - Best performing one selected  
    - Pipeline with StandardScaler  
    """)
    st.stop()


# ---------------------- HOME PAGE ----------------------
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


# ---------------------- PREDICTION OUTPUT ----------------------
if predict_btn:
    sample = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    pred = pipeline.predict(sample)
    crop = le.inverse_transform(pred)[0]

    st.markdown(
        f"<div class='result-box'><h2>🌟 Recommended Crop: {crop.upper()}</h2></div>",
        unsafe_allow_html=True
    )

    df_compare = pd.DataFrame({
        "Parameter": ["N", "P", "K", "Temperature", "Humidity", "pH", "Rainfall"],
        "Your Input": [N, P, K, temperature, humidity, ph, rainfall]
    })

    st.bar_chart(df_compare.set_index("Parameter"))
