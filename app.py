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


# ---------------------- PREMIUM CSS ----------------------
css = """
<style>

/* White Text Everywhere */
html, body, [class*="css"] {
    color: white !important;
}

/* TOP NAVIGATION BAR */
.navbar {
    width: 100%;
    background: rgba(0,0,0,0.55);
    padding: 15px;
    display: flex;
    justify-content: center;
    gap: 50px;
    border-radius: 0 0 12px 12px;
    backdrop-filter: blur(8px);
}

.nav-item {
    color: white;
    font-size: 20px;
    font-weight: 700;
    cursor: pointer;
    padding: 8px 15px;
    border-radius: 6px;
    transition: 0.3s;
}

.nav-item:hover {
    background: rgba(255,255,255,0.22);
}

/* Active Page */
.active {
    background: rgba(0,166,255,0.35);
    border-bottom: 3px solid #00c6ff;
}

/* Title Section */
.title-box {
    width: 100%;
    background: rgba(0,0,0,0.50);
    padding: 25px;
    margin-top: 10px;
    text-align: center;
    border-radius: 12px;
    backdrop-filter: blur(5px);
}

.title-text {
    font-size: 50px;
    font-weight: 900;
    color: white;
    text-shadow: 3px 3px 12px black;
}

/* Input Card */
.input-card {
    background: rgba(0,0,0,0.55);
    padding: 25px;
    border-radius: 15px;
    backdrop-filter: blur(12px);
    box-shadow: 0px 4px 12px rgba(0,0,0,0.7);
    margin-top: 20px;
}

/* Predict Button */
.stButton > button {
    width: 100%;
    background: #00c6ff;
    color: black;
    font-weight: 700;
    padding: 12px;
    border-radius: 8px;
    border: none;
    font-size: 18px;
}
.stButton > button:hover {
    background: #0099cc;
}

/* Result Box */
.result-box {
    background: rgba(0,0,0,0.6);
    padding: 18px;
    border-left: 6px solid #00c6ff;
    border-radius: 10px;
    margin-top: 15px;
}

</style>
"""
st.markdown(css, unsafe_allow_html=True)


# TOP NAVIGATION BAR (CUSTOM)
selected_page = st.session_state.get("selected_page", "Home")

cols = st.columns([1,1,1,1,1])
with cols[0]:
    if st.button("🏠 Home", key="home_btn"):
        selected_page = "Home"
with cols[1]:
    if st.button("📘 About Project", key="about_btn"):
        selected_page = "About"
with cols[2]:
    if st.button("📊 Dataset Info", key="data_btn"):
        selected_page = "Dataset"
with cols[3]:
    if st.button("🤖 Model Used", key="model_btn"):
        selected_page = "Model"

st.session_state["selected_page"] = selected_page

# TITLE
st.markdown("""
    <div class="title-box">
        <h1 class="title-text">🌾 Crop Recommendation System</h1>
    </div>
""", unsafe_allow_html=True)


# LOAD MODEL
pipeline = joblib.load("crop_pipeline.pkl")
le = joblib.load("label_encoder.pkl")


# ---------------------- PAGE CONTENT ----------------------

# ABOUT PROJECT
if selected_page == "About":
    st.subheader("📘 About The Project")
    st.write("""
    This system predicts the best crop using environmental parameters.
    It uses ML algorithms to generate crop recommendation.
    """)
    st.stop()

# DATASET
if selected_page == "Dataset":
    st.subheader("📊 Dataset Information")
    st.write("""
    - 22 crops  
    - 7 environmental features  
    - Clean and balanced dataset  
    """)
    st.stop()

# MODEL USED
if selected_page == "Model":
    st.subheader("🤖 Machine Learning Model")
    st.write("""
    - Multiple ML models tested  
    - Best-performing one selected  
    - Pipeline includes StandardScaler  
    """)
    st.stop()


# HOME PAGE (MAIN)
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

# PREDICTION
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
