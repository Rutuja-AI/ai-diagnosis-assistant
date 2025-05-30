import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Config
st.set_page_config(
    page_title="AI Diagnosis Assistant",
    page_icon="🧠",
    layout="centered"
)

# Session State for test mode
if "sample_mode" not in st.session_state:
    st.session_state.sample_mode = False

# CSS Styling
st.markdown("""
    <style>
        html, body {
            background-color: #121212;
            color: white;
        }
        .main {
            background: linear-gradient(145deg, #1f1f1f, #2c2c2c);
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 0 20px rgba(0,255,200,0.1);
        }
        .stButton>button {
            color: white;
            background-color: #00f5d4;
            border-radius: 8px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='main'>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align:center; color:#00f5d4;'>🏥 AI Diagnosis Assistant</h1>", unsafe_allow_html=True)
st.write("##### Enter your health details below to check your diabetes risk:")

# Use Sample Button
colA, colB = st.columns([1, 1])
with colA:
    if st.button("🔁 Use Sample Test Values"):
        st.session_state.sample_mode = True
with colB:
    if st.button("🔄 Reset Values"):
        st.session_state.sample_mode = False

# Inputs with test values
col1, col2 = st.columns(2)
with col1:
    pregnancies = st.number_input(
        "Pregnancies", 0, 20,
        value=2 if st.session_state.sample_mode else 0,
        help="Number of times the person has been pregnant. Affects insulin regulation."
    )
    glucose = st.number_input(
        "Glucose Level", 0, 200,
        value=135 if st.session_state.sample_mode else 0,
        help="Plasma glucose concentration (mg/dL). High values may indicate pre-diabetes or diabetes."
    )
    bp = st.number_input(
        "Blood Pressure", 0, 140,
        value=70 if st.session_state.sample_mode else 0,
        help="Diastolic blood pressure (mm Hg). High BP is a common diabetes risk factor."
    )
    skin = st.number_input(
        "Skin Thickness", 0, 100,
        value=30 if st.session_state.sample_mode else 0,
        help="Skin fold thickness (mm). Can indicate insulin resistance."
    )

with col2:
    insulin = st.number_input(
        "Insulin Level", 0, 300,
        value=100 if st.session_state.sample_mode else 0,
        help="Serum insulin (μU/ml). Low insulin or resistance may signal diabetes."
    )
    bmi = st.number_input(
        "BMI", 0.0, 70.0,
        value=28.0 if st.session_state.sample_mode else 0.0,
        help="Body Mass Index — weight to height ratio. Obesity increases diabetes risk."
    )
    dpf = st.number_input(
        "Diabetes Pedigree Function", 0.0, 3.0,
        value=0.5 if st.session_state.sample_mode else 0.0,
        help="Probability of diabetes based on family history. Higher = more risk."
    )
    age = st.number_input(
        "Age", 10, 100,
        value=32 if st.session_state.sample_mode else 10,
        help="Age of the person. Diabetes risk increases with age, especially after 45."
    )

# Prediction Button
if st.button("🧠 Predict Risk"):
    # Validation
    if any([
        pregnancies == 0,
        glucose == 0,
        bp == 0,
        skin == 0,
        insulin == 0,
        bmi == 0.0,
        dpf == 0.0,
        age == 10
    ]):
        st.error("⚠️ Please enter all health details (or use the sample values).")
    else:
        with st.spinner("Analyzing your health data..."):
            input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
            proba = model.predict_proba(input_data)[0][1] * 100
            result = model.predict(input_data)[0]

            st.markdown("---")
            if result == 1:
                st.error(f"⚠️ High Risk of Diabetes\n\n🔢 Confidence: {proba:.2f}%")
                st.warning("Please consult a healthcare professional for a detailed evaluation.")
            else:
                st.success(f"✅ You are at low risk!\n\n🔢 Confidence: {100 - proba:.2f}%")
                st.info("Keep living a healthy lifestyle 💪")
                st.balloons()

# Disclaimer
st.markdown("""
<small>
⚠️ *Note: Some values like glucose, insulin, and skin thickness require lab testing.  
If unsure, consult a physician or use the sample test mode to simulate a prediction.*
</small>
""", unsafe_allow_html=True)

# Footer + Sidebar
st.sidebar.markdown("### 🧠 AI Mode: Active")
st.sidebar.success("Model loaded and ready to predict.")
st.markdown("</div>", unsafe_allow_html=True)
