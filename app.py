import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Set up page styling
st.set_page_config(page_title="F1 Strategy Predictor", page_icon="🏎️", layout="centered")

st.title("🏎️ F1 Pit Stop Strategy Room")
st.write("Input current race telemetry variables to predict if a driver will head to the pit lane this lap.")

# 1. Load the trained XGBoost model safely
@st.cache_resource
def load_model():
    with open("f1_pit_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except FileNotFoundError:
    st.error("❌ 'f1_pit_model.pkl' not found. Make sure it's in the same folder as this app.py file!")

st.markdown("---")
st.subheader("📋 Current Lap Telemetry")

# 2. Setup user input parameters matching our training features exactly:
# Features expected by model: ['lap', 'position', 'tyre_age', 'lap_time_secs', 'lap_time_delta']

col1, col2 = st.columns(2)

with col1:
    lap = st.number_input("Current Lap Number", min_value=1, max_value=80, value=15)
    position = st.slider("Current Track Position", min_value=1, max_value=20, value=5)
    tyre_age = st.number_input("Current Tyre Age (Laps on this stint)", min_value=1, max_value=60, value=12)

with col2:
    lap_time_secs = st.number_input("Last Lap Time (Seconds)", min_value=50.0, max_value=120.0, value=85.4, step=0.1)
    # A positive delta means they are dropping off their average pace
    lap_time_delta = st.slider("Lap Time Delta to Personal Average (Seconds)", min_value=-5.0, max_value=10.0, value=1.2, step=0.1)

st.markdown("---")

# 3. Handle Prediction Logic when button is clicked
if st.button("🔮 Run Strategy Prediction", use_container_width=True):
    # Construct a DataFrame matching the precise column order used during model training
    input_data = pd.DataFrame([{
        'lap': lap,
        'position': position,
        'tyre_age': tyre_age,
        'lap_time_secs': lap_time_secs,
        'lap_time_delta': lap_time_delta
    }])
    
    # Get raw prediction and prediction probabilities
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    pit_probability = probabilities[1] * 100

    # Display clean results interface
    st.subheader("📊 Strategic Assessment:")
    
    if prediction == 1:
        st.error(f"🚨 **BOX BOX BOX!** The model predicts a **PIT STOP** sequence (Confidence: {pit_probability:.1f}%)")
        st.markdown("💡 *Reasoning:* The current pace degradation combined with tire wear indicates the driver is hitting the performance cliff or entering the pit window.")
    else:
        st.success(f"🟢 **STAY OUT.** The model predicts the driver will **LAP AGAIN** (Stay Out Confidence: {probabilities[0]*100:.1f}%)")
        st.markdown("💡 *Reasoning:* Telemetry suggests the current pace is stable enough to extend the stint.")