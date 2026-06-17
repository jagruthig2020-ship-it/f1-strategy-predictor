import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="F1 Anticipatory Strategy", page_icon="🔮", layout="centered")

st.title("🔮 F1 Anticipatory Strategy Room")
st.write("Input real-time telemetry to predict if a driver will head to pits **on the upcoming lap**.")

@st.cache_resource
def load_model():
    with open("f1_pit_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.markdown("---")
st.subheader("📋 Current Telemetry Matrix")

col1, col2 = st.columns(2)

with col1:
    lap = st.number_input("Current Lap Number", min_value=1, max_value=80, value=20)
    position = st.slider("Current Track Position", min_value=1, max_value=20, value=4)
    tyre_age = st.number_input("Current Tyre Age (Laps on current stint)", min_value=1, max_value=60, value=15)

with col2:
    lap_time_secs = st.number_input("Last Lap Duration (Seconds)", min_value=50.0, max_value=120.0, value=78.5)
    lap_time_delta = st.slider("Pace Deviation from Average (Seconds)", min_value=-5.0, max_value=10.0, value=0.8, step=0.1)

st.subheader("📜 Historic & Track Profile Baselines")
col3, col4 = st.columns(2)

with col3:
    track_avg_pit_lap = st.slider("Track Historical Average Pit Window (Lap)", min_value=5, max_value=50, value=22)
with col4:
    driver_avg_stint_length = st.slider("Driver's Historic Average Tyre Lifespan (Laps)", min_value=5, max_value=40, value=18)

st.markdown("---")

if st.button("🔮 Calculate Predictive Strategy", use_container_width=True):
    # Construct data matching our strict model training features matrix order
    input_data = pd.DataFrame([{
        'lap': lap,
        'position': position,
        'tyre_age': tyre_age,
        'lap_time_secs': lap_time_secs,
        'lap_time_delta': lap_time_delta,
        'track_avg_pit_lap': track_avg_pit_lap,
        'driver_avg_stint_length': driver_avg_stint_length
    }])
    
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    pit_probability = probabilities[1] * 100

    st.subheader("📊 Strategic Window Assessment:")
    
    if prediction == 1:
        st.error(f"🚨 **BOX NEXT LAP!** Prepare pit crew. Prediction for UPCOMING lap is a PIT STOP (Probability: {pit_probability:.1f}%)")
        st.markdown("💡 *Strategy insight:* Telemetry and historical baselines suggest the pit sequence must be triggered immediately to secure track position.")
    else:
        st.success(f"🟢 **STAY OUT NEXT LAP.** (Stay Out Certainty: {probabilities[0]*100:.1f}%)")
        st.markdown("💡 *Strategy insight:* Target driver is performing within stable performance bounds. Extend stint.")
