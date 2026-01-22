import streamlit as st
import joblib
import numpy as np

model = joblib.load("traffic_signal_model.pkl")

st.title("🚦 Traffic Signal Failure Detection System")

traffic_density = st.selectbox("Traffic Density", ["Low", "Medium", "High"])
avg_wait_time = st.number_input("Average Waiting Time (seconds)", 0, 500)
power_status = st.selectbox("Power Status", ["Normal", "Fluctuating", "Off"])
sensor_status = st.selectbox("Sensor Status", ["Working", "Faulty"])
weather = st.selectbox("Weather Condition", ["Clear", "Rainy", "Foggy"])
time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])

mapping = {
    "Low": 0, "Medium": 1, "High": 2,
    "Normal": 2, "Fluctuating": 1, "Off": 0,
    "Working": 1, "Faulty": 0,
    "Clear": 0, "Rainy": 1, "Foggy": 2,
    "Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3
}

input_data = np.array([[
    mapping[traffic_density],
    avg_wait_time,
    mapping[power_status],
    mapping[sensor_status],
    mapping[weather],
    mapping[time_of_day]
]])

if st.button("Predict Signal Status"):
    result = model.predict(input_data)
    if result[0] == 1:
        st.error("⚠️ Traffic Signal FAILURE Detected")
    else:
        st.success("✅ Traffic Signal is Working Normally")

