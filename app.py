import streamlit as st
import numpy as np
import pickle

# Load Model and Scaler
with open("disease_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="Disease Prediction System", page_icon="🩺")

st.title("🩺 AI Disease Prediction System")
st.write("Enter the patient details below to predict the disease.")

# User Inputs
age = st.number_input("Age", min_value=1, max_value=120, value=25)

bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.5)

blood_pressure = st.number_input("Blood Pressure", min_value=50, max_value=250, value=120)

glucose = st.number_input("Glucose Level", min_value=50, max_value=500, value=100)

cholesterol = st.number_input("Cholesterol", min_value=50, max_value=400, value=180)

heart_rate = st.number_input("Heart Rate", min_value=30, max_value=220, value=72)

smoking = st.selectbox("Smoking", [0, 1])

alcohol = st.selectbox("Alcohol", [0, 1])

physical_activity = st.number_input("Physical Activity (Hours/Week)", min_value=0, max_value=40, value=5)

family_history = st.selectbox("Family History", [0, 1])

if st.button("Predict Disease"):

    data = np.array([[age,
                      bmi,
                      blood_pressure,
                      glucose,
                      cholesterol,
                      heart_rate,
                      smoking,
                      alcohol,
                      physical_activity,
                      family_history]])

    data = scaler.transform(data)

    prediction = model.predict(data)

    st.success(f"Predicted Disease: {prediction[0]}")