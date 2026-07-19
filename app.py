import streamlit as st
import numpy as np
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Disease Prediction",
    page_icon="🩺",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
with open("disease_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🩺 Disease Prediction")
st.sidebar.markdown("---")
st.sidebar.info(
    """
This application predicts whether a patient is likely to have a disease based on health parameters.

Developed using:
- Python
- Scikit-Learn
- Streamlit
"""
)

# ---------------- TITLE ----------------
st.title("🩺 AI Disease Prediction System")
st.markdown("### Enter Patient Information")

st.markdown("---")

# ---------------- INPUTS ----------------

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 25)

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=22.5,
        format="%.1f"
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        50,
        250,
        120
    )

    glucose = st.number_input(
        "Glucose Level",
        50,
        500,
        100
    )

    cholesterol = st.number_input(
        "Cholesterol",
        50,
        400,
        180
    )

with col2:

    heart_rate = st.number_input(
        "Heart Rate",
        30,
        220,
        72
    )

    smoking = st.selectbox(
        "Smoking",
        ["No", "Yes"]
    )

    alcohol = st.selectbox(
        "Alcohol",
        ["No", "Yes"]
    )

    physical_activity = st.number_input(
        "Physical Activity (Hours/Week)",
        0,
        40,
        5
    )

    family_history = st.selectbox(
        "Family History",
        ["No", "Yes"]
    )

# Convert Yes/No to 0/1

smoking = 1 if smoking == "Yes" else 0
alcohol = 1 if alcohol == "Yes" else 0
family_history = 1 if family_history == "Yes" else 0

st.markdown("---")

# ---------------- BUTTON ----------------

if st.button("🔍 Predict Disease", use_container_width=True):

    data = np.array([[

        age,
        bmi,
        blood_pressure,
        glucose,
        cholesterol,
        heart_rate,
        smoking,
        alcohol,
        physical_activity,
        family_history

    ]])

    data = scaler.transform(data)

    prediction = model.predict(data)

    probability = model.predict_proba(data)

    confidence = np.max(probability) * 100

    st.progress(int(confidence))

    st.write(f"### Prediction Confidence: **{confidence:.2f}%**")

    st.markdown("---")

    if prediction[0] == 0:

        st.success("## ✅ Healthy")

        st.info("""
No disease is predicted based on the entered values.

### Health Tips

✔ Eat healthy food

✔ Exercise regularly

✔ Drink enough water

✔ Sleep 7–8 hours daily

✔ Regular health checkups
""")

    else:

        st.error("## ⚠ Disease Detected")

        st.warning("""
The model predicts a possibility of disease.

### Recommendations

✔ Consult a doctor

✔ Get medical tests

✔ Maintain healthy lifestyle

✔ Follow medical advice
""")

st.markdown("---")

st.caption()