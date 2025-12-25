import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from predictor import predict_diabetes

st.set_page_config(page_title="Disease Risk Predictor", layout="centered")
st.title("🩺 Disease Risk Predictor")

# Sidebar for disease selection
st.sidebar.header("Select Disease")
disease = st.sidebar.selectbox(
    "Choose a condition to assess:",
    ["Home", "Diabetes", "Heart Disease (coming soon)", "Hypertension (coming soon)"]
)

# Feature engineering function
def engineer_features(df):
    df['Age_squared'] = df['Age'] ** 2
    df['BMI_Age'] = df['BMI'] * df['Age']
    df['Glucose_Insulin'] = df['Glucose'] / (df['Insulin'] + 1)
    df['Glucose_per_BMI'] = df['Glucose'] / (df['BMI'] + 1)
    return df

# Initialize session state for prediction result
if "diabetes_result" not in st.session_state:
    st.session_state.diabetes_result = None

# Home
if disease == "Home":
    st.subheader("Dashboard Overview")

    # Display last prediction result and patient data
    result = st.session_state.get("diabetes_result", None)
    if result and isinstance(result, dict) and "label" in result:
        st.info(f"Last Predicted Outcome: {result['label']} ({result['confidence']}% confidence)")

        patient_df = pd.DataFrame([result["input"]])
        st.write("Patient Data Used for Prediction")
        st.dataframe(patient_df)

        # Visualization 1: Glucose vs BMI
        st.write("🔬 Glucose vs BMI")
        fig1, ax1 = plt.subplots()
        ax1.scatter(patient_df["BMI"], patient_df["Glucose"], color="blue", label=result["label"])
        ax1.set_xlabel("BMI")
        ax1.set_ylabel("Glucose Level")
        ax1.set_title("Patient Glucose vs BMI")
        ax1.legend()
        st.pyplot(fig1)

        # Visualization 2: Age vs Glucose
        st.write("📈 Age vs Glucose")
        fig2, ax2 = plt.subplots()
        ax2.bar(patient_df["Age"], patient_df["Glucose"], color="purple")
        ax2.set_xlabel("Age")
        ax2.set_ylabel("Glucose Level")
        ax2.set_title("Patient Age vs Glucose")
        st.pyplot(fig2)

        # Visualization 3: Age vs BMI
        st.write("📊 Age vs BMI")
        fig3, ax3 = plt.subplots()
        ax3.bar(patient_df["Age"], patient_df["BMI"], color="orange")
        ax3.set_xlabel("Age")
        ax3.set_ylabel("BMI")
        ax3.set_title("Patient Age vs BMI")
        st.pyplot(fig3)
    else:
        st.warning("No prediction available yet. Please run a prediction from the selected tab.")

# Diabetes prediction form
elif disease == "Diabetes":
    st.subheader("🧪 Enter Patient Data for Diabetes Prediction")

    age = st.slider("Age", 18, 100)
    glucose = st.number_input("Glucose Level", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0)
    insulin = st.number_input("Insulin", min_value=0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0)
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    pregnancies = st.number_input("Pregnancies", min_value=0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0)

    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    if st.button("Predict"):
        input_df = pd.DataFrame([[age, glucose, bmi, insulin, blood_pressure, diabetes_pedigree,
                                  pregnancies, skin_thickness]],
                                columns=['Age', 'Glucose', 'BMI', 'Insulin', 'BloodPressure',
                                         'DiabetesPedigreeFunction',
                                         'Pregnancies', 'SkinThickness'])

        input_df = engineer_features(input_df)
        result = predict_diabetes(input_df)

        if isinstance(result, dict) and "label" in result:
            st.session_state.diabetes_result = result
            st.session_state.prediction_history.append(result)
            st.success(f"Predicted Outcome: {result['label']} ({result['confidence']}% confidence)")
        else:
            st.error(result.get("error", "Prediction failed."))

# Placeholder for future modules
else:
    st.info("🚧 This module is under development. Stay tuned!")