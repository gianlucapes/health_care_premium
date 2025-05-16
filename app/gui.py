import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

from app.front_end.prediction_helper import predict

def main_gui():
    # Sample Data (Replace with your actual trained model and data)
    data = {
        'age': np.random.randint(18, 65, 100),
        'gender': np.random.choice(['Male', 'Female'], 100),
        'bmi': np.random.uniform(18, 40, 100),
        'children': np.random.randint(0, 5, 100),
        'smoker': np.random.choice(['No', 'Yes'], 100),
        'region': np.random.choice(['Northwest', 'Southeast', 'Northeast', 'Southwest'], 100),
        'charges': np.random.uniform(1000, 50000, 100)
    }
    df = pd.DataFrame(data)

    # Preprocessing for the sample model
    le = LabelEncoder()
    df['gender_encoded'] = le.fit_transform(df['gender'])
    df['smoker_encoded'] = le.fit_transform(df['smoker'])
    df['region_encoded'] = le.fit_transform(df['region'])

    X = df[['age', 'gender_encoded', 'bmi', 'children', 'smoker_encoded', 'region_encoded']]
    y = df['charges']

    model = LinearRegression()
    model.fit(X, y)

    # Streamlit Frontend
    st.title("Health Insurance Cost Prediction")

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            gender = st.selectbox("Gender", ['Male', 'Female'])
            bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
            children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
            bmi_category = st.selectbox("BMI Category", ['Normal', 'Obesity', 'Overweight', 'Underweight'])
            marital_status = st.selectbox("Marital Status", ['Unmarried', 'Married'])
        with col2:
            region = st.selectbox("Region", ['Northwest', 'Southeast', 'Northeast', 'Southwest'])
            income_level = st.selectbox("Income Level", ['<10L', '10L - 25L', '> 40L', '25L - 40L'])
            insurance_plan = st.selectbox("Insurance Plan", ['Bronze', 'Silver', 'Gold'])
            medical_history = st.selectbox("Medical History", ['Diabetes', 'High blood pressure', 'No Disease',
                                                            'Diabetes & High blood pressure', 'Thyroid', 'Heart disease',
                                                            'High blood pressure & Heart disease', 'Diabetes & Thyroid',
                                                            'Diabetes & Heart disease'])
            employment_status = st.selectbox("Employment Status", ['Salaried', 'Self-Employed', 'Freelancer'])
            smoking_status = st.selectbox("Smoking Status", ['No Smoking', 'Regular', 'Occasional', 'Does Not Smoke', 'Not Smoking',
                                                            'Smoking=0'])

    if st.button("Predict"):
        # Prepare input data for the model
        input_data = pd.DataFrame({
            'age': [age],
            'gender': [gender],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoking_status in ['Regular', 'Occasional', 'Smoking=0']],
            'region': [region],
            # Add other features as needed based on your model
        })

        predict(input_data)