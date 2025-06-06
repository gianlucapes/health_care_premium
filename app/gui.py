import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

from app.front_end.prediction_helper import predict

def main_gui():
    # codebasics ML course: codebasics.io, all rights reserverd
    # Define the page layout
    st.title('Health Insurance Cost Predictor')

    categorical_options = {
        'Gender': ['Male', 'Female'],
        'BMI Category': ['Normal', 'Obesity', 'Overweight', 'Underweight'],
        'Smoking Status': ['No Smoking', 'Regular', 'Occasional'],
        'Employment Status': ['Salaried', 'Self-Employed', 'Freelancer', ''],
        'Region': ['Northwest', 'Southeast', 'Northeast', 'Southwest'],
        'Medical History': [
            'No Disease', 'Diabetes', 'High blood pressure', 'Diabetes & High blood pressure',
            'Thyroid', 'Heart disease', 'High blood pressure & Heart disease', 'Diabetes & Thyroid',
            'Diabetes & Heart disease'
        ],
        'Insurance Plan': ['Bronze', 'Silver', 'Gold'],
        'Stress Level' : ['Medium', 'High', 'Low'],
        'Physical Activity' : ['Medium', 'Low' , 'High']
    }

    # Create four rows of three columns each
    row1 = st.columns(2)
    row2 = st.columns(3)
    row3 = st.columns(3)
    row4 = st.columns(3)

    # Assign inputs to the grid
    with row1[0]:
        age = st.number_input('Age', min_value=18, step=1, max_value=100)
    with row1[1]:
        income_lakhs = st.number_input('Income in Lakhs', step=1, min_value=0, max_value=200)

    with row2[0]:
        insurance_plan = st.selectbox('Insurance Plan', categorical_options['Insurance Plan'])
    with row2[1]:
        employment_status = st.selectbox('Employment Status', categorical_options['Employment Status'])
    with row2[2]:
        gender = st.selectbox('Gender', categorical_options['Gender'])

    with row3[0]:
        bmi_category = st.selectbox('BMI Category', categorical_options['BMI Category'])

    with row3[1]:
        smoking_status = st.selectbox('Smoking Status', categorical_options['Smoking Status'])
    with row3[2]:
        region = st.selectbox('Region', categorical_options['Region'])

    with row4[0]:
        medical_history = st.selectbox('Medical History', categorical_options['Medical History'])
    with row4[1]:
        stress_level = st.selectbox('Stress Level', categorical_options['Stress Level'])
    with row4[2]:
        physical_activity = st.selectbox('Physical Activity', categorical_options['Physical Activity'])

    # Create a dictionary for input values
    input_dict = {
        'Age': age,
        'Income in Lakhs': income_lakhs,
        'Insurance Plan': insurance_plan,
        'Employment Status': employment_status,
        'Gender': gender,
        'BMI Category': bmi_category,
        'Smoking Status': smoking_status,
        'Region': region,
        'Medical History': medical_history,
        "Stress Level" : stress_level,
        "Physical Activity" : physical_activity
    }

    # Button to make prediction
    if st.button('Predict'):
        prediction = predict(input_dict)
        st.success(f'Predicted Health Insurance Cost: {prediction}')
