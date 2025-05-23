from joblib import load
import pandas as pd

model_rest=load("app/artifacts/model_rest.joblib")
model_young=load("app/artifacts/model_rest.joblib")
scaler_rest=load("app/artifacts/scaler_rest.joblib")
scaler_young=load("app/artifacts/scaler_young.joblib")

def calculate_normalized_risk(medical_history):
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }

    # Split the medical history into potential two parts and convert to lowercase
    diseases = medical_history.lower().split(" & ")

    # Calculate the total risk score by summing the risk scores for each part
    medical_history_score = sum(risk_scores.get(disease, 0) for disease in diseases)  # Default to 0 if disease not found

    #combined_total_risk += medical_history_score 

    max_score = 14 # risk score for heart disease (8) + second max risk score (6) for diabetes or high blood pressure
    min_score = 0  # Since the minimum score is always 0

    # Normalize the total risk score
    normalized_risk_score = (medical_history_score - min_score) / (max_score - min_score)

    return normalized_risk_score


def calculate_normalized_ls_risk(physical_activities,stress_levels):
    physical_activity_risk={
        "high": 0,
        "medium": 1,
        "low":4
    }
    stress_level_risk={
        "high": 4,
        "medium": 1,
        "low":0
    }

    physical_activities=physical_activities.lower().split(" & ")
    stress_levels=stress_levels.lower().split(" & ")

    physical_activity_score = sum(physical_activity_risk.get(physical_activity, 0) for physical_activity in physical_activities)
    stress_score = sum(stress_level_risk.get(stress_level, 0) for stress_level in stress_levels)

    lf_risk=physical_activity_score + stress_score

    max_score = 8
    min_score = 0

    normalized_lf_risk_score = (lf_risk - min_score) / (max_score - min_score)

    return normalized_lf_risk_score


def handle_scaling(age, df):
    # scale age and income_lakhs column
    if age <= 25:
        scaler_object = scaler_young
    else:
        scaler_object = scaler_rest

    cols_to_scale = scaler_object['cols_to_scale']
    scaler = scaler_object['scaler']

    df['income_level'] = None # since scaler object expects income_level supply it. This will have no impact on anything
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    df.drop('income_level', axis='columns', inplace=True)

    return df


def preprocess_input(input_dict):
    expected_columns = [
        "age",
        "income_lakhs",
        "insurance_plan",
        "normalized_ls_risk_score",
        "normalized_risk_score",
        "gender_Male",
        "region_northwest",           # MUST BE LOWERCASE
        "region_southeast",           # MUST BE LOWERCASE
        "region_southwest",           # MUST BE LOWERCASE
        "bmi_category_obesity",       # MUST BE LOWERCASE
        "bmi_category_overweight",    # MUST BE LOWERCASE
        "bmi_category_underweight",   # MUST BE LOWERCASE
        "smoking_status_occasional",  # MUST BE LOWERCASE
        "smoking_status_regular",     # MUST BE LOWERCASE
        "employment_status_salaried", # MUST BE LOWERCASE
        "employment_status_self_employed" # MUST BE LOWERCASE
    ]

    insurance_plan_encoding = {"Bronze":1,"Silver":2,"Gold":3}
    physical_activity_risk={"High": 0,"Medium": 1,"Low":4}
    stress_level_risk={"High": 4,"Medium": 1,"Low":0}
    df=pd.DataFrame(0,columns=expected_columns,index=[0])
    bmi= input_dict['BMI Category']

    # Populate the DataFrame based on input_dict
    for key, value in input_dict.items():
        if key == 'Gender' and value == 'Male':
            df['gender_Male'] = 1
        elif key == 'Region':
            # Assign to lowercase columns as per model's expectation
            if value == 'Northwest':
                df['region_northwest'] = 1
            elif value == 'Southeast':
                df['region_southeast'] = 1
            elif value == 'Southwest':
                df['region_southwest'] = 1
        elif key == 'BMI Category':
            # Assign to lowercase columns
            if value == 'Obesity':
                df['bmi_category_obesity'] = 1
            elif value == 'Overweight':
                df['bmi_category_overweight'] = 1
            elif value == 'Underweight':
                df['bmi_category_underweight'] = 1
        elif key == 'Smoking Status':
            # Assign to lowercase columns
            if value == 'Occasional':
                df['smoking_status_occasional'] = 1
            elif value == 'Regular':
                df['smoking_status_regular'] = 1
        elif key == 'Employment Status':
            # Assign to lowercase columns
            if value == 'Salaried':
                df['employment_status_salaried'] = 1
            elif value == 'Self-Employed':
                df['employment_status_self_employed'] = 1
        elif key == 'Insurance Plan':
            df['insurance_plan'] = insurance_plan_encoding.get(value, 1)
        elif key == 'Age':
            df['age'] = value
        elif key == 'Income in Lakhs':
            df['income_lakhs'] = value
        elif key == 'Physical Activity':
            df['physical_activity'] = physical_activity_risk.get(value,1)
        elif key == 'Stress Level':
            df['stress_level'] = stress_level_risk.get(value,1)
        

    # Calculate 'normalized_risk_score' AFTER assigning 'genetical_risk'
    df['normalized_risk_score'] = calculate_normalized_risk(input_dict['Medical History'])
    df['normalized_ls_risk_score'] = calculate_normalized_ls_risk(input_dict['Physical Activity'],input_dict['Stress Level'])
    df.drop(columns=['physical_activity','stress_level'], inplace=True)
    df = handle_scaling(input_dict['Age'], df)

    return df



def predict(input_dict):
    print(input_dict)
    input_df = preprocess_input(input_dict)
    if input_dict['Age'] <= 25:
        prediction = model_young.predict(input_df)
    else:
        prediction = model_rest.predict(input_df)

    return int(prediction[0])