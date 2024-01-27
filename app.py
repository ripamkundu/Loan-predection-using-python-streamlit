import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer

# loan model
loaded_pipeline = joblib.load('pipeline.pkl')

# Streamlit app
def main():
    st.title("**Loan Prediction App**")
    single_data_point = pd.DataFrame()

    single_data_point ["Educations"] = st.selectbox("**Educations**", ["---Select Your Education---","Non_Metric", "Matric", "Inter", "Gradutation", "PG"])
    single_data_point ["Area"] = st.selectbox("**Area**", ["---Select Your Area---",'Urban','Rural'])
    single_data_point["Married_Status"]=st.selectbox("**Married Status**", ["---Select Your Married Status---",'Married','UnMaried'])
    single_data_point["Criminal_Status"]=st.selectbox("**Criminal Status**", ["---Select Your Criminal Status---",'Yes','No'])
    single_data_point["Employed_type"]= st.selectbox("**Employed Type**", ["---Select Your Employement Type---",'Business', 'Salaried', 'Professional', 'Unemployed'])
    single_data_point["Spouse_Education"] =st.selectbox("Spouse Education", ["---Select Your Spouse Education---","Non_Metric", "Matric", "Inter", "Gradutation", "PG"])
    single_data_point["Gender"]= st.selectbox("**Gender**", ["---Select Your Gender---",'Male','Female','Transgender'])
    single_data_point["Age"]=st.selectbox("**Age**", ["---Select Your Age Range---",'Below_20','20-30','30-40','40+'])
    single_data_point["Dependents"]=st.selectbox("**Dependents**", ["---Select Your Dependents---",'0','1','2','3','4'])
    single_data_point["City_Type"]=st.selectbox("**City Type**", ["---Select Your City Type---",'Metro', 'City', 'Town', 'Village'])
    single_data_point["Home"]=st.selectbox("**Home**", ["---Select Your Home---",'Rented','Owned'])
    single_data_point["Vehicle"]= st.selectbox("**Vehicle**", ["---Select Your Vehicle---",'Wheeler_2','Wheeler_4'])
    single_data_point["Income"] = st.selectbox("**Income**", ["---Select Your Incmone---",'0-10000', '10000-25000', '25000-50000', '50000-75000', '75000+'])
    single_data_point["Cibil"] = st.selectbox("**CIBIL**", ["---Select Your Cibil Score---",'0-300','300-500','500-700','700-900'])
    single_data_point["Spouse_Income"]= st.selectbox("**Spouse Income**", ["---Select Your Spouse Income---",'0-10000', '10000-25000', '25000-50000', '50000-75000', '75000+'])
    single_data_point["Pin_code"]= st.selectbox("**Pin Code**", ["---Select Your Pin Code---",'allowed','not_allowed'])
    single_data_point["City_Name"] = st.selectbox("**City Name**", ["---Select Your City Name---",'allowed','not_allowed'])

    user_info = st.text_area("Enter user information (JSON format):")

    if st.button("Predict Loan Status"):
        st.info("**Prediction Result**")
        if user_info:
            try:
                additional_data = json.loads(user_info)
                additional_data_df = pd.DataFrame([additional_data])
                user_data = pd.concat([single_data_point, additional_data_df], ignore_index=True)
                print(user_data)
                prediction = loaded_pipeline.predict(user_data)
                print(prediction)
                st.write(f"Loan status: {prediction}")
            except json.JSONDecodeError as e:
                st.error(f"Error predicting loan status: {str(e)}")
    else:
        st.error("Please enter valid user information in JSON format.")
    print(single_data_point)

if __name__ == "__main__":
    main()
    
    



