import joblib
import pandas as pd
import streamlit as st

def percentage(integer_part):
    # scaling 30-100 into 1-100 range to calculate loan percentage
  x = integer_part - 30
  cal = x / 0.707
  loan_percentage = round(cal)
  # scaling 1-100 into 1000-10,000 to calculate loan amount
  ratio = 9000/99
  amount = ratio * (loan_percentage - 1)
  loan_amount = round(amount + 1000)
  # loan_amount =round((loan_percentage / 100) * 10000)
  return loan_amount

# Load the trained model
loaded_pipeline = joblib.load('pipeline2.pkl')

def main():
    st.title("""Loan Prediction App""")
    
    with st.form(key='myform', clear_on_submit=False):
        # Dictionary to store form inputs
        single_data_point = {
            "Educations": st.selectbox("**Educations**", ["---Select Your Education---","Non_Metric", "Matric", "Inter", "Gradutation", "PG"]),
            "Area": st.selectbox("**Area**", ["---Select Your Area---",'Urban','Rural']),
            "Married_Status": st.selectbox("**Married Status**", ["---Select Your Married Status---",'Married','UnMaried']),
            "Criminal_Status" :st.selectbox("**Criminal Status**", ["---Select Your Criminal Status---",'Yes','No']),
            "Employed_type" : st.selectbox("**Employed Type**", ["---Select Your Employement Type---",'Business', 'Salaried', 'Professional', 'Unemployed']),
            "Spouse_Education"  :st.selectbox("Spouse Education", ["---Select Your Spouse Education---","Non_Metric", "Matric", "Inter", "Gradutation", "PG"]),
            "Gender" : st.selectbox("**Gender**", ["---Select Your Gender---",'Male','Female','Transgender']),
            "Age"  :st.selectbox("**Age**", ["---Select Your Age Range---",'Below_20','20-30','30-40','40+']),
            "Dependents"  : st.selectbox("**Dependents**", ["---Select Your Dependents---",'0','1','2','3','4']),
            "City_Type" :st.selectbox("**City Type**", ["---Select Your City Type---",'Metro', 'City', 'Town', 'Village']),
            "Home" : st.selectbox("**Home**", ["---Select Your Home---",'Rented','Owned']),
            "Vehicle" : st.selectbox("**Vehicle**", ["---Select Your Vehicle---",'Wheeler_2','Wheeler_4']),
            "Income" : st.selectbox("**Income**", ["---Select Your Incmone---",'0-10000', '10000-25000', '25000-50000', '50000-75000', '75000+']),
            "Cibil" : st.selectbox("**CIBIL**", ["---Select Your Cibil Score---",'0-300','300-500','500-700','700-900']),
            "Spouse_Income" : st.selectbox("**Spouse Income**", ["---Select Your Spouse Income---",'0-10000', '10000-25000', '25000-50000', '50000-75000', '75000+']),
            "Pin_code" : st.selectbox("**Pin Code**", ["---Select Your Pin Code---",'allowed','not_allowed']),
            "City_Name" : st.selectbox("**City Name**", ["---Select Your City Name---",'allowed','not_allowed']),
            # Add other form inputs similarly
        }

        submit_button = st.form_submit_button(label='Predict Loan Status')

    if submit_button:
        st.info("**Prediction Result**")
        try:
            # Create a DataFrame from the collected user input
            user_data = pd.DataFrame([single_data_point])
            print(user_data)
            prediction = loaded_pipeline.predict(user_data)
            xy = prediction[0]
            abc = int(xy)
            
           
            st.write(f"**Loan status** => {abc}")
            amount = percentage(abc)
            st.write(f"**Loan Amount** => {amount}")
            # if abc >30:
                # amount = percentage(abc)
            #     st.write(f"**Loan Amount** => {amount}")
            # else:
            #    st.write(f"**You are Not Elegible**") 

            # del user_data
        except Exception as e:
            st.error(f"Error predicting loan status: {str(e)}")

        
if __name__ == "__main__":
    main()
    
