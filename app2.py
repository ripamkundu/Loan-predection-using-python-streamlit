import joblib
import pandas as pd
import streamlit as st
import random

def percentage(integer_part):
    # scaling 30-100 into 1-100 range to calculate loan percentage
    x = integer_part - 30
    cal = x / 0.707
    loan_percentage = round(cal)
    # scaling 1-100 into 1000-10,000 to calculate loan amount
    ratio = 9000/99
    amount = ratio * (loan_percentage - 1)
    loan_amount = round(amount + 1000)
    return loan_amount

# Load the trained model
loaded_pipeline = joblib.load('pipeline2.pkl')

def fill_form_and_store_results(num_entries=1700):
    results = []

    for i in range(num_entries):
        single_data_point = {
            "Educations": random.choice(["Non_Metric", "Matric", "Inter", "Graduation", "PG"]),
            "Area": random.choice(['Urban', 'Rural']),
            "Married_Status": random.choice(['Married', 'UnMarried']),
            "Criminal_Status": random.choice(['Yes', 'No']),
            "Employed_type": random.choice(['Business', 'Salaried', 'Professional', 'Unemployed']),
            "Spouse_Education": random.choice(["Non_Metric", "Matric", "Inter", "Graduation", "PG"]),
            "Gender": random.choice(['Male', 'Female', 'Transgender']),
            "Age": random.choice(['Below_20', '20-30', '30-40', '40+']),
            "Dependents": random.choice(['0', '1', '2', '3', '4']),
            "City_Type": random.choice(['Metro', 'City', 'Town', 'Village']),
            "Home": random.choice(['Rented', 'Owned']),
            "Vehicle": random.choice(['Wheeler_2', 'Wheeler_4']),
            "Income": random.choice(['0-10000', '10000-25000', '25000-50000', '50000-75000', '75000+']),
            "Cibil": random.choice(['0-300', '300-500', '500-700', '700-900']),
            "Spouse_Income": random.choice(['0-10000', '10000-25000', '25000-50000', '50000-75000', '75000+']),
            "Pin_code": random.choice(['allowed', 'not_allowed']),
            "City_Name": random.choice(['allowed', 'not_allowed']),
        }
        # Create a DataFrame from the collected user input
        user_data = pd.DataFrame([single_data_point])
        print(user_data)

        try:
            prediction = loaded_pipeline.predict(user_data)
            xy = prediction[0]
            abc = int(xy)

            # Initialize result_entry without "Loan Amount"
            result_entry = {
                "Loan Status": abc,
                # Add other fields as needed
            }

            # Check the condition and assign "Loan Amount" accordingly
            if abc > 30:
                amount = percentage(abc)
                result_entry["Loan Amount"] = amount
            else:
                result_entry["Loan Amount"] = "You Are not Eligible"

            results.append(result_entry)

        except Exception as e:
            print(f"Error predicting loan status: {str(e)}")
        
        results_df = pd.DataFrame(results)
        # Save the results to a CSV file
        results_df.to_csv('loan_prediction_results.csv', index=False)

if __name__ == "__main__":
    fill_form_and_store_results()
