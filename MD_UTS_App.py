# Import libraries
import streamlit as st
from MD_UTS_OOP import LoanModel
from MD_UTS_Inference import LoanInference

# Streamlit title
st.title('Loan Approval Prediction')

# Input validation
person_age = st.number_input('Age', min_value = 20.0, max_value = 100.0)
person_income = st.number_input('Income', min_value = 0.0)
person_emp_exp = st.number_input('Employment Experience', min_value = 0)
loan_amnt = st.number_input('Loan Amount', min_value = 1000.0)
loan_int_rate = st.number_input('Loan Interest Rate (%)', min_value = 0.0)
loan_percent_income = st.number_input('Loan Percent of Income', min_value = 0.0)
cb_person_cred_hist_length = st.number_input('Credit History Length (years)', min_value = 0)
credit_score = st.number_input('Credit Score', min_value = 300, max_value = 850)
person_gender = st.selectbox('Gender', ['male', 'female'])
person_education = st.selectbox('Education', ['Bachelor', 'Master', 'Associate', 'High School', 'Doctorate'])
person_home_ownership = st.selectbox('Home Ownership', ['OWN', 'RENT', 'MORTGAGE', 'OTHER'])
loan_intent = st.selectbox('Loan Intent', ['PERSONAL', 'VENTURE', 'EDUCATION', 'MEDICAL', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT'])
previous_loan_defaults_on_file = st.selectbox('Previous Defaults', ['Yes', 'No'])

# Input data
input_data = {
    'person_age': person_age,
    'person_income': person_income,
    'person_emp_exp': person_emp_exp,
    'loan_amnt': loan_amnt,
    'loan_int_rate': loan_int_rate,
    'loan_percent_income': loan_percent_income,
    'cb_person_cred_hist_length': cb_person_cred_hist_length,
    'credit_score': credit_score,
    'person_gender': person_gender,
    'person_education': person_education,
    'person_home_ownership': person_home_ownership,
    'loan_intent': loan_intent,
    'previous_loan_defaults_on_file': previous_loan_defaults_on_file
}

# Inference class
model_path = 'loan_model.pkl'
loan_inference = LoanInference(model_path)

# Prediction
result = loan_inference.predict(input_data)

# Results
st.write(f"Prediction: {result['Status']}")
st.write(f"Probability: {result['Confidence']}")