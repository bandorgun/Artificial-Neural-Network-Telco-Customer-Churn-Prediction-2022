import json
import pandas as pd
import pickle
import streamlit as st
import requests


# load column transformer

column_transformer = pickle.load(open("model/column_transformer.pkl", "rb"))


st.title("Customer's Churn Application")
tenure = st.number_input(label='Tenure (Months): ',
    min_value=0.0,
    max_value=1000.0,
    step=1e-0,
    format="%.0f")

TotalCharges = st.number_input(label='Total Charges ($): ',
    min_value=0.0,
    max_value=1000000.0,
    step=1e-0,
    format="%.0g")

MonthlyCharges = st.number_input(label='Monthly Charges ($): ',
    min_value=0.0,
    max_value=100000.0,
    step=1e-0,
    format="%.0g")

Dependents = st.selectbox(label='Dependents: ', options= ('No', 'Yes'))

Partner = st.selectbox(label='Partner: ', options= ('No', 'Yes'))

PaperlessBilling = st.selectbox(label='Paperless Billing: ', options= ('No', 'Yes'))

Contract = st.selectbox(label='Contract: ', options= ('Month-to-month', 'One year', 'Two year'))

InternetService = st.selectbox(label='Internet Service: ', options= ('DSL', 'Fiber optic', 'No'))

OnlineSecurity = st.selectbox(label='Online Security: ', options= ('No', 'Yes', 'No internet service'))

TechSupport = st.selectbox(label='Tech Support: ', options= ('No', 'Yes', 'No internet service'))

DeviceProtection = st.selectbox(label='Device Protection: ', options= ('No', 'Yes', 'No internet service'))

OnlineBackup = st.selectbox(label='Online Backup: ', options= ('Yes', 'No', 'No internet service'))

PaymentMethod = st.selectbox(label='Payment Method: ', options= ('Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)'))



new_data = {'tenure': tenure,
         'TotalCharges': TotalCharges,
         'MonthlyCharges' : MonthlyCharges,
         'Dependents' : Dependents,
         'Partner' : Partner,
         'PaperlessBilling' : PaperlessBilling,
         'Contract' : Contract, 
         'InternetService' : InternetService,
         'OnlineSecurity' : OnlineSecurity, 
         'TechSupport' : TechSupport,  
         'DeviceProtection' : DeviceProtection, 
         'OnlineBackup' : OnlineBackup, 
         'PaymentMethod' : PaymentMethod 
         }
new_data = pd.DataFrame([new_data])

# build feature
new_data = column_transformer.transform(new_data)
new_data = new_data.tolist()

# inference
URL = "https://backend-ml1-ftds013.herokuapp.com/v1/models/churn_model:predict"
param = json.dumps({
        "signature_name":"serving_default",
        "instances":new_data
    })
r = requests.post(URL, data=param)

if r.status_code == 200:
    res = r.json()
    if res['predictions'][0][0] > 0.5:
        st.title("Churn")
    else:
        st.title("Not Churn")
else:
    st.title("Unexpected Error")