import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from datetime import datetime
import zipfile 
import os 
import tempfile


def load_model():
    #Upload the zipped pickle file
    with zipfile.ZipFile("saved_steps.zip", "r") as zip_ref:
        zip_ref.extractall("saved_steps")

    # Check if the folder exists
    if os.path.exists("saved_steps"):
        # Access the folder and load the CSV file into a DataFrame
        file_path = os.path.join("saved_steps", "saved_steps.pkl")

        # Load the pickle file
        with open(file_path, 'rb') as file:
                data = pickle.load(file)

            # Use the loaded model (example: make a prediction)
        
    #with open('saved_steps.pkl', 'rb') as file:
        #data = pickle.load(file)
    return data

data = load_model()

model_loaded = data["model"]
Subprogram_encoded = data["Subprogram_encoded"]
LoanStatus_encoded=data["LoanStatus_encoded"]
description_encoded=data["description_encoded"]
BusinessType_encoded=data["BusinessType_encoded"]


def show_predict_page():
    st.title('Loan Prediction')
    st.write("""### Please fill the following details""")
    industry=('Retail and Wholesale','Automotive', 'Manufacturing',
              'Healthcare','Construction','Technology and IT','Food and Beverage',
              'Financial Services','Professional Services','Leisure and Hospitality',
              'Agriculture','Textile and Apparel','Media and Entertainment',
              'Logistics and Transportation','Real Estate','Education',
              'Environmental Services','Utilities','Personal Services','Miscellaneous Services',
              'Real Estate and Rental and Leasing','Accommodation and Food Services',
              'Information and Media','Public Administration',
              'Arts, Entertainment, and Recreation' )
    Business_Type=('CORPORATION', 'INDIVIDUAL', 'PARTNERSHIP', 'other')
    Subprogram_or_Guaranty=('Guaranty', 'Contract Guaranty', 
                         'Revolving Line of Credit Exports - Sec. 7(a) (14)', 
                         'International Trade - Sec, 7(a) (16)', 
                         'Seasonal Line of Credit', 'Small General Contractors - Sec. 7(a) (9)', 
                         'Pollution Control Guaranteed Loans - Sec. 7(a)(12)', 
                          'Co-GTY with Import/Export', 'Greenline - Revolving L. of Cred. - Fixed Assets',
                           'Greenline - Revolving L. of Cred. - Current Assets',
                         'Domestic Revolving Line of Credit - Fixed Assets', 'Domestic Revolving Line of Credit - Current Assets', 
                         'Standard Asset Based', 'Small Asset Based', 'FA$TRK (Small Loan Express)','Special Markets Program',
                          'Defense Loans and Technical Assistance, Funded 9/26/95', 'USCAIP Guaranty (NAFTA)',
                           'Y2K Loan', 'Community Express')
    
    GrossApproval=st.number_input(label='Please enter the Gross approved loan amount', min_value=0)
    Subprogram = st.selectbox("Guaranty Type", Subprogram_or_Guaranty)
    TerminMonths=st.number_input(label='Please enter the term of loan in month', min_value=0)
    NAICSDescription=st.selectbox("Industry Type", industry)
    BusinessType= st.selectbox("CBusiness Type", Business_Type)

    
    if st.button('Predict'): 
        try:
            X = np.array([[GrossApproval, Subprogram, TerminMonths, NAICSDescription, BusinessType]]) 
            X[:, 1] = Subprogram_encoded.transform(X[:, 1]) 
            X[:, 3] = description_encoded.transform(X[:, 3]) 
            X[:, 4] = BusinessType_encoded.transform(X[:, 4]) 
            X = X.astype(float) 
            # Make prediction 
            Loan = model_loaded.predict(X)
            #  Display result 
            st.subheader(f"The estimated Loan code is {Loan[0]:.2f}")
            st.write("\n Above code represents") 
            st.write("\n 0: Bad Loan")
            st.write("\n 1: Good Loan") 
        except Exception as e: 
            st.error(f"An error occurred: {e}")
    
