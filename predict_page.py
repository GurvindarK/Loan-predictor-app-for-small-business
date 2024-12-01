import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
import zipfile 
import os 
import tempfile

@st.cache_data
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
    # Setting the background color of the app
    st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .centered-title {
        text-align: center;
        color: #8B4513; /* SaddleBrown color that resembles wood */
    }
    .stTextInput > div > div > input {
        background-color: #000000; /* Black background */
        color: #FFFFFF; /* White text color */
        border: 2px solid #FF4500; /* Orange border color */
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    st.markdown("<h1 class='centered-title'>Welcome To Small Business Loan Prediction App</h1>", unsafe_allow_html=True)
    st.image("https://qvcfinancial.com/wp-content/uploads/2021/03/business_loan-2.jpg")
    # Avatar Image using a url
    img1 ="https://www.websparks.sg/wp-content/uploads/2023/02/promotion-cover-scaled.jpg"
    img2 ="https://i.pinimg.com/originals/02/91/9e/02919e06923cc6497af0d321d78fa51f.gif"
       

    # Using the custom class for centered title
    
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
    Business_Type=('CORPORATION', 'INDIVIDUAL', 'PARTNERSHIP', 'OTHER')
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
    
    GrossApproval=st.number_input(label='Please enter the loan amount', min_value=1000)
    Subprogram = st.selectbox("Guaranty Type", Subprogram_or_Guaranty)
    TerminMonths=st.number_input(label='Please enter the term of loan in months',min_value=1)
    NAICSDescription=st.selectbox("Industry Type", industry)
    BusinessType= st.selectbox("Business Type", Business_Type)

    
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
            if Loan[0]==0: 
                 st.image(img2)
                 st.subheader("OHH its a bad loan, don't get upset,Predictions aren't always accurate, and there's always a chance for things to turn around. Keep your spirits up and stay hopeful!")
                 

                 st.markdown(
                            """
                            <p style='color: #FF4500; font-weight: bold;'>Note: Changing the type of Guaranty can enhance your chances of getting the loan approved.</p>
                            """,
                            unsafe_allow_html=True
                            )

                 st.markdown("<h1 style='text-align: center; color: #800080;'>BEST OF LUCK!</h1>", unsafe_allow_html=True)
                 
            else:
                 st.image(img1)
                 st.subheader("Congratulations, you have high chances to get loan approved")
                 
            
            
                
            
        except Exception as e: 
            st.error(f"An error occurred: {e}")
show_predict_page()
    
