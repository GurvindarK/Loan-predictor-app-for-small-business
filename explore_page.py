import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from datetime import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os
import zipfile
def load_data():
        # Unzip the file
    with zipfile.ZipFile("data.zip", "r") as zip_ref:
        zip_ref.extractall("data")

    # Check if the folder exists
    if os.path.exists("data"):
        # Access the folder and load the CSV file into a DataFrame
        csv_file_path = os.path.join("data", "foia-7a-fy1991-fy1999-asof-240930.csv")
        df = pd.read_csv(csv_file_path)
        st.title("Explore Small Business Loan")
         
        st.write("Data's first few rows:") 
        st.write(df.head())
        st.write("Data description")
        st.write(df.describe().T)
        columns_to_drop = ['Program','NAICSCode', 'AsofDate', 'BorrName', 'BorrStreet', 'BorrCity','DeliveryMethod','FranchiseCode', 'FranchiseName',
       'ProjectCounty', 'ProjectState', 'SBADistrictOffice', 'BorrState', 'BorrZip', 'BankName', 'BankFDICNumber', 'BankNCUANumber',
       'BankStreet', 'BankCity', 'BankState', 'BankZip', 'ApprovalDate', 'ApprovalFY',
       'CongressionalDistrict', 'BusinessAge', 'PaidinFullDate', 'ChargeoffDate', 'GrossChargeoffAmount',
       'RevolverStatus', 'JobsSupported','SBAGuaranteedApproval','FirstDisbursementDate', 'SoldSecondMarketInd',  'InitialInterestRate', 'FixedorVariableInterestRate']
        df = df.drop(columns=columns_to_drop, axis=1)
        return df
st.cache_data   


def show_explore_page():
   df = load_data()
   sns.set_theme(style="whitegrid")
   # Distribution of Loan Amounts (GrossApproval)
   
   #Plotting the histogram using Matplotlib and Seaborn
   fig, ax = plt.subplots(figsize=(5, 6))
   sns.histplot(df["GrossApproval"].dropna(), kde=True, bins=50, color="blue", ax=ax)
   ax.set_title("Distribution of Loan Amounts (GrossApproval)", fontsize=10, fontweight='bold')
   ax.set_xlabel("Loan Amount ($)", fontsize=10)
   ax.set_ylabel("Frequency", fontsize=10)

   st.pyplot(fig)

   
   fig1, ax = plt.subplots(figsize=(5, 5))
   loan_status_counts = df["LoanStatus"].value_counts()
   plt.pie(loan_status_counts, labels=loan_status_counts.index, autopct='%1.1f%%', startangle=80, colors=sns.color_palette("pastel"))
   plt.title("Loan Status Proportions", fontsize=10,fontweight='bold')
   st.pyplot(fig1)
   
   
   fig2, ax = plt.subplots(figsize=(5, 5))
   top_business_types = df["BusinessType"].value_counts()
   sns.barplot(x=top_business_types.index, y=top_business_types.values, palette="coolwarm")
   plt.title("Business Types Receiving Loans", fontsize=10,fontweight='bold')
   plt.xlabel("Business Type", fontsize=10)
   plt.ylabel("Number of Loans", fontsize=10)
   plt.xticks(rotation=45, ha="right")
   st.pyplot(fig2)

   
   fig3, ax = plt.subplots(figsize=(5,5))
   sns.histplot(df["TerminMonths"].dropna(), bins=30, color="green", alpha=0.7)
   plt.title("Distribution of Term in Months", fontsize=10,fontweight='bold')
   plt.xlabel("Term (Months)", fontsize=10)
   plt.ylabel("Frequency", fontsize=10)
   plt.grid(axis='y', linestyle='--', alpha=0.7)
   st.pyplot(fig3)
   