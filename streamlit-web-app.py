# Import packages

import streamlit as st
import pandas as pd
import joblib

# Load model pipeline object

model = joblib.load("model.joblib")

###################
# Build Site
###################

# Title
st.title("Purchase Prediction Model")
st.subheader("Enter customer information and submit for likelihood to purchase")

# Age input Form

age = st.number_input(
    label="01. Enter the Customer's Age",  # Text for the user
    min_value=18,
    max_value=120,
    value=35,  # Prepopulated value
)

# Gender input Form

gender = st.radio(label="02. Enter the Customer's Gender", options=["M", "F"])

# Credit Score inpu Form

credit_score = st.number_input(
    label="03. Enter the Customer's Credit Score",  # Text for the user
    min_value=0,
    max_value=1000,
    value=500,  # Prepopulated value
)

# Submit inputs to model

if st.button("Submit for Prediction"):
    # Store Data to Dataframe

    new_data = pd.DataFrame(
        {"age": [age], "gender": [gender], "credit_score": [credit_score]}
    )
    # Apply Model for Prediction

    pred_proba = model.predict_proba(new_data)[0][1]

    # Output Prediction

    st.subheader(
        f"Based on these customer attributes, our model predicts a purchase probability of {pred_proba:.0%}"
    )
