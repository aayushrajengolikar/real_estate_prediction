import streamlit as st
import joblib
import pandas as pd
import gdown
import os

# -------------------------
# MODEL LOADING FROM GOOGLE DRIVE
# -------------------------

CLASSIFIER_URL = "https://drive.google.com/uc?id=13NYDd36y9amKQD_FQ6oT4Q1Wk_OWlMbQ"
REGRESSOR_URL  = "https://drive.google.com/uc?id=1mDnyWHHy_vukRTCrKD-VfeF5HnEhVXv6"

CLASSIFIER_PATH = "RandomForestClassifier_model.pkl"
REGRESSOR_PATH = "RandomForestRegressor_model.pkl"

# Download if not exists
if not os.path.exists(CLASSIFIER_PATH):
    gdown.download(CLASSIFIER_URL, CLASSIFIER_PATH, quiet=False)
if not os.path.exists(REGRESSOR_PATH):
    gdown.download(REGRESSOR_URL, REGRESSOR_PATH, quiet=False)

# Load models
rf_classifier = joblib.load(CLASSIFIER_PATH)
rf_regressor = joblib.load(REGRESSOR_PATH)

# -------------------------
# STREAMLIT APP INTERFACE
# -------------------------

st.title("Real Estate Price & Investment Prediction App")
st.write("Provide the property details below:")

# State dropdown
states_list = [
    "Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh","Goa","Gujarat",
    "Haryana","Himachal Pradesh","Jharkhand","Karnataka","Kerala","Madhya Pradesh",
    "Maharashtra","Manipur","Meghalaya","Mizoram","Nagaland","Odisha",
    "Punjab","Rajasthan","Sikkim","Tamil Nadu","Telangana","Tripura",
    "Uttar Pradesh","Uttarakhand","West Bengal","Delhi","Jammu & Kashmir"
]
State = st.selectbox("State", states_list)

# City & Locality
City = st.text_input("City")
Locality = st.text_input("Locality")

# Property Type
Property_Type = st.selectbox(
    "Property Type",
    ["Apartment", "Independent House", "Villa", "Plot", "Penthouse"]
)

# BHK
BHK = st.number_input("BHK", min_value=1, max_value=10, value=2)

# Current Price & Price per SqFt
Price_in_Lakhs = st.number_input("Current Price (Lakhs)", min_value=0.0)
Price_per_SqFt = st.number_input("Price per SqFt (Lakhs)", min_value=0.0)

# Year Built
Year_Built = st.number_input("Year Built", min_value=1900, max_value=2025, value=2015)

# Furnished Status
Furnished_Status = st.selectbox("Furnished Status", ["Furnished", "Unfurnished", "Semi-furnished"])

# Floor No & Total Floors
Floor_No = st.number_input("Floor No", min_value=0)
Total_Floors = st.number_input("Total Floors", min_value=1)

# Age of Property
Age_of_Property = st.number_input("Age of Property (years)", min_value=0)

# Nearby Schools & Hospitals
Nearby_Schools = st.number_input("Nearby Schools (within 5km)", min_value=0)
Nearby_Hospitals = st.number_input("Nearby Hospitals (within 5km)", min_value=0)

# Public Transport, Parking, Security
Public_Transport_Accessibility = st.selectbox("Public Transport Accessibility", ["Low", "Medium", "High"])
Parking_Space = st.selectbox("Parking Space", ["Yes", "No"])
Security = st.selectbox("Security", ["Yes", "No"])

# Amenities Category & Facing
Amenities_Category = st.selectbox("Amenities Category", ["Basic", "Standard", "Premium", "Luxury"])
Facing = st.selectbox("Facing", ["East", "West", "North", "South"])

# Owner Type & Availability
Owner_Type = st.selectbox("Owner Type", ["Owner", "Builder", "Broker"])
Availability_Status = st.selectbox("Availability Status", ["Ready to Move", "Under Construction"])

# -------------------------
# PREDICTION BUTTON
# -------------------------

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "State": State,
        "City": City,
        "Locality": Locality,
        "Property_Type": Property_Type,
        "BHK": BHK,
        "Price_in_Lakhs": Price_in_Lakhs,
        "Price_per_SqFt": Price_per_SqFt,
        "Year_Built": Year_Built,
        "Furnished_Status": Furnished_Status,
        "Floor_No": Floor_No,
        "Total_Floors": Total_Floors,
        "Age_of_Property": Age_of_Property,
        "Nearby_Schools": Nearby_Schools,
        "Nearby_Hospitals": Nearby_Hospitals,
        "Public_Transport_Accessibility": Public_Transport_Accessibility,
        "Parking_Space": Parking_Space,
        "Security": Security,
        "Amenities_Category": Amenities_Category,
        "Facing": Facing,
        "Owner_Type": Owner_Type,
        "Availability_Status": Availability_Status
    }])

    # Make predictions
    class_pred = rf_classifier.predict(input_df)[0]
    price_pred = rf_regressor.predict(input_df)[0]

    st.subheader("Predictions:")
    st.write(f"Future Price after 5 years (Lakhs): {price_pred:.2f}")
    st.write("Investment Decision: Good Investment" if class_pred == 1 else "Investment Decision: Not a Good Investment")