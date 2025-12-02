import streamlit as st
import pandas as pd
import joblib
import os
import gdown

# -------------------------
# APP TITLE
# -------------------------
st.title("Real Estate Price & Investment Prediction App")
st.write("Provide the property details below:")

# -------------------------
# MODEL DOWNLOAD AND LOAD
# -------------------------
model_folder = "models"
if not os.path.exists(model_folder):
    os.makedirs(model_folder)

# Google Drive URLs
classifier_url = "https://drive.google.com/uc?id=13NYDd36y9amKQD_FQ6oT4Q1Wk_OWlMbQ"
regressor_url = "https://drive.google.com/uc?id=1mDnyWHHy_vukRTCrKD-VfeF5HnEhVXv6"

classifier_path = os.path.join(model_folder, "RandomForestClassifier_model.pkl")
regressor_path = os.path.join(model_folder, "RandomForestRegressor_model.pkl")

# Download models if not present
if not os.path.exists(classifier_path):
    st.info("Downloading classifier model from Drive...")
    gdown.download(classifier_url, classifier_path, fuzzy=True)

if not os.path.exists(regressor_path):
    st.info("Downloading regressor model from Drive...")
    gdown.download(regressor_url, regressor_path, fuzzy=True)

# Load models
rf_classifier = joblib.load(classifier_path)
rf_regressor = joblib.load(regressor_path)

# -------------------------
# INPUT FIELDS
# -------------------------
states_list = [
    "Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh","Goa","Gujarat",
    "Haryana","Himachal Pradesh","Jharkhand","Karnataka","Kerala","Madhya Pradesh",
    "Maharashtra","Manipur","Meghalaya","Mizoram","Nagaland","Odisha",
    "Punjab","Rajasthan","Sikkim","Tamil Nadu","Telangana","Tripura",
    "Uttar Pradesh","Uttarakhand","West Bengal","Delhi","Jammu & Kashmir"
]

State = st.selectbox("State", states_list)
City = st.text_input("City")
Locality = st.text_input("Locality")
Property_Type = st.selectbox("Property Type", ["Apartment", "Independent House", "Villa", "Plot", "Penthouse"])
BHK = st.number_input("BHK", min_value=1, max_value=10, value=2)
Price_in_Lakhs = st.number_input("Current Price (Lakhs)", min_value=0.0)
Price_per_SqFt = st.number_input("Price per SqFt (Lakhs)", min_value=0.0)
Year_Built = st.number_input("Year Built", min_value=1900, max_value=2025, value=2015)
Furnished_Status = st.selectbox("Furnished Status", ["Furnished", "Unfurnished", "Semi-furnished"])
Floor_No = st.number_input("Floor No", min_value=0)
Total_Floors = st.number_input("Total Floors", min_value=1)
Age_of_Property = st.number_input("Age of Property (years)", min_value=0)
Nearby_Schools = st.number_input("Nearby Schools (within 5km)", min_value=0)
Nearby_Hospitals = st.number_input("Nearby Hospitals (within 5km)", min_value=0)
Public_Transport_Accessibility = st.selectbox("Public Transport Accessibility", ["Low", "Medium", "High"])
Parking_Space = st.selectbox("Parking Space", ["Yes", "No"])
Security = st.selectbox("Security", ["Yes", "No"])
Amenities_Category = st.selectbox("Amenities Category", ["Basic", "Standard", "Premium", "Luxury"])
Facing = st.selectbox("Facing", ["East", "West", "North", "South"])
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

    # Predict
    class_pred = rf_classifier.predict(input_df)[0]
    price_pred = rf_regressor.predict(input_df)[0]

    st.subheader("Predictions:")
    st.write(f"Future Price after 5 years (Lakhs): **{price_pred:.2f}**")
    st.write("Investment Decision: **Good Investment**" if class_pred == 1 else "**Not a Good Investment**")
