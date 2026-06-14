import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="skmadhavi21/tourism_model", filename="best_tourism_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism Package Selection Predict App")
st.write("""
This application predicts the likelihood of a tourism package will be selected by customer or not based on its operational parameters.
Please enter the sensor and configuration data below to get a prediction.
""")

passport_mapping = {"No": 0, "Yes": 1}

# User input
Age = st.number_input("Age", min_value=18, max_value=100, value=30)
Occupation = st.selectbox("Customers Occupation",["Salaried", "Small Business","Large Business","Free Lancer" ])
Gender = st.selectbox("Customers Gender", ["Male", "Female"])
MaritalStatus = st.selectbox("Customers Marital Status", ["Married","Single","Divorced"])
passport_selection = st.selectbox("Do customer hold passport?", options=list(passport_mapping.keys()))
Designation = st.selectbox("Customer Designation",["Executive","Manager","Senior Manager","AVP","VP"])
CityTier = st.number_input("City Tier", min_value=1, max_value=3, value=2)
PreferredPropertyStar = st.number_input("Preferred Property Star", min_value=1, max_value=5, value=4)
NumberOfTrips = st.number_input("Number of Trips", min_value=0, max_value=50, value=2)
TypeofContact_mapping = st.selectbox("How the customer was contacted?", ["Company Invited","Self Enquiry"])
ProductPitched = st.selectbox("Which package discuss with customer?", ["Basic","Deluxe","Standard","Super Deluxe","King"])
DurationOfPitch = st.number_input("Duration of Pitch", min_value=5, max_value=100, value=5)
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=5)
NumberOfFollowups = st.number_input("Number of Followups", min_value=0, max_value=6, value=3)


passport_numeric = passport_mapping[passport_selection]

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'Occupation': Occupation,
    'Gender': Gender,
    'MaritalStatus': MaritalStatus,
    'Passport': passport_numeric,
    'Designation': Designation,
    "CityTier": CityTier,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'TypeofContact': TypeofContact_mapping,
    'ProductPitched': ProductPitched,
    'DurationOfPitch': DurationOfPitch,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'NumberOfFollowups': NumberOfFollowups
}])

input_data['Tier_Transformed'] = 4 - input_data['CityTier']


if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "The customer has purchased a package " if prediction == 1 else "The customer has not purchased a package"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
