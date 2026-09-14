import streamlit as st
import pandas as pd
import numpy as np
import joblib

# LOAD MODEL (pipeline)
@st.cache_resource
model = joblib.load("model.pkl")

st.title("Airbnb Price Predictor")

# USER INPUTS (basic only)

room_type = st.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room"])

minimum_nights = st.number_input("Minimum Nights", 1, 365, 1)

number_of_reviews = st.number_input("Number of Reviews", 0, 1000, 10)

availability_365 = st.slider("Availability (days/year)", 0, 365, 200)

host_listings = st.number_input("Host Listings Count", 1, 50, 1)

latitude = st.number_input("Latitude", value=40.7)
longitude = st.number_input("Longitude", value=-73.9)

neighbourhood_group = st.selectbox(
    "Neighbourhood Group",
    ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
)

neighbourhood = st.text_input("Neighbourhood", "Midtown")

# PREDICT BUTTON
if st.button("Predict Price "):

    # CREATE INPUT DATAFRAME
    input_data = pd.DataFrame({
    "room_type": [room_type],
    "minimum_nights": [minimum_nights],
    "number_of_reviews": [number_of_reviews],
    "availability_365": [availability_365],
    "calculated_host_listings_count": [host_listings],
    "latitude": [latitude],
    "longitude": [longitude],
    "neighbourhood_group": [neighbourhood_group]
})

    # PREDICTION
    prediction = model.predict(input_data)
    st.success(f"Estimated Price: ₹{round(prediction[0], 2)} per night")
