import streamlit as st
import pandas as pd
import pickle

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# ---------------------------------
# Load Model
# ---------------------------------
@st.cache_resource
def load_model():
    with open("model/car_price_pipeline.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

# ---------------------------------
# App Title
# ---------------------------------
st.title("🚗 Car Price Prediction System")
st.markdown(
    "Predict the **market price of a car** using a trained Machine Learning model."
)

# ---------------------------------
# Sidebar Inputs
# ---------------------------------
st.sidebar.header("🔧 Car Specifications")

Brand = st.sidebar.selectbox("Brand", [
    "Toyota", "Honda", "Ford", "BMW", "Mercedes", "Hyundai", "Kia"
])

Condition = st.sidebar.selectbox("Condition", [
    "New", "Used", "Certified"
])

FuelType = st.sidebar.selectbox("Fuel Type", [
    "Petrol", "Diesel", "Hybrid", "Electric"
])

Transmission = st.sidebar.selectbox("Transmission", [
    "Manual", "Automatic"
])

DriveType = st.sidebar.selectbox("Drive Type", [
    "FWD", "RWD", "AWD"
])

BodyType = st.sidebar.selectbox("Body Type", [
    "Sedan", "SUV", "Hatchback", "Coupe"
])

AccidentHistory = st.sidebar.selectbox("Accident History", [
    "Yes", "No"
])

Insurance = st.sidebar.selectbox("Insurance", [
    "Yes", "No"
])

RegistrationStatus = st.sidebar.selectbox("Registration Status", [
    "Registered", "Unregistered"
])

st.sidebar.markdown("---")
st.sidebar.header("📊 Numeric Details")

CarAge = st.sidebar.slider("Car Age (years)", 0, 30, 3)
Mileage = st.sidebar.number_input("Mileage (km)", min_value=0, value=30000)
EngineSize = st.sidebar.slider("Engine Size (L)", 0.8, 6.0, 1.5)
Horsepower = st.sidebar.slider("Horsepower", 50, 1000, 120)
Torque = st.sidebar.slider("Torque", 50, 1500, 150)
Doors = st.sidebar.selectbox("Doors", [2, 3, 4, 5])
Seats = st.sidebar.selectbox("Seats", [2, 4, 5, 7])

FuelEfficiency = st.sidebar.slider(
    "Fuel Efficiency (L/100km)", 2.0, 20.0, 6.5
)

# ---------------------------------
# Prediction
# ---------------------------------
input_data = pd.DataFrame([{
    "Brand": Brand,
    "CarAge": CarAge,
    "Condition": Condition,
    "Mileage(km)": Mileage,
    "EngineSize(L)": EngineSize,
    "FuelType": FuelType,
    "Horsepower": Horsepower,
    "Torque": Torque,
    "Transmission": Transmission,
    "DriveType": DriveType,
    "BodyType": BodyType,
    "Doors": Doors,
    "Seats": Seats,
    "AccidentHistory": AccidentHistory,
    "Insurance": Insurance,
    "RegistrationStatus": RegistrationStatus,
    "FuelEfficiency(L/100km)": FuelEfficiency
}])

st.markdown("### 📈 Prediction Result")

if st.button("Predict Car Price 🚀"):
    prediction = model.predict(input_data)[0]

    st.success(f"💰 Estimated Car Price: **${prediction:,.2f}**")
