
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler (Replace with actual file paths)
model = load_model('path_to_your_model.h5')  # Replace with your model path
scaler = load_model('path_to_your_scaler.h5')  # Replace with your scaler path

# Streamlit App Layout
st.title('PCOS Diagnosis Prediction')
st.write('Enter the following details to predict if you have PCOS:')

# Input fields for the user
age = st.number_input('Age', min_value=0, max_value=100, value=30)
bmi = st.number_input('BMI', min_value=10, max_value=50, value=24)
menstrual_irregularity = st.selectbox('Menstrual Irregularity', options=[0, 1], index=0)
testosterone_level = st.number_input('Testosterone Level (ng/dL)', min_value=0.0, max_value=200.0, value=20.0)
antral_follicle_count = st.number_input('Antral Follicle Count', min_value=0, max_value=50, value=15)

# Create input array for prediction
user_input = np.array([age, bmi, menstrual_irregularity, testosterone_level, antral_follicle_count]).reshape(1, -1)

# Scale input data
scaled_input = scaler.transform(user_input)

# Predict the result
if st.button('Predict PCOS Diagnosis'):
    prediction = model.predict(scaled_input)
    result = 'PCOS Detected' if prediction[0] > 0.5 else 'No PCOS Detected'
    st.write(f'The predicted result is: {result}')
