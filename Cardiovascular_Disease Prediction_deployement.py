import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the saved model
with open('Randomforest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the columns
columns = ['id', 'age', 'education', 'sex', 'is_smoking', 'cigsPerDay', 'BPMeds',
           'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'BMI',
           'heartRate', 'glucose', 'TenYearCHD', 'Mean_bp']

# Initialize Min-Max Scaler
scaler = MinMaxScaler()

# Function to take user input and make prediction
def predict_chd(input_data):
    input_data_scaled = scaler.fit_transform([input_data])
    prediction = model.predict(input_data_scaled)
    return prediction

# Streamlit app
st.title("CHD Prediction App")

# Create input fields
id = st.number_input('ID', min_value=0)
age = st.slider('Age', min_value=0, max_value=120, value=30)
education = st.selectbox('Education Level', [1, 2, 3, 4])
sex = st.radio('Sex', [0, 1])  # 0: Female, 1: Male
is_smoking = st.radio('Smoking Status', [0, 1])  # 0: No, 1: Yes
cigsPerDay = st.slider('Cigarettes Per Day', min_value=0, max_value=100, value=0)
BPMeds = st.radio('On BP Meds', [0, 1])  # 0: No, 1: Yes
prevalentStroke = st.radio('Prevalent Stroke', [0, 1])  # 0: No, 1: Yes
prevalentHyp = st.radio('Prevalent Hypertension', [0, 1])  # 0: No, 1: Yes
diabetes = st.radio('Diabetes', [0, 1])  # 0: No, 1: Yes
totChol = st.slider('Total Cholesterol', min_value=100, max_value=400, value=200)
BMI = st.slider('BMI', min_value=10.0, max_value=50.0, value=25.0)
heartRate = st.slider('Heart Rate', min_value=30, max_value=200, value=70)
glucose = st.slider('Glucose Level', min_value=40, max_value=300, value=100)
TenYearCHD = st.radio('Ten Year CHD', [0, 1])  # This is typically the target, so you might not include this in input
Mean_bp = st.slider('Mean Blood Pressure', min_value=50, max_value=200, value=100)

# Prepare input data
input_data = [id, age, education, sex, is_smoking, cigsPerDay, BPMeds,
              prevalentStroke, prevalentHyp, diabetes, totChol, BMI,
              heartRate, glucose, TenYearCHD, Mean_bp]

# Make prediction
if st.button('Predict'):
    prediction = predict_chd(input_data)
    st.write(f'Prediction: {"Positive" if prediction[0] == 1 else "Negative"}')
