# -*- coding: utf-8 -*-
"""
Created on Sun May 18 22:44:53 2025

@author: luuduc
"""

import numpy as np
import pickle
import streamlit as st

# loading the saved model
# loaded_model = pickle.load(open('trained_model.sav', 'rb'))
model_data = pickle.load(open('trained_model.sav', 'rb'))
loaded_model = model_data['model']
scaler = model_data['scaler']

# creating a function for Prediction

def diabetes_prediction(input_data):
    # Chuyển input thành mảng numpy dạng số thực
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Quan trọng: chuẩn hóa dữ liệu trước khi dự đoán
    std_data = scaler.transform(input_data_reshaped)

    prediction = loaded_model.predict(std_data)
    print(prediction)

    if prediction[0] == 0:
        return 'You are not at risk for diabetes based on the data entered.'
    else:
        return 'You are at high risk of diabetes. Please get checked soon.'
  
    
def main():
    
    
    # giving a title
    st.title('Diabetes Prediction Web App')
    
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person')
    
    # Pregnancies = st.text_input('Number of Pregnancies')
    # Glucose = st.text_input('Glucose Level')
    # BloodPressure = st.text_input('Blood Pressure value')
    # SkinThickness = st.text_input('Skin Thickness value')
    # Insulin = st.text_input('Insulin Level')
    # BMI = st.text_input('BMI value')
    # DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    # Age = st.text_input('Age of the Person')
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
        
    st.success(diagnosis)
    
    
    
if __name__ == '__main__':
    main()
    
    
    
