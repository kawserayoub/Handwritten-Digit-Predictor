import streamlit as st
import numpy as np
import cv2
import joblib

# Loading the trained model
best_rf_model = joblib.load('best_rf_model.pkl')  

# Function to process images
def process_image(img_data):
    img_gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (28, 28))
    img_norm = img_resized / 255.0
    img_inverted = 255.0 - img_norm * 255.0
    threshold_value = np.average(img_inverted)
    img_inverted[img_inverted <= threshold_value] = 0
    img_flat = img_inverted.ravel().reshape(1, -1)
    return img_flat, img_gray

# Streamlit UI
st.title('Handwritten Digit Predictor')

#Displaying the digits
if st.button('Press ME to predict predefined digits'):
    for digit in range(10):
        file_path = f"C:\\numbers\\{digit}.jpg"
        orig_img = cv2.imread(file_path)
        img_flat, img_gray = process_image(orig_img)
        
        st.image(cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB), caption=f'Image: {digit}')
        prediction = best_rf_model.predict(img_flat)
        st.write(f'Predicted Number: {prediction[0]}')

