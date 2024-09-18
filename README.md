# Digit Predictor

## Project Overview

This project aims to recognize handwritten digits using the MNIST dataset. It explores various machine learning models, including Logistic Regression, Random Forest, and Support Vector Machine (SVM). The models were trained and evaluated for their effectiveness in classifying digits. The Random Forest model proved to be the most accurate, with a test accuracy of 96.91%. Additionally, a web-based application was built using Streamlit to allow real-time digit predictions from user-uploaded images.

## Files

- **Digit_Predictor_.ipynb**: A Jupyter notebook that contains the implementation of multiple machine learning models to classify handwritten digits using the MNIST dataset. The notebook includes data fetching, preprocessing, training, and evaluation of the models.
  
- **digit_predictor_streamlit.py**: A Python script that uses the trained Random Forest model to build a simple web application with Streamlit. Users can upload images of digits, and the app will predict the corresponding number.
  
- **Digit_Predictor_Rapport.pdf**: A detailed report covering the methodology, model training, and results of the digit recognition project. The report includes explanations of the models used, their performance, and recommendations based on the findings.

## Key Analyses

1. **Logistic Regression**: Used to estimate the probability of a digit class. It achieved a validation accuracy of 92.24%.
   
2. **Random Forest**: An ensemble method that combines multiple decision trees. This model outperformed others with a validation accuracy of 97.36% and a test accuracy of 96.91%.

3. **Support Vector Machine (SVM)**: Applied with an RBF kernel for digit classification. SVM achieved a validation accuracy of 96.87%.

4. **Grid Search CV**: Hyperparameter tuning was applied to the Random Forest model to optimize its performance. The final model was used in the Streamlit app.

5. **Streamlit App**: A web-based application built to predict digits from real-world images using the trained Random Forest model.

## How to Use

1. **Jupyter Notebook**:
   - Open the `Digit_Predictor_.ipynb` file in Jupyter Notebook.
   - Execute the cells to train the models and see the results of the evaluation.

2. **Streamlit App**:
   - Run the `digit_predictor_streamlit.py` script using Streamlit.
   - Upload an image of a handwritten digit, and the app will predict the corresponding number.

3. **PDF Report**:
   - Review `Digit_Predictor_Rapport.pdf` for an in-depth analysis of the project, including theoretical explanations of the models, training methodology, and the results.

## Tools & Technologies

- **Python**: The programming language used for model development and data processing.
- **Scikit-learn**: For implementing machine learning models such as Logistic Regression, Random Forest, and SVM.
- **Streamlit**: A framework used to build the interactive web application for digit prediction.
- **MNIST Dataset**: The dataset used for training the models, consisting of 70,000 images of handwritten digits.
