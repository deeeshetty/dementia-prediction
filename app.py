import streamlit as st
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin 
import pickle


def replace_values(arr):
    unique_values = np.unique(arr)
    replacement_dict = {'Unknown': 0, 'Sedentary': 1, 'Mild Activity': 2, 'Moderate Activity': 3}
    replacement_dict_full = {val: replacement_dict.get(val, val) for val in unique_values}
    return np.vectorize(replacement_dict_full.get)(arr)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        column_names = ['AlcoholLevel', 'HeartRate', 'BloodOxygenLevel', 'BodyTemperature',
                        'Weight', 'MRI_Delay', 'Age', 'Family_History', 'Smoking_Status',
                        'APOE_ε4', 'Physical_Activity', 'Depression_Status',
                        'Cognitive_Test_Scores', 'Chronic_Health_Conditions']

        column_indices = {name: idx for idx, name in enumerate(column_names)}
        num_rows = X.shape[0]
        new_features = np.zeros((num_rows, X.shape[1] + 8))  

        new_features[:, :X.shape[1]] = X

        idx_Cognitive_Test_Scores = column_indices['Cognitive_Test_Scores']
        idx_AlcoholLevel = column_indices['AlcoholLevel']
        idx_MRI_Delay = column_indices['MRI_Delay']
        idx_BodyTemperature = column_indices['BodyTemperature']
        idx_Age = column_indices['Age']
        idx_Weight = column_indices['Weight']
        idx_Physical_Activity = column_indices['Physical_Activity']
        idx_Smoking_Status = column_indices['Smoking_Status']
        idx_BloodOxygenLevel = column_indices['BloodOxygenLevel']
        idx_Family_History = column_indices['Family_History']

        new_features[:, X.shape[1]] = X[:, idx_Cognitive_Test_Scores] ** 2
        new_features[:, X.shape[1] + 1] = X[:, idx_AlcoholLevel] ** 2
        new_features[:, X.shape[1] + 2] = np.sqrt(np.clip(X[:, idx_MRI_Delay], a_min=0, a_max=None))
        new_features[:, X.shape[1] + 3] = X[:, idx_BodyTemperature] / X[:, idx_Age]
        new_features[:, X.shape[1] + 4] = X[:, idx_BodyTemperature] / X[:, idx_Weight]
        new_features[:, X.shape[1] + 5] = X[:, idx_Physical_Activity] / X[:, idx_Age]
        new_features[:, X.shape[1] + 6] = X[:, idx_Smoking_Status] * X[:, idx_BloodOxygenLevel]
        new_features[:, X.shape[1] + 7] = X[:, idx_Family_History] * X[:, idx_Weight]

        new_features = new_features[:, X.shape[1]:]

        return new_features
# Define the function used in the model
def replace_ordinal_values(arr):
    replacement_dict = {'Unknown': 0, 'Sedentary': 1, 'Mild Activity': 2, 'Moderate Activity': 3}
    unique_values = np.unique(arr)
    replacement_dict_full = {val: replacement_dict.get(val, val) for val in unique_values}
    return np.vectorize(replacement_dict_full.get)(arr)


# Load the trained model from a .pkl file
model_path = 'log_reg_best_model_new.pkl'

try:
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# Apply custom CSS for padding and font size
st.markdown("""
    <style>
    .stTabs [role="tab"] {
        padding-top: 20px;  /* Add padding on top */
    }
    </style>
    """, unsafe_allow_html=True)

# Feature names
feature_names = ['AlcoholLevel', 'HeartRate', 'BloodOxygenLevel', 'BodyTemperature',
                 'Weight', 'MRI_Delay', 'Age', 'Family_History', 'Smoking_Status',
                 'APOE_ε4', 'Physical_Activity', 'Depression_Status',
                 'Cognitive_Test_Scores', 'Chronic_Health_Conditions']

CSS = open("assets/css/styles.css", 'r').read()
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

# Create a banner section using the .top-section class
st.markdown('<div class="top-section"></div>', unsafe_allow_html=True)

# # Streamlit app
# st.title('Dementia Prediction')

# Define menu options
menu = ['Introduction', 'Dementia Prediction']

st.markdown("""
    <style>
    .stTabs [data-baseweb="tab"] {
        font-size: 28px;  /* Increase the font size */
        padding-top: 20px;  /* Add padding on top */
    }
    </style>
    """, unsafe_allow_html=True)

# Display menu
tab1,tab2 = st.tabs(menu)

# Handle menu selections
with tab1:
    st.markdown(
        """
        <div style="color:#000000;">
        <p><strong>Welcome to the Dementia Prediction App!</strong></p>
        <p>This project focuses on predicting dementia onset by using various health and lifestyle factors. We are targeting healthcare professionals who are assessing their patients' risk for dementia. Early prediction can lead to early intervention, significantly slowing the disease’s progression and improving the patient’s quality of life. Furthermore, it can inform public health strategies, helping healthcare systems and policymakers to allocate resources effectively, design targeted prevention programs, and monitor disease trends over time.</p>
        </div>
        """, unsafe_allow_html=True
    )
with tab2:
    st.write('Enter patient details:')
    
    # Input form for dementia prediction
    form = st.form(key='prediction_form')
    inputs = []

    df = pd.read_csv("dementia_patients_health_data.csv")

    for feature in feature_names:
        if feature in ['Family_History', 'Smoking_Status', 'APOE_ε4', 'Depression_Status', 'Chronic_Health_Conditions', 'Physical_Activity']:
            unique_values = df[feature].unique()
            unique_values = [str(value) for value in unique_values]
            input_value = form.selectbox(f'{feature}:', unique_values)
        else:
            # Handle numerical inputs
            input_value = form.number_input(f'{feature}:', value=0.0)
        inputs.append(input_value)

    submit_button = form.form_submit_button('Predict')

    if submit_button:
        # Prepare input data
        input_data = pd.DataFrame([inputs], columns=feature_names)
        
        # Make prediction directly with the loaded model (includes preprocessing)
        prediction = loaded_model.predict(input_data)[0]
        
        # Display prediction with custom styling
        if prediction == 0:
                st.markdown("""
        <div style="background-color: #d4edda; color: #155724; padding: 15px; border-radius: 5px; font-size: 24px;">
            🎉 No Risk of Dementia
        </div>
        """, unsafe_allow_html=True)
        else:
                st.markdown("""
        <div style="background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px; font-size: 24px;">
            ⚠️ Risk of Dementia
        </div>
        """, unsafe_allow_html=True)


        




