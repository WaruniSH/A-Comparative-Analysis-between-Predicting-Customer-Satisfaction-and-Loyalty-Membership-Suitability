import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from PIL import Image
from sklearn.preprocessing import OneHotEncoder

@st.cache_data
def load_data():
    df = pd.read_csv("restaurant_customer_satisfaction.csv")
    return df

@st.cache_data
def get_fvalue(val):
    feature_dict = {"No": 1, "Yes": 2}
    return feature_dict.get(val, None)

def get_value(val, my_dict):
    return my_dict.get(val, None)

# Sidebar setup
st.sidebar.image('SatisfactionPredictor.png', width=150)  # Adjust the width as needed
app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Prediction', 'Data Visualization', 'About Us', 'Contact Us'])  

if app_mode == 'Home':
    #st.title('PREDICTING CUSTOMER SATISFACTION')
    st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: orange;'>PREDICTING CUSTOMER SATISFACTION</h1>
    </div>
    """, unsafe_allow_html=True)
    st.image('resturant.png')

    #st.subheader("About the App")
    st.write("""
        Welcome to the **Customer Satisfaction Predictor**! 
        
        This app is designed to help restaurant owners and managers understand their customers' satisfaction levels. By using advanced machine learning models, we predict whether a customer is satisfied based on various factors, such as:

        - Age, income, and loyalty program membership
        - Frequency of visits and preferred cuisine
        - Ratings for service, food, and ambiance
        - Dining occasions and times of visit

        With this predictive tool, you can gain insights into your customers' dining experiences and improve their overall satisfaction. Whether you are a restaurant owner seeking to enhance customer loyalty or a curious customer looking to explore how different factors influence satisfaction, this app is for you!

        Start by navigating to the **Prediction** page, or explore visual insights on the **Data Visualization** page. For any inquiries, visit our **Contact Us** page.
    """)
